from pathlib import Path
import torch
import json
import traceback
import psutil
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from pytorch_inference import ocr_inference, save_text
from paths import *
from postprocess import *
import gc

# === CONFIG ===
SELECTED_IMAGES = [
    r"C:\Demo\fascicoli_del_personale\csaFPApp\test\temp\002346_GIANNINO ANTONIETTA\page_00000018_desk.png",
]

RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

def get_memory_usage():
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    gpu_mem = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2) if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem


def load_model_fp16(model_dir: Path, device: torch.device):
    print(f"🚀 Loading processor and model from: {model_dir} on {device}")
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir))

    print(f"📂 Loading FP16 weights from {FP16_MODEL_PATH}")
    state_dict = torch.load(FP16_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model = model.half()
    model.to(device)
    model.eval()
    print(f"✅ Model ready on {device} (dtype={next(model.parameters()).dtype})")
    return model, processor

def ocr_inference(model, processor, image_path, language="Italian", scale=None, max_new_tokens=4096):
    import time
    cpu_before, gpu_before = get_memory_usage()
    print(f"[DEBUG] CPU before: {cpu_before:.2f} MB, GPU before: {gpu_before:.2f} MB")
    start = time.time()

    image = Image.open(image_path).convert("RGB")
    print(f"[DEBUG] Original image size: {image.size}")
    
    if scale:
        w, h = image.size
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size)
        print(f"[DEBUG] Resized image to {scale*100:.0f}% -> {new_size}")

    prompt = f"Extract only the plain text from the image, without any formatting, tags, captions, page numbers, or watermarks. Do not translate anything. Do not include HTML, LaTeX, descriptions, or metadata. The document is in {language}. Return only the OCR text, nothing else.  "

    messages = [
        {"role": "system", "content": "You are a helpful OCR engine."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]}
    ]

    print(f"[DEBUG] Applying processor chat template")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
            
    with torch.no_grad():
        print(f"[DEBUG] Generating OCR output...")
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        ocr_text = processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()
    
    latency = time.time() - start
    cpu_after, gpu_after = get_memory_usage()
    print(f"[DEBUG] CPU after: {cpu_after:.2f} MB, GPU after: {gpu_after:.2f} MB, latency: {latency:.2f}s")

    mem_usage = (cpu_before, cpu_after, gpu_before, gpu_after)
    print(f"[DEBUG] Tokens generated: {output_ids.shape[-1]}")

    return ocr_text, latency, output_ids.shape[-1], mem_usage


def main():
    all_performances = []

    # Carica modello e processor
    model, processor = load_model_fp16(MODEL_ID, DEVICE)

    for img_path_str in SELECTED_IMAGES:
        img_path = Path(img_path_str)

        print(f"\n=== OCR su {img_path.name} con resize 40%-120% ===")

        for perc in range(40, 121):  # 40% -> 120%
            scale = perc / 100

            print(f"\n[DEBUG] Processing {img_path.name} at {perc}% scale")
            try:
                ocr_text, latency, tokens, mem_usage = ocr_inference(
                    model=model,
                    processor=processor,
                    image_path=img_path,
                    scale=scale
                )
            except Exception as e:
                print(f"[ERROR] OCR failed for {img_path.name} at {perc}%: {e}")
                print(traceback.format_exc())
                continue

            out_name = f"{img_path.stem}_fp16_resized_{perc}.txt"
            out_path = RESULTS_PATH / out_name
            save_text(out_path, ocr_text)
            print(f"[DEBUG] Saved OCR text to {out_path}")

            all_performances.append({
                "image": img_path.name,
                "percentage": perc,
                "latency": latency,
                "tokens_generated": tokens,
                "cpu_mem_before": mem_usage[0],
                "cpu_mem_after": mem_usage[1],
                "gpu_mem_before": mem_usage[2],
                "gpu_mem_after": mem_usage[3],
                "output_file": str(out_path)
            })

    # Pulizia memoria
    print("[DEBUG] Cleaning up memory")
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    # Salva sommario delle performance
    summary_path = RESULTS_PATH / "ocr_performance_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_performances, f, indent=2)

    print(f"\n✅ OCR completato. Report salvato in {summary_path}")


if __name__ == "__main__":
    main()
