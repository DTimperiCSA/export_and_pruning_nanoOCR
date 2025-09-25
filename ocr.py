import time
import datetime
import os
import psutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm  # Barra di progresso
from paths import *  # your folder with BELGRADO_OCR, MODEL_ID, FP16_MODEL_PATH
from postprocess import * 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 2121000  # max pixels allowed as area

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Parametri per white detection
WHITE_THRESHOLD = 250        # pixel >= questo valore sono "bianchi"
FRACTION_REQUIRED = 0.999    # almeno 99.9% di pixel bianchi

def get_memory_usage():
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    gpu_mem = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2) if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem


def is_white_image(image_path):
    """Ritorna True se l'immagine è praticamente tutta bianca"""
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        mask = np.all(arr >= WHITE_THRESHOLD, axis=-1)
        fraction_white = mask.sum() / mask.size
        return fraction_white >= FRACTION_REQUIRED, fraction_white
    except Exception as e:
        return False, 0.0


def clean_ocr_output_dynamic(output_text):
    """Remove intro lines, keep only OCR text"""
    lines = output_text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        lower = line.strip().lower()
        if lower in ["system", "user", "assistant"]:
            start_idx = i + 1
        else:
            if start_idx > 0:
                start_idx = i
                break
    while start_idx < len(lines) and lines[start_idx].strip() == "":
        start_idx += 1
    return "\n".join(lines[start_idx:]).strip()


def ocr_inference(model, processor, image_path, language="Italian", max_new_tokens=4096):
    cpu_before, gpu_before = get_memory_usage()
    start = time.time()

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    area = w * h

    if area > THRESHOLD:
        scale = (THRESHOLD / area) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h))

    prompt = (
        f"Extract only the plain text from the image, without any formatting, tags, "
        f"captions, page numbers, or watermarks. Do not translate anything. "
        f"Do not include HTML, LaTeX, descriptions, or metadata. "
        f"The document is in {language}. Return only the OCR text, nothing else."
    )

    messages = [
        {"role": "system", "content": "You are a helpful OCR engine."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    repetition_penalty=1.15, do_sample=False)
        ocr_text = processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

    cpu_after, gpu_after = get_memory_usage()
    elapsed = time.time() - start
    output_tokens = output_ids.shape[-1]
    mem_usage = (cpu_before, cpu_after, gpu_before, gpu_after)

    return ocr_text, elapsed, output_tokens, mem_usage


def save_text(output_path, text, postprocess=True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text.strip())
    if postprocess and text.strip():
        remove_before_headline(output_path, output_path)


def collect_images(root_path):
    image_paths = []

    for folder in Path(root_path).rglob("*"):
        if not folder.is_dir():
            continue

        # Conta immagini e txt nella cartella corrente
        images = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
        txts = [p for p in folder.iterdir() if p.suffix.lower() == ".txt"]

        if len(images) == 0:
            continue  # cartella senza immagini
        if len(images) == len(txts):
            continue  # cartella già processata

        # Aggiungi tutte le immagini non ancora processate
        image_paths.extend(images)

    return image_paths


def process_images(image_paths, model, processor):
    total_time = 0
    total_tokens = 0
    white_images_count = 0
    min_time = float('inf')
    max_time = 0
    ocr_done = 0
    ocr_skipped = 0
    results = []

    for img_path in tqdm(image_paths, desc="Processing images", unit="image"):
        relative_path = img_path.relative_to(BELGRADO_OCR)
        output_name = relative_path.stem + ".txt"
        output_path = BELGRADO_OCR / relative_path.parent / output_name

        is_white, frac = is_white_image(img_path)
        if is_white:
            white_images_count += 1
            save_text(output_path, "", postprocess=False)
            results.append({
                "image": str(relative_path),
                "latency": 0,
                "tokens": 0,
                "cpu_mem_before": 0,
                "cpu_mem_after": 0,
                "gpu_mem_before": 0,
                "gpu_mem_after": 0,
            })
            ocr_skipped += 1
        else:
            if output_path.exists():
                results.append({
                    "image": str(relative_path),
                    "latency": 0,
                    "tokens": 0,
                    "cpu_mem_before": 0,
                    "cpu_mem_after": 0,
                    "gpu_mem_before": 0,
                    "gpu_mem_after": 0,
                })
                ocr_skipped += 1
            else:
                # Run OCR
                ocr_text, latency, tokens, mem_usage = ocr_inference(model, processor, img_path)
                save_text(output_path, ocr_text)

                total_time += latency
                total_tokens += tokens
                min_time = min(min_time, latency)
                max_time = max(max_time, latency)

                results.append({
                    "image": str(relative_path),
                    "latency": latency,
                    "tokens": tokens,
                    "cpu_mem_before": mem_usage[0],
                    "cpu_mem_after": mem_usage[1],
                    "gpu_mem_before": mem_usage[2],
                    "gpu_mem_after": mem_usage[3],
                })
                ocr_done += 1

    # Performance summary dopo ogni cartella
    num_images = len(image_paths)
    avg_time = total_time / ocr_done if ocr_done > 0 else 0
    print(f"\n🎯 Performance summary per cartella:")
    print(f"- Total images: {num_images}")
    print(f"- OCR done: {ocr_done}")
    print(f"- OCR skipped: {ocr_skipped}")
    print(f"- White images: {white_images_count}")
    print(f"- Avg OCR time: {avg_time:.2f}s")
    print(f"- Min OCR time: {min_time:.2f}s")
    print(f"- Max OCR time: {max_time:.2f}s")
    print(f"- Total OCR tokens: {total_tokens}")

    return results


def main():
    print(f"🚀 Loading processor and model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)

    print(f"📂 Loading FP16 weights from {FP16_MODEL_PATH}")
    state_dict = torch.load(FP16_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.half().to(DEVICE).eval()
    print(f"✅ Model ready on {DEVICE} (dtype={next(model.parameters()).dtype})")

    image_paths = collect_images(BELGRADO_OCR)
    if not image_paths:
        print(f"⚠️ No images found in {BELGRADO_OCR}")
        return

    print(f"📸 Found {len(image_paths)} images in {BELGRADO_OCR}")

    results = process_images(image_paths, model, processor)

    # Salva i risultati se necessario
    # csv_path = Path("results") / "ocr_performance.csv"
    # print(f"📊 Saving performance report to {csv_path}")
    # save_results_to_csv(results, csv_path)


if __name__ == "__main__":
    while(True):
        print(f"Run iniziata alle {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main()
        print(f"Run finita alle {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Riavvio processamento...")
        print("\n")
