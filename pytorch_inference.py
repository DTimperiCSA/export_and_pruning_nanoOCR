import time
import os
import psutil
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import csv
from paths import OCR_PATHS  # your folder with images

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "your_model_id_here"  # set your model id or path here
FP16_MODEL_PATH = Path("exported_models/nanoocr_fp16.pth")

def get_memory_usage():
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    gpu_mem = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2) if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem

def ocr_inference(model, processor, image_path, resize=True, max_new_tokens=4096):
    image = Image.open(image_path).convert("RGB")
    if resize:
        width, height = image.size
        image = image.resize((width // 2, height // 2))

    prompt = "Extract text from document naturally, return tables in HTML, equations in LaTeX."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]}
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    cpu_before, gpu_before = get_memory_usage()
    with torch.no_grad():
        start = time.time()
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        latency = time.time() - start
        output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    cpu_after, gpu_after = get_memory_usage()

    mem_usage = (cpu_before, cpu_after, gpu_before, gpu_after)
    return output_text, latency, output_ids.shape[-1], mem_usage

def save_text(output_path, text):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text.strip())

def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
    state_dict = torch.load(FP16_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.half()
    model.to(DEVICE)
    model.eval()

    # Find all desk images
    image_paths = list(Path(OCR_PATHS).glob("*_desk.png"))
    if not image_paths:
        print(f"No images matching '*_desk.png' found in {OCR_PATHS}")
        return

    results = []

    for img_path in image_paths:
        filename = img_path.stem.replace("_desk", "")
        
        # OCR with resize
        ocr_text_resized, latency_resized, tokens_resized, mem_resized = ocr_inference(model, processor, img_path, resize=True)
        output_resized_path = img_path.parent / f"{filename}_fp16_resized.txt"
        save_text(output_resized_path, ocr_text_resized)

        # OCR without resize
        ocr_text_noresize, latency_noresize, tokens_noresize, mem_noresize = ocr_inference(model, processor, img_path, resize=False)
        output_noresize_path = img_path.parent / f"{filename}_fp16_noresize.txt"
        save_text(output_noresize_path, ocr_text_noresize)

        print(f"Processed {img_path.name}: resized {latency_resized:.2f}s, noresize {latency_noresize:.2f}s")

        results.append({
            "image": img_path.name,
            "resized_latency": latency_resized,
            "resized_tokens": tokens_resized,
            "resized_cpu_mem_before": mem_resized[0],
            "resized_cpu_mem_after": mem_resized[1],
            "resized_gpu_mem_before": mem_resized[2],
            "resized_gpu_mem_after": mem_resized[3],
            "noresize_latency": latency_noresize,
            "noresize_tokens": tokens_noresize,
            "noresize_cpu_mem_before": mem_noresize[0],
            "noresize_cpu_mem_after": mem_noresize[1],
            "noresize_gpu_mem_before": mem_noresize[2],
            "noresize_gpu_mem_after": mem_noresize[3],
        })

    # Save performance CSV
    csv_path = Path(OCR_PATHS) / "ocr_fp16_performance.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Performance report saved to: {csv_path}")

if __name__ == "__main__":
    main()
