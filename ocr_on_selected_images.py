import time
import os
import psutil
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import csv
from paths import *  # your folder with images
from postprocess import *

# === CONFIG ===
SELECTED_IMAGES = [
    r"C:\Demo\fascicoli_del_personale\csaFPApp\test\temp\002346_GIANNINO ANTONIETTA\page_00000029_desk.png",
    r"C:\Demo\fascicoli_del_personale\csaFPApp\test\temp\002346_GIANNINO ANTONIETTA\page_00000052_desk.png",
    r"C:\Demo\fascicoli_del_personale\csaFPApp\test\temp\002346_GIANNINO ANTONIETTA\page_00000049_desk.png",
    r"C:\Demo\fascicoli_del_personale\csaFPApp\test\temp\002346_GIANNINO ANTONIETTA\page_00000046_desk.png",
    r"C:\Demo\fascicoli_del_personale\csaFPApp\test\temp\002346_GIANNINO ANTONIETTA\page_00000044_desk.png",
    r"C:\Demo\fascicoli_del_personale\csaFPApp\test\temp\002346_GIANNINO ANTONIETTA\page_00000082_desk.png",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 2121000  # max pixels allowed as area
MAX_TOKENS = 8000

"""
Ottenuto dalle performance sull OCR su questo PC con:

[DEBUG] Processing page_00000018_desk.png at 50% scale
[DEBUG] CPU before: 23351.31 MB, GPU before: 7283.49 MB
[DEBUG] Original image size: (2467, 3443)
[DEBUG] Resized image to 50% -> (1233, 1721)
[DEBUG] Applying processor chat template
[DEBUG] Generating OCR output...
[DEBUG] CPU after: 23347.18 MB, GPU after: 7331.72 MB, latency: 13.84s
[DEBUG] Tokens generated: 3087
💾 Saving OCR result to results\page_00000018_desk_fp16_resized_50.txt
Processed file saved to results\page_00000018_desk_fp16_resized_50.txt
[DEBUG] Saved OCR text to results\page_00000018_desk_fp16_resized_50.txt
"""

def safe_ocr(model, processor, image=None, image_path=None, language="Italian", max_context=MAX_TOKENS, max_new_tokens=None):
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
    w, h = image.size
    area = w * h
    print(f"📐 Original image: {w}x{h} ({area} pixels)")

    # --- Scale if above pixel threshold ---
    if area > THRESHOLD:
        scale = (THRESHOLD / area) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h))
        print(f"🔧 Image scaled by {scale:.3f} -> {new_w}x{new_h} ({new_w*new_h} pixels)")
    else:
        print("✅ Image within threshold, no scaling applied")

    # --- Build OCR prompt ---
    prompt = (
        f"Extract only the plain text from the image, without any formatting, tags, "
        f"captions, page numbers, or watermarks. Do not translate anything. "
        f"Do not include HTML, LaTeX, descriptions, or metadata. "
        f"The document is in {language}. Return only the OCR text, nothing else."
    )
    messages = [
        {"role": "system", "content": "You are a helpful OCR engine."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"} if image_path else {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    # Estimate prompt length
    dummy_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    dummy_inputs = processor(text=[dummy_text], images=[image], return_tensors="pt", padding=True)
    prompt_len = dummy_inputs["input_ids"].shape[-1]

    available_tokens = max_context - prompt_len

    # --- Run actual inference ---
    cpu_before, gpu_before = get_memory_usage()
    start = time.time()

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

    # Adjust generation parameters
    gen_kwargs = {
        "do_sample": False,
        "max_length": max_context,  # total limit
    }
    if max_new_tokens:
        gen_kwargs["max_new_tokens"] = min(max_new_tokens, available_tokens)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
        ocr_text = processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

    
    output_tokens = output_ids.shape[-1]


    # --- Controllo token e split ---
    if output_tokens > available_tokens:
        print(f"⚠️ Page too dense for {output_tokens} tokens more than {available_tokens}. Splitting image...")
        mid = h // 2
        top = image.crop((0, 0, w, mid))
        bottom = image.crop((0, mid, w, h))
        # Ricorsione passando le slice
        top_text, *_ = safe_ocr(model, processor, image=top, language=language, max_context=max_context)
        bottom_text, *_ = safe_ocr(model, processor, image=bottom, language=language, max_context=max_context)
        return top_text + "\n" + bottom_text, 0, 0, (0, 0, 0, 0)


def get_memory_usage():
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    gpu_mem = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2) if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem


def clean_ocr_output_dynamic(output_text):
    """
    Dynamically removes the system/user/assistant intro lines from
    model output, leaving only the OCR text.
    """
    lines = output_text.splitlines()
    start_idx = 0

    # look for the line after the last known role marker
    for i, line in enumerate(lines):
        lower = line.strip().lower()
        if lower in ["system", "user", "assistant"]:
            start_idx = i + 1
        else:
            # first line that is not a role marker is likely OCR start
            if start_idx > 0:
                start_idx = i
                break

    # Remove blank lines before OCR
    while start_idx < len(lines) and lines[start_idx].strip() == "":
        start_idx += 1

    return "\n".join(lines[start_idx:]).strip()


def ocr_inference(model, processor, image_path, language="Italian", max_new_tokens=MAX_TOKENS):
    cpu_before, gpu_before = get_memory_usage()
    start = time.time()

    # --- Load image ---
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    area = w * h
    print(f"📐 Original image: {w}x{h} ({area} pixels)")

    # --- Scale image if needed ---
    if area > THRESHOLD:
        scale = (THRESHOLD / area) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h))
        print(f"🔧 Image scaled by {scale:.3f} -> {new_w}x{new_h} ({new_w*new_h} pixels)")
    else:
        print("✅ Image within threshold, no scaling applied")

    # --- Prepare OCR prompt ---
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

    # --- Perform inference ---
    print("⏳ Running OCR inference...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
            
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.15, do_sample=False)
        ocr_text = processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

    cpu_after, gpu_after = get_memory_usage()
    elapsed = time.time() - start
    output_tokens = output_ids.shape[-1]
    mem_usage = (cpu_before, cpu_after, gpu_before, gpu_after)

    print(f"✅ OCR done in {elapsed:.2f}s")
    print(f"💻 CPU memory: {cpu_before:.1f}MB -> {cpu_after:.1f}MB | "
          f"GPU memory: {gpu_before:.1f}MB -> {gpu_after:.1f}MB")
    print(f"📝 Output tokens (approx): {output_tokens}")

    return ocr_text, elapsed, output_tokens, mem_usage


def save_text(output_path, text):
    print(f"💾 Saving OCR result to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text.strip())
    remove_before_headline(output_path, output_path)  # post-process to remove intro lines


def main(use_safe=True):
    print(f"🚀 Loading processor and model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)

    print(f"📂 Loading FP16 weights from {FP16_MODEL_PATH}")
    state_dict = torch.load(FP16_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    print("✅ Weights loaded")

    model = model.half()
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model ready on {DEVICE} (dtype={next(model.parameters()).dtype})")

    # --- select images (constant list or directory glob) ---
    image_paths = list(Path(OCR_TEST_PATH).glob("*_desk.png"))
    if not image_paths:
        print(f"⚠️ No images matching '*_desk.png' found in {OCR_TEST_PATH}")
        return

    print(f"📸 Found {len(image_paths)} images to process in {OCR_TEST_PATH}")

    results = []

    for img_path in SELECTED_IMAGES:
        img_path = Path(img_path)
        filename = img_path.stem.replace("_desk", "")
        print(f"\n=== Processing {img_path.name} ===")

        # Choose safe vs normal
        if use_safe:
            ocr_text, latency, tokens, mem = safe_ocr(
                model, processor, img_path, language="Italian", max_context=MAX_TOKENS
            )
        else:
            ocr_text, latency, tokens, mem = ocr_inference(
                model, processor, img_path, language="Italian", max_new_tokens=MAX_TOKENS
            )

        output_folder = img_path.parent / "results_safe_ocr"
        output_folder.mkdir(exist_ok=True, parents=True)

        output_path = output_folder / f"{filename}_fp16_resized.txt"
        save_text(output_path, ocr_text)

        print(f"✅ Finished {img_path.name}: {latency:.2f}s, tokens={tokens}")

        results.append({
            "image": img_path.name,
            "latency": latency,
            "tokens": tokens,
            "cpu_mem_before": mem[0],
            "cpu_mem_after": mem[1],
            "gpu_mem_before": mem[2],
            "gpu_mem_after": mem[3],
        })

    # --- Save performance CSV ---
    csv_path = Path(OCR_TEST_PATH) / "ocr_fp16_performance.csv"
    print(f"\n📊 Saving performance report to {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("✅ All done!")


if __name__ == "__main__":
    main(use_safe=False)  # set to False to run the original ocr_inference
