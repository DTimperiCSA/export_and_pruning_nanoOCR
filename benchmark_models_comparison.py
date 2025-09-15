import time, os, psutil
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import difflib
import editdistance
import csv
from paths import *

# ---------------- CONFIG ----------------
FP32_PATH = Path("exported_models/nanoocr_fp32.pth")
FP16_PATH = Path("exported_models/nanoocr_fp16.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = ASSETS_DIR

# ---------------- METRIC FUNCTIONS ----------------
def save_outputs_to_dir(output_dir, image_paths, outputs):
    output_dir.mkdir(parents=True, exist_ok=True)
    for img_path, text in zip(image_paths, outputs):
        out_file = output_dir / (img_path.stem + ".txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"💾 Saved OCR result to: {out_file}")

def levenshtein_distance(s1, s2):
    return editdistance.eval(s1, s2)

def normalized_levenshtein(s1, s2):
    dist = levenshtein_distance(s1, s2)
    return dist / max(len(s1), len(s2)) if max(len(s1), len(s2)) > 0 else 0

def cer(s1, s2):
    return levenshtein_distance(s1, s2) / len(s2) if len(s2) > 0 else 0

def wer(s1, s2):
    ref_words = s2.split()
    hyp_words = s1.split()
    return editdistance.eval(hyp_words, ref_words) / len(ref_words) if len(ref_words) > 0 else 0

def text_similarity(text1, text2):
    return difflib.SequenceMatcher(None, text1, text2).ratio() * 100

# ---------------- OCR ----------------
def get_memory_usage():
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    gpu_mem = torch.cuda.memory_allocated(DEVICE) / (1024**2) if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem

def ocr_inference(model, image_path, processor, max_new_tokens=4096):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image = image.resize((width // 2, height // 2))  # Resize to half original

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

    return output_text, latency, output_ids.shape[-1], (cpu_before, cpu_after, gpu_before, gpu_after)

def ocr_inference_no_resize(model, image_path, processor, max_new_tokens=4096):
    image = Image.open(image_path).convert("RGB")  # NO resize here

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

    return output_text, latency, output_ids.shape[-1], (cpu_before, cpu_after, gpu_before, gpu_after)

# ---------------- BENCHMARK ----------------
def benchmark(name, model_path=None, is_fp16=False, from_hf=False, no_resize=False):
    print(f"\n🚀 Benchmarking: {name}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if from_hf:
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
    else:
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    if is_fp16:
        model = model.half()

    model.to(DEVICE)
    model.eval()

    image_paths = list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError("No images in ASSETS_DIR.")

    total_time = 0
    total_tokens = 0
    outputs = []

    print("🔥 Warm-up pass...")
    if no_resize:
        ocr_inference_no_resize(model, image_paths[0], processor)
    else:
        ocr_inference(model, image_paths[0], processor)

    for img in image_paths:
        if no_resize:
            out, latency, tokens, _ = ocr_inference_no_resize(model, img, processor)
        else:
            out, latency, tokens, _ = ocr_inference(model, img, processor)
        print(f"🕒 {img.name}: {latency:.2f}s, Tokens: {tokens}")
        outputs.append(out)
        total_time += latency
        total_tokens += tokens

    avg_time = total_time / len(image_paths)
    throughput = len(image_paths) / total_time
    tokens_per_sec = total_tokens / total_time
    cpu_mem, gpu_mem = get_memory_usage()

    return {
        "name": name,
        "avg_latency": avg_time,
        "throughput": throughput,
        "tokens_per_sec": tokens_per_sec,
        "cpu_mem": cpu_mem,
        "gpu_mem": gpu_mem,
        "outputs": outputs,
        "image_paths": image_paths,
    }

# ---------------- RUN BENCHMARKS ----------------
results = [
    benchmark("HuggingFace", from_hf=True),
    benchmark("Local FP32", model_path=FP32_PATH),
    benchmark("Local FP16", model_path=FP16_PATH, is_fp16=True),
    benchmark("No Resize", model_path=FP32_PATH, no_resize=True),
    benchmark("No Resize", model_path=FP16_PATH, no_resize=True),
]

# ---------------- SAVE OUTPUTS ----------------
save_outputs_to_dir(HUGGING_FACE_RES, results[0]["image_paths"], results[0]["outputs"])
save_outputs_to_dir(PYTORCH_INFERENCE_DIR / "fp32", results[1]["image_paths"], results[1]["outputs"])
save_outputs_to_dir(PYTORCH_INFERENCE_DIR / "fp16", results[2]["image_paths"], results[2]["outputs"])
save_outputs_to_dir(PYTORCH_INFERENCE_DIR / "fp32_no_resize", results[3]["image_paths"], results[3]["outputs"])
save_outputs_to_dir(PYTORCH_INFERENCE_DIR / "fp16_no_resize", results[4]["image_paths"], results[4]["outputs"])

# ---------------- PRINT PERFORMANCE REPORT ----------------
print("\n📊 === PERFORMANCE COMPARISON ===")
for r in results:
    print(f"\n🧠 {r['name']}:")
    print(f"  Avg Latency:     {r['avg_latency']:.2f} s/image")
    print(f"  Throughput:      {r['throughput']:.2f} image/s")
    print(f"  Tokens/sec:      {r['tokens_per_sec']:.1f}")
    print(f"  CPU Mem Usage:   {r['cpu_mem']:.1f} MB")
    print(f"  GPU Mem Usage:   {r['gpu_mem']:.1f} MB")

# ---------------- COMPARE OUTPUTS ----------------
print("\n🤖 === OUTPUT SIMILARITY COMPARISON ===")
baseline = results[0]["outputs"]  # HuggingFace baseline
image_paths = results[0]["image_paths"]

similarity_csv = PYTORCH_INFERENCE_DIR / "similarity_report.csv"
with open(similarity_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Image", "Compared Model", "Similarity (%)", "Levenshtein",
        "Normalized Levenshtein", "CER", "WER"
    ])

    for model_result in results[1:]:  # Skip HuggingFace baseline
        print(f"\n🔍 Comparing: {model_result['name']} vs HuggingFace")
        for i, img_path in enumerate(image_paths):
            ref = baseline[i]
            hyp = model_result["outputs"][i]

            sim = text_similarity(ref, hyp)
            lev = levenshtein_distance(ref, hyp)
            lev_norm = normalized_levenshtein(ref, hyp)
            cer_val = cer(hyp, ref)
            wer_val = wer(hyp, ref)

            print(f"\n{img_path.name}:")
            print(f"  Similarity:             {sim:.2f}%")
            print(f"  Levenshtein distance:   {lev}")
            print(f"  Normalized Levenshtein: {lev_norm:.3f}")
            print(f"  CER:                    {cer_val:.3f}")
            print(f"  WER:                    {wer_val:.3f}")

            writer.writerow([
                img_path.name,
                model_result["name"],
                f"{sim:.2f}",
                lev,
                f"{lev_norm:.3f}",
                f"{cer_val:.3f}",
                f"{wer_val:.3f}"
            ])

print(f"\n📄 Similarity results saved to: {similarity_csv}")
