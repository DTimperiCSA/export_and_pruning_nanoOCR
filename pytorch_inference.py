import time, os, psutil
from pathlib import Path
from PIL import Image
import torch
from torch.cuda.amp import autocast
from transformers import AutoProcessor, AutoModelForImageTextToText
import difflib
import editdistance

from paths import *  # MODEL_ID, ASSETS_DIR, HUGGING_FACE_RES, PYTORCH_INFERENCE_DIR must be defined

# ---------------- DEBUG ENVIRONMENT SETUP ----------------
print("=== ENVIRONMENT INFO ===")
print(f"PyTorch version: {torch.__version__}")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("Transformers not found!")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("⚠️ CUDA not available, using CPU.")

print(f"Model ID: {MODEL_ID}")
print(f"Assets directory: {ASSETS_DIR}")
print(f"Model path: {Path('exported_models/nanoocr_model.pth')}")
print("=========================\n")

# ---------------- CONFIG ----------------
PTH_PATH = Path("exported_models/nanoocr_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = ASSETS_DIR

# ---------------- HELPERS ----------------
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

def get_memory_usage():
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    gpu_mem = torch.cuda.memory_allocated(DEVICE) / (1024**2) if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem

def text_similarity(text1, text2):
    return difflib.SequenceMatcher(None, text1, text2).ratio() * 100

def save_ocr_outputs(output_dir_base, image_dir, outputs):
    save_dir = output_dir_base / image_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

    for img_path, text in zip(image_files, outputs):
        out_path = save_dir / (img_path.stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Saved OCR output: {out_path}")

# ---------------- OCR FUNCTION ----------------
def ocr_inference(model, image_path, processor, max_new_tokens=4096):
    print(f"\n🔍 Processing image: {image_path.name}")
    
    image = Image.open(image_path).convert("RGB")
    print(f"Original image size: {image.size}")
    image = image.resize((1024, 1024))
    print(f"Resized image to: {image.size}")

    prompt = "Extract text from document naturally, return tables in HTML, equations in LaTeX."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]}
    ]
    
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) \
        if hasattr(processor, "apply_chat_template") else prompt
    
    inputs = processor(text=[text_input], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    cpu_before, gpu_before = get_memory_usage()

    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE):
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Inputs device: {[v.device for v in inputs.values()]}")
            print(f"max_new_tokens: {max_new_tokens}")

            start = time.time()
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            latency = time.time() - start
            print(f"Generation time: {latency:.2f}s")
            tokens_generated = output_ids.shape[-1]
            output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    cpu_after, gpu_after = get_memory_usage()

    print(f"🕒 Latency: {latency:.3f}s | Tokens generated: {tokens_generated}")
    print(f"🧠 CPU Mem Before/After: {cpu_before:.1f} / {cpu_after:.1f} MB")
    print(f"🧠 GPU Mem Before/After: {gpu_before:.1f} / {gpu_after:.1f} MB")

    return output_text, latency, tokens_generated

# ---------------- LOAD PROCESSOR ----------------
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# ---------------- RUN BENCHMARK FUNCTION ----------------
def benchmark_model(model_name, load_local=False):
    torch.cuda.empty_cache()
    
    if load_local:
        print(f"\n🚀 Loading Local PyTorch model: {model_name}...")
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
        print("Loading local state dict...")
        state_dict = torch.load(PTH_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(f"\n🌐 Loading HF model: {model_name}...")
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
    
    model.to(DEVICE)
    model.eval()

    images = list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg"))
    print(f"🖼️ Found {len(images)} images for benchmarking.\n")

    if not images:
        raise ValueError("No images found in the assets directory.")

    # Warm-up
    print("🔥 Running warm-up...")
    _text, warm_latency, _ = ocr_inference(model, images[0], processor)
    
    total_time, total_tokens = 0, 0
    outputs = []
    
    for img_path in images:
        text, latency, tokens = ocr_inference(model, img_path, processor)
        outputs.append(text)
        total_time += latency
        total_tokens += tokens
    
    avg_time = total_time / len(images)
    throughput = len(images) / total_time
    tokens_per_sec = total_tokens / total_time
    cpu_mem, gpu_mem = get_memory_usage()
    model_size = os.path.getsize(PTH_PATH) / (1024**2)
    
    # Free GPU
    del model
    torch.cuda.empty_cache()
    
    return {
        "name": model_name,
        "warmup": warm_latency,
        "avg_time": avg_time,
        "throughput": throughput,
        "tokens_per_sec": tokens_per_sec,
        "cpu_mem": cpu_mem,
        "gpu_mem": gpu_mem,
        "outputs": outputs,
        "model_size_MB": model_size,
    }

# ---------------- RUN BENCHMARK AND SAVE RESULTS ----------------
results = []

hf_result = benchmark_model("HuggingFace", load_local=False)
results.append(hf_result)
save_ocr_outputs(HUGGING_FACE_RES, IMAGE_DIR, hf_result["outputs"])

local_result = benchmark_model("LocalPyTorch", load_local=True)
results.append(local_result)
save_ocr_outputs(PYTORCH_INFERENCE_DIR, IMAGE_DIR, local_result["outputs"])

# ---------------- REPORT ----------------
print("\n===== 📊 PERFORMANCE COMPARISON =====")
for r in results:
    print(f"\n📌 {r['name']}:")
    print(f"  Warm-up: {r['warmup']:.3f} s")
    print(f"  Avg Latency: {r['avg_time']:.3f} s/img")
    print(f"  Throughput: {r['throughput']:.2f} img/s")
    print(f"  Tokens/sec: {r['tokens_per_sec']:.1f}")
    print(f"  CPU Mem: {r['cpu_mem']:.1f} MB | GPU Mem: {r['gpu_mem']:.1f} MB")
    print(f"  Model Size: {r['model_size_MB']:.1f} MB")

# ---------------- OUTPUT SIMILARITY ----------------
print("\n===== 🤖 OUTPUT SIMILARITY (HF vs Local) =====")
for i, img_path in enumerate(list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg"))):
    ref = results[0]["outputs"][i]
    hyp = results[1]["outputs"][i]
    
    sim = text_similarity(ref, hyp)
    lev = levenshtein_distance(ref, hyp)
    lev_norm = normalized_levenshtein(ref, hyp)
    cer_val = cer(hyp, ref)
    wer_val = wer(hyp, ref)
    
    print(f"\n{img_path.name}:")
    print(f"  Similarity: {sim:.2f}%")
    print(f"  Levenshtein distance: {lev}")
    print(f"  Normalized Levenshtein: {lev_norm:.3f}")
    print(f"  CER: {cer_val:.3f}")
    print(f"  WER: {wer_val:.3f}")
