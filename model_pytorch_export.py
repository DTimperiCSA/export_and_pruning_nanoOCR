from pathlib import Path
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from paths import *

# ---------------- CONFIG ----------------
SAVE_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL (FP32) ----------------
model_fp32 = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    device_map=device,
)
model_fp32.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# ---------------- OCR FUNCTION ----------------
def ocr_page(image_path, model, processor, max_new_tokens=4096):
    prompt = """Extract the text from the above document as if you were reading it naturally.
Return the tables in html format. Return equations in LaTeX. Wrap watermarks in <watermark></watermark>.
Wrap page numbers in <page_number></page_number>. Describe images in <img></img> if no caption."""

    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

# ---------------- RUN OCR ----------------
result = ocr_page(IMAGE_PATH, model_fp32, processor, max_new_tokens=15000)
print("OCR Result (FP32):\n", result)

# ---------------- SAVE FP32 MODEL ----------------
fp32_path = SAVE_DIR / "nanoocr_fp32.pth"
model_fp32_cpu = model_fp32.to("cpu")
torch.save(model_fp32_cpu.state_dict(), fp32_path)
print(f"✅ Saved FP32 model to: {fp32_path}")

# ---------------- CONVERT TO FP16 ----------------
model_fp16 = model_fp32.half()
model_fp16.eval()
model_fp16_cpu = model_fp16.to("cpu")
fp16_path = SAVE_DIR / "nanoocr_fp16.pth"
torch.save(model_fp16.state_dict(), fp16_path)
print(f"✅ Saved FP16 model to: {fp16_path}")
