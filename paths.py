from pathlib import Path

DESKTOP_DIR = Path.home() / "Desktop"
REPO_DIR = DESKTOP_DIR / "export_and_pruning_nanoOCR"
HUGGING_FACE_RES = REPO_DIR /"huggingface_inference"
PYTORCH_INFERENCE_DIR = REPO_DIR / "pytorch_inference"

OCR_TEST_PATH = Path(r"C:\Demo\fascicoli_del_personale\csaFPApp\test\temp\002346_GIANNINO ANTONIETTA")

ASSETS_DIR = REPO_DIR / "assets"

IMAGE_PATH = ASSETS_DIR / "benchmark.png"  
MODEL_ID = "nanonets/Nanonets-OCR-s"
SAVE_DIR = Path("exported_models")
FP16_MODEL_PATH = Path("exported_models/nanoocr_fp16.pth")

BELGRADO_OCR = Path(r"C:\Belgrado\Fascicoli")