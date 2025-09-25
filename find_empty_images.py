import os
from pathlib import Path
from PIL import Image
import numpy as np
import csv

# Folder with images to check
IMAGE_ROOT = r"C:\Belgrado\Fascicoli\ACCORINTI_PABLO_JAVIER_2011"  # adjust if needed

# Where to save results
RESULTS_CSV = "white_images.csv"

# Supported formats
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Tolerance: how "white" an image must be (0 = pure white only, >0 allows tiny noise)
WHITE_THRESHOLD = 250   # pixel values must be >= this (0–255)
FRACTION_REQUIRED = 0.999  # at least 99.9% of pixels must be white-ish


def collect_images(root_path):
    return [p for p in Path(root_path).rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]


def is_white_image(image_path):
    """Return True if the image is completely (or almost) white."""
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)

        # Normalize pixel values to range [0,255]
        mask = np.all(arr >= WHITE_THRESHOLD, axis=-1)  # True if pixel is white-enough
        fraction_white = mask.sum() / mask.size

        return fraction_white >= FRACTION_REQUIRED, fraction_white
    except Exception as e:
        print(f"⚠️ Could not process {image_path}: {e}")
        return False, 0.0


def main():
    image_paths = collect_images(IMAGE_ROOT)
    if not image_paths:
        print(f"⚠️ No images found in {IMAGE_ROOT}")
        return

    print(f"📸 Found {len(image_paths)} images, checking for white-only ones...")

    results = []
    for img_path in image_paths:
        is_white, frac = is_white_image(img_path)
        print(f"{img_path} -> {'WHITE' if is_white else 'not white'} ({frac:.4f})")

        results.append({
            "image": str(img_path.relative_to(IMAGE_ROOT)),
            "white_fraction": frac,
            "is_white": is_white
        })


if __name__ == "__main__":
    main()
