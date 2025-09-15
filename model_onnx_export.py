import os
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

# ---------------- CONFIG ----------------
MODEL_ID = "nanonets/Nanonets-OCR-s"
ONNX_DIR = Path("onnx_export")
ONNX_PATH = ONNX_DIR / "nanoocr_forward.onnx"
DEVICE = "cpu"   # change to "cuda" if you want GPU export
OPSET = 17
# -----------------------------------------


class NanoOCRWrapper(nn.Module):
    """
    A wrapper around AutoModelForImageTextToText that ensures decoder_input_ids
    are always valid when exporting.
    """

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, pixel_values, decoder_input_ids=None):
        if decoder_input_ids is None:
            bos_token_id = (
                self.tokenizer.bos_token_id
                or self.tokenizer.cls_token_id
                or 0
            )
            batch_size = pixel_values.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=pixel_values.device,
            )

        outputs = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,  # must disable cache for ONNX
        )
        return outputs.logits


def export_to_onnx(model, processor, tokenizer, save_path, opset=17, device="cpu"):
    model.to(device)
    model.eval()

    wrapper = NanoOCRWrapper(model, tokenizer).to(device)

    # Create dummy inputs
    dummy_pixel_values = torch.randn(1, 3, 384, 384, device=device)
    dummy_decoder_input_ids = torch.tensor(
        [[tokenizer.bos_token_id or tokenizer.cls_token_id or 0]],
        device=device,
    )

    os.makedirs(save_path.parent, exist_ok=True)

    print(f"Exporting to ONNX: {save_path}")
    torch.onnx.export(
        wrapper,
        (dummy_pixel_values, dummy_decoder_input_ids),
        f=save_path.as_posix(),
        input_names=["pixel_values", "decoder_input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "decoder_input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print("✅ Export finished:", save_path)
    return save_path


def main():
    print("=== START ===")
    print("Loading model and processor...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

    onnx_path = export_to_onnx(model, processor, tokenizer, ONNX_PATH, opset=OPSET, device=DEVICE)

    print("=== DONE ===")
    print("ONNX model saved at:", onnx_path)


if __name__ == "__main__":
    main()
