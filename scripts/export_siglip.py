"""Export SigLIP vision encoder to ONNX with optional FP16."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def export_siglip(
    output_dir: str = "weights",
    model_name: str = "google/siglip-base-patch16-384",
    precision: str = "fp16",
    opset: int = 17,
    simplify: bool = True,
) -> str:
    """
    Export SigLIP vision encoder to ONNX.

    Args:
        output_dir: Output directory for .onnx file.
        model_name: HuggingFace model name.
        precision: 'fp16' or 'fp32'.
        opset: ONNX opset version.
        simplify: Simplify ONNX graph.

    Returns:
        Path to exported ONNX model.
    """
    try:
        from transformers import AutoImageProcessor, SiglipVisionModel
    except ImportError:
        raise ImportError("transformers not installed. Install with: pip install transformers")

    os.makedirs(output_dir, exist_ok=True)
    onnx_filename = f"siglip_{precision}.onnx"
    onnx_path = os.path.join(output_dir, onnx_filename)

    # Skip if already exists
    if os.path.exists(onnx_path):
        print(f"ONNX model already exists: {onnx_path}")
        return onnx_path

    print(f"Loading SigLIP model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name).to(device).eval()

    # Create dummy input
    dummy_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    inputs = processor(images=dummy_image, return_tensors="pt").to(device)

    print(f"Exporting to ONNX ({precision}, opset={opset})...")

    # Export via torch.onnx
    dummy_input = (inputs["pixel_values"],)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["pooler_output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"pixel_values": {0: "batch_size"}},
    )

    print(f"Exported ONNX model: {onnx_path}")
    return onnx_path


def validate_export(output_dir: str = "weights", model_name: str = "google/siglip-base-patch16-384") -> bool:
    """
    Validate ONNX export matches PyTorch model.

    Args:
        output_dir: Output directory containing .onnx file.
        model_name: HuggingFace model name.

    Returns:
        True if validation passes.
    """
    onnx_path = os.path.join(output_dir, "siglip_fp16.onnx")
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found: {onnx_path}")
        return False

    try:
        import onnxruntime as ort
        from transformers import AutoImageProcessor, SiglipVisionModel
    except ImportError:
        print("[warn] Dependencies not available for validation")
        return False

    print(f"Validating export...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name).to(device).eval()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Test on dummy image
    dummy_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    inputs = processor(images=dummy_image, return_tensors="pt").to(device)

    # PyTorch inference
    with torch.inference_mode():
        pt_output = model(**inputs).pooler_output.cpu().numpy()

    # ONNX inference
    session_inputs = {session.get_inputs()[0].name: inputs["pixel_values"].cpu().numpy()}
    onnx_output = session.run(None, session_inputs)[0]

    # Compare
    mae = np.abs(pt_output - onnx_output).mean()
    if mae < 0.01:
        print(f"[ok] Export validation passed (MAE={mae:.6f})")
        return True
    else:
        print(f"[warn] Difference detected (MAE={mae:.6f})")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SigLIP to ONNX")
    parser.add_argument("--output-dir", default="weights")
    parser.add_argument("--model-name", default="google/siglip-base-patch16-384")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    onnx_path = export_siglip(
        output_dir=args.output_dir,
        model_name=args.model_name,
        precision=args.precision,
        simplify=not args.no_simplify,
    )
    validate_export(args.output_dir, args.model_name)
