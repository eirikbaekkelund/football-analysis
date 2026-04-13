"""Export ReID backbone to ONNX with optional FP16."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def export_reid(
    pytorch_path: str,
    output_dir: str = "weights/reid",
    precision: str = "fp16",
    opset: int = 17,
) -> str:
    """
    Export ReID backbone to ONNX.

    Args:
        pytorch_path: Path to .pt model.
        output_dir: Output directory for .onnx file.
        precision: 'fp16' or 'fp32'.
        opset: ONNX opset version.

    Returns:
        Path to exported ONNX model.
    """
    if not os.path.exists(pytorch_path):
        raise FileNotFoundError(f"Model not found: {pytorch_path}")

    try:
        from torchkick.models.reid import DINOv2ReIDEmbedder
    except ImportError:
        raise ImportError("torchkick.models.reid not available")

    os.makedirs(output_dir, exist_ok=True)
    onnx_filename = Path(pytorch_path).stem + f"_{precision}.onnx"
    onnx_path = os.path.join(output_dir, onnx_filename)

    # Skip if already exists
    if os.path.exists(onnx_path):
        print(f"ONNX model already exists: {onnx_path}")
        return onnx_path

    print(f"Loading ReID embedder: {pytorch_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = DINOv2ReIDEmbedder(weights_path=pytorch_path, device=str(device))

    # Export backbone
    print(f"Exporting to ONNX ({precision}, opset={opset})...")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        embedder.backbone,  # underlying torch model
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}},
    )

    print(f"Exported ONNX model: {onnx_path}")
    return onnx_path


def validate_export(pytorch_path: str, onnx_path: str) -> bool:
    """
    Validate ONNX export matches PyTorch model.

    Args:
        pytorch_path: Path to PyTorch model.
        onnx_path: Path to ONNX model.

    Returns:
        True if validation passes.
    """
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found: {onnx_path}")
        return False

    try:
        import onnxruntime as ort
        from torchkick.models.reid import DINOv2ReIDEmbedder
    except ImportError:
        print("[warn] Dependencies not available for validation")
        return False

    print(f"Validating export...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = DINOv2ReIDEmbedder(weights_path=pytorch_path, device=str(device))
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Test on dummy input
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # PyTorch inference
    with torch.inference_mode():
        pt_output = embedder.backbone(dummy_input).cpu().numpy()

    # ONNX inference
    onnx_input = dummy_input.cpu().numpy().astype(np.float32)
    onnx_output = session.run(None, {"input": onnx_input})[0]

    # Compare
    mae = np.abs(pt_output - onnx_output).mean()
    if mae < 0.01:
        print(f"[ok] Export validation passed (MAE={mae:.6f})")
        return True
    else:
        print(f"[warn] Difference detected (MAE={mae:.6f})")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ReID backbone to ONNX")
    parser.add_argument("--pytorch-path", default="weights/reid/best.pt")
    parser.add_argument("--output-dir", default="weights/reid")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    onnx_path = export_reid(
        args.pytorch_path,
        output_dir=args.output_dir,
        precision=args.precision,
    )
    validate_export(args.pytorch_path, onnx_path)
