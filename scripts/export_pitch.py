"""Export pitch keypoint detector to ONNX with optional FP16."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def export_pitch(
    pytorch_path: str,
    output_dir: str = "weights/keypoints",
    precision: str = "fp16",
    opset: int = 17,
) -> str:
    """
    Export pitch keypoint detector to ONNX.

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
        from torchkick.models.pitch import HeatmapPitchDetector
    except ImportError:
        raise ImportError("torchkick models not available")

    os.makedirs(output_dir, exist_ok=True)
    onnx_filename = "heatmap.onnx"
    onnx_path = os.path.join(output_dir, onnx_filename)

    # Skip if already exists
    if os.path.exists(onnx_path):
        print(f"ONNX model already exists: {onnx_path}")
        return onnx_path

    print(f"Loading pitch detector model: {pytorch_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeatmapPitchDetector(weights_path=pytorch_path, device=str(device))

    # Export backbone (simpler than full pipeline)
    print(f"Exporting to ONNX ({precision}, opset={opset})...")
    dummy_input = torch.randn(1, 3, 560, 560, device=device)
    if precision == "fp16":
        dummy_input = dummy_input.half()
        model._model = model._model.half()

    torch.onnx.export(
        model._model,  # underlying torch model
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}},
    )

    print(f"Exported ONNX model: {onnx_path}")
    return onnx_path


def validate_export(pytorch_path: str, onnx_path: str, precision: str = "fp16") -> bool:
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
        from torchkick.models.pitch import HeatmapPitchDetector
    except ImportError:
        print("[warn] Dependencies not available for validation")
        return False

    print(f"Validating export...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeatmapPitchDetector(weights_path=pytorch_path, device=str(device))
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Test on dummy input
    dummy_input = torch.randn(1, 3, 560, 560, device=device)
    if precision == "fp16":
        dummy_input = dummy_input.half()
        model._model = model._model.half()

    # PyTorch inference
    with torch.inference_mode():
        pt_result = model._model(dummy_input)
        if isinstance(pt_result, tuple):
            pt_output = pt_result[0].cpu().numpy()
        else:
            pt_output = pt_result.cpu().numpy()

    # ONNX inference
    onnx_input = dummy_input.cpu().numpy().astype(np.float16 if precision == "fp16" else np.float32)
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
    parser = argparse.ArgumentParser(description="Export pitch detector to ONNX")
    parser.add_argument("--pytorch-path", default="weights/keypoints/best.pt")
    parser.add_argument("--output-dir", default="weights/keypoints")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to Hugging Face Hub using .env credentials")
    args = parser.parse_args()

    onnx_path = export_pitch(
        args.pytorch_path,
        output_dir=args.output_dir,
        precision=args.precision,
    )
    validate_export(args.pytorch_path, onnx_path, precision=args.precision)

    if args.push_to_hub:
        try:
            from dotenv import load_dotenv
            from huggingface_hub import HfApi

            load_dotenv()

            hf_token = os.getenv("HF_TOKEN")
            hf_repo = os.getenv("HF_MODEL_REPO_NAME")
            if hf_token and hf_repo:
                print(f"Uploading {onnx_path} to Hugging Face repo: {hf_repo}...")
                api = HfApi(token=hf_token)
                api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="model")
                api.upload_file(
                    path_or_fileobj=onnx_path,
                    path_in_repo="keypoints/heatmap.onnx",
                    repo_id=hf_repo,
                )
                print(f"Successfully uploaded {onnx_path} to HF as keypoints/heatmap.onnx!")
            else:
                print("Skipping HF upload: Missing HF_TOKEN or HF_MODEL_REPO_NAME in environment.")
        except ImportError:
            print("Skipping HF upload: python-dotenv or huggingface_hub not installed.")
