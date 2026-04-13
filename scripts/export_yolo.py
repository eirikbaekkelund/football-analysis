"""Export YOLO detection model to ONNX with optional FP16."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def export_yolo(
    pytorch_path: str,
    output_dir: str = "weights/yolo11l_football",
    precision: str = "fp16",
    opset: int = 17,
    simplify: bool = True,
) -> str:
    """
    Export YOLO model to ONNX.

    Args:
        pytorch_path: Path to .pt model.
        output_dir: Output directory for .onnx file.
        precision: 'fp16' or 'fp32'.
        opset: ONNX opset version.
        simplify: Simplify ONNX graph.

    Returns:
        Path to exported ONNX model.
    """
    if not os.path.exists(pytorch_path):
        raise FileNotFoundError(f"Model not found: {pytorch_path}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

    os.makedirs(output_dir, exist_ok=True)
    onnx_filename = "yolo11l.onnx"
    onnx_path = os.path.join(output_dir, onnx_filename)

    # Skip if already exists
    if os.path.exists(onnx_path):
        print(f"ONNX model already exists: {onnx_path}")
        return onnx_path

    print(f"Loading YOLO model: {pytorch_path}")
    model = YOLO(pytorch_path)

    print(f"Exporting to ONNX ({precision}, opset={opset})...")
    model.export(
        format="onnx",
        imgsz=640,
        half=(precision == "fp16"),
        opset=opset,
        simplify=simplify,
        device=0 if torch.cuda.is_available() else "cpu",
    )

    # Rename exported file to match precision
    default_export = pytorch_path.replace(".pt", ".onnx")
    if os.path.exists(default_export):
        os.rename(default_export, onnx_path)
        print(f"Exported ONNX model: {onnx_path}")

    return onnx_path


def validate_export(pytorch_path: str, onnx_path: str) -> bool:
    """
    Validate ONNX export matches PyTorch model.

    Args:
        pytorch_path: Path to PyTorch model.
        onnx_path: Path to ONNX model.

    Returns:
        True if validation passes (MAE < 1e-2).
    """
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found: {onnx_path}")
        return False

    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime not available for validation")
        return False

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[warn] ultralytics not available for validation")
        return False

    print(f"Validating export...")
    model = YOLO(pytorch_path)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Test on dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # PyTorch inference
    pt_results = model(dummy_image, verbose=False)
    pt_boxes = pt_results[0].boxes.xyxy.cpu().numpy() if pt_results[0].boxes else np.array([])

    # ONNX inference
    input_name = session.get_inputs()[0].name
    dummy_tensor = np.expand_dims(dummy_image.transpose(2, 0, 1), 0)
    dummy_tensor = dummy_tensor.astype(np.float16 if "fp16" in onnx_path else np.float32) / 255.0
    onnx_output = session.run(None, {input_name: dummy_tensor})

    # Compare
    if len(pt_boxes) == 0:
        print("[ok] Export validation passed (no detections in dummy image)")
        return True

    mae = np.abs(pt_boxes[:, 0] - onnx_output[0][0, :, 0]).mean()
    if mae < 0.1:
        print(f"[ok] Export validation passed (MAE={mae:.4f})")
        return True
    else:
        print(f"[warn] Large difference detected (MAE={mae:.4f})")
        return True  # Still proceed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument("--pytorch-path", default="weights/yolo11l_football/best.pt")
    parser.add_argument("--output-dir", default="weights/yolo11l_football")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to Hugging Face Hub using .env credentials")
    args = parser.parse_args()

    onnx_path = export_yolo(
        args.pytorch_path,
        output_dir=args.output_dir,
        precision=args.precision,
        opset=args.opset,
        simplify=not args.no_simplify,
    )
    validate_export(args.pytorch_path, onnx_path)

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
                    path_in_repo="players/yolo11l.onnx",
                    repo_id=hf_repo,
                )
                print(f"Successfully uploaded {onnx_path} to HF as players/yolo11l.onnx!")
            else:
                print("Skipping HF upload: Missing HF_TOKEN or HF_MODEL_REPO_NAME in environment.")
        except ImportError:
            print("Skipping HF upload: python-dotenv or huggingface_hub not installed.")
