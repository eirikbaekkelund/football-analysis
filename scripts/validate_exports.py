"""Validate all model exports for correctness."""

from __future__ import annotations

import json
import os
from datetime import datetime


def validate_all_exports() -> dict:
    """
    Validate all exported models.

    Returns:
        Dict with validation results.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    # YOLO validation
    print("\n" + "=" * 60)
    print("Validating YOLO export...")
    print("=" * 60)
    try:
        from scripts.export_yolo import validate_export as validate_yolo

        success = validate_yolo(
            "weights/yolo11l_football/best.pt",
            "weights/yolo11l_football/best_fp16.onnx",
        )
        results["models"]["yolo_detection"] = {"status": "pass" if success else "fail"}
    except Exception as e:
        print(f"[error] YOLO validation failed: {e}")
        results["models"]["yolo_detection"] = {"status": "error", "error": str(e)}

    # SigLIP validation
    print("\n" + "=" * 60)
    print("Validating SigLIP export...")
    print("=" * 60)
    try:
        from scripts.export_siglip import validate_export as validate_siglip

        success = validate_siglip("weights", "google/siglip-base-patch16-384")
        results["models"]["siglip_embedder"] = {"status": "pass" if success else "fail"}
    except Exception as e:
        print(f"[error] SigLIP validation failed: {e}")
        results["models"]["siglip_embedder"] = {"status": "error", "error": str(e)}

    # Pitch validation
    print("\n" + "=" * 60)
    print("Validating Pitch detector export...")
    print("=" * 60)
    try:
        from scripts.export_pitch import validate_export as validate_pitch

        if os.path.exists("weights/keypoints/best.pt"):
            success = validate_pitch(
                "weights/keypoints/best.pt",
                "weights/keypoints/best_fp16.onnx",
            )
            results["models"]["pitch_keypoints"] = {"status": "pass" if success else "fail"}
        else:
            results["models"]["pitch_keypoints"] = {"status": "skipped", "reason": "weights not found"}
    except Exception as e:
        print(f"[error] Pitch validation failed: {e}")
        results["models"]["pitch_keypoints"] = {"status": "error", "error": str(e)}

    # ReID validation
    print("\n" + "=" * 60)
    print("Validating ReID export...")
    print("=" * 60)
    try:
        from scripts.export_reid import validate_export as validate_reid

        if os.path.exists("weights/reid/best.pt"):
            success = validate_reid(
                "weights/reid/best.pt",
                "weights/reid/best_fp16.onnx",
            )
            results["models"]["reid_embedder"] = {"status": "pass" if success else "fail"}
        else:
            results["models"]["reid_embedder"] = {"status": "skipped", "reason": "weights not found"}
    except Exception as e:
        print(f"[error] ReID validation failed: {e}")
        results["models"]["reid_embedder"] = {"status": "error", "error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    for model_name, result in results["models"].items():
        status = result.get("status", "unknown")
        print(f"  {model_name}: {status}")

    # Save report
    report_path = "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved: {report_path}")

    return results


if __name__ == "__main__":
    validate_all_exports()
