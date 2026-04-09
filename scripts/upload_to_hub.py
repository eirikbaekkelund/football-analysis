"""
Upload model weights and annotation data to HuggingFace Hub.

Reads HF_TOKEN, HF_MODEL_REPO, and HF_DATASET_REPO from .env.
Add to .env:
    HF_TOKEN=hf_...
    HF_MODEL_REPO=your-username/torchkick
    HF_DATASET_REPO=your-username/torchkick-annotations

Usage:
    python scripts/upload_to_hub.py                     # upload everything
    python scripts/upload_to_hub.py --models-only
    python scripts/upload_to_hub.py --data-only
    python scripts/upload_to_hub.py --data-dir croatia_czechia_annotations/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Load .env from repo root
_ROOT = Path(__file__).parent.parent
_ENV = _ROOT / ".env"

if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "your-username/torchkick")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "your-username/torchkick-annotations")

# Model weight paths relative to repo root
MODEL_FILES = [
    ("models/players/yolo11l.pt", "players/yolo11l.pt"),
    ("models/keypoints/heatmap.pt", "keypoints/heatmap.pt"),
]


def upload_models(api) -> None:
    print(f"\nUploading models → {HF_MODEL_REPO}")
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
    for local_rel, repo_path in MODEL_FILES:
        local = _ROOT / local_rel
        if not local.exists():
            print(f"  [skip] {local_rel} not found")
            continue
        print(f"  {local_rel} → {repo_path}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=repo_path,
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )
    print("  Models done.")


def upload_data(api, data_dir: str) -> None:
    folder = Path(data_dir)
    if not folder.exists():
        print(f"[error] data dir not found: {folder}", file=sys.stderr)
        sys.exit(1)
    print(f"\nUploading dataset ({folder.name}) → {HF_DATASET_REPO}")
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo=folder.name,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        ignore_patterns=["*.DS_Store"],
    )
    print("  Dataset done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload weights / data to HuggingFace Hub")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models-only", action="store_true", help="Upload model weights only")
    group.add_argument("--data-only", action="store_true", help="Upload annotation data only")
    parser.add_argument(
        "--data-dir",
        default=str(_ROOT / "croatia_czechia_annotations"),
        help="Path to annotation folder to upload (default: croatia_czechia_annotations/)",
    )
    args = parser.parse_args()

    if not HF_TOKEN:
        print("[error] HF_TOKEN not set. Add it to .env:\n  HF_TOKEN=hf_...", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[error] huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    if args.models_only:
        upload_models(api)
    elif args.data_only:
        upload_data(api, args.data_dir)
    else:
        upload_models(api)
        upload_data(api, args.data_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
