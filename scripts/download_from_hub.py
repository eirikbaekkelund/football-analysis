"""
Download model weights and annotation datasets from HuggingFace Hub.

Reads HF_TOKEN, HF_MODEL_REPO, and HF_DATASET_REPO from .env (optional —
public repos don't require a token).

Usage:
    python scripts/download_from_hub.py                     # weights + data
    python scripts/download_from_hub.py --models-only
    python scripts/download_from_hub.py --data-only
    python scripts/download_from_hub.py --data-only --data-name croatia_czechia_annotations
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_ENV = _ROOT / ".env"

if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "eirikbaekkelund/torchkick")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "eirikbaekkelund/torchkick-annotations")

MODEL_FILES = [
    ("players/yolo11l.pt", _ROOT / "models/players/yolo11l.pt"),
    ("keypoints/heatmap.pt", _ROOT / "models/keypoints/heatmap.pt"),
]


def download_models(api) -> None:
    print(f"\nDownloading models ← {HF_MODEL_REPO}")
    for repo_path, local_path in MODEL_FILES:
        if local_path.exists():
            print(f"  [skip] {local_path.relative_to(_ROOT)} already exists")
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {repo_path} → {local_path.relative_to(_ROOT)}")
        api.hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=repo_path,
            local_dir=str(_ROOT / "models"),
            repo_type="model",
        )
    print("  Models done.")


def download_data(api, data_name: str) -> None:
    dest = _ROOT / "data" / data_name
    if dest.exists():
        print(f"\n  [skip] {dest.relative_to(_ROOT)} already exists")
        return
    print(f"\nDownloading dataset ({data_name}) ← {HF_DATASET_REPO}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    api.snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=f"{data_name}/*",
        local_dir=str(_ROOT / "data"),
    )
    print("  Dataset done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download weights / data from HuggingFace Hub")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models-only", action="store_true")
    group.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--data-name",
        default="croatia_czechia_annotations",
        help="Folder name inside the dataset repo to download (default: croatia_czechia_annotations)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError:
        import sys

        print("[error] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    # Attach convenience methods so we can call them via api object
    api = HfApi(token=HF_TOKEN)
    api.hf_hub_download = lambda **kw: hf_hub_download(token=HF_TOKEN, **kw)
    api.snapshot_download = lambda **kw: snapshot_download(token=HF_TOKEN, **kw)

    if args.models_only:
        download_models(api)
    elif args.data_only:
        download_data(api, args.data_name)
    else:
        download_models(api)
        download_data(api, args.data_name)

    print("\nAll done.")


if __name__ == "__main__":
    main()
