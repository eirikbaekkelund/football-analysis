"""
Fine-tune player detector and pitch keypoint detector on annotation tool exports.

Converts the annotation export layout:
    data_dir/
      images/      *.jpg
      players/     *.txt   (YOLO detection)
      keypoints/   *.txt   (YOLO-pose 32-kp)

Into the format expected by each training pipeline, then runs fine-tuning
from the existing model checkpoints.

Usage:
    # Fine-tune both models
    python scripts/finetune.py --data-dir data/croatia_czechia_annotations/

    # Detection only
    python scripts/finetune.py --data-dir data/croatia_czechia_annotations/ --detection-only

    # Keypoints only
    python scripts/finetune.py --data-dir data/croatia_czechia_annotations/ --keypoints-only

    # Custom checkpoints / epochs
    python scripts/finetune.py \\
        --data-dir data/croatia_czechia_annotations/ \\
        --player-weights models/players/yolo11l.pt \\
        --pitch-weights  models/keypoints/heatmap.pt \\
        --epochs 20
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent


def _prepare_dataset(data_dir: Path, out_dir: Path, val_frac: float = 0.15) -> tuple[Path, Path]:
    """
    Convert annotation export → two dataset roots:
        out_dir/detection/   for YOLO player detection
        out_dir/keypoints/   for DINOv2 heatmap pitch keypoints

    Returns (detection_dir, keypoints_dir).
    """
    subdirs = []
    if (data_dir / "images").exists() and (data_dir / "players").exists() and (data_dir / "keypoints").exists():
        subdirs.append(data_dir)
    else:
        for d in data_dir.iterdir():
            if d.is_dir() and (d / "images").exists() and (d / "players").exists() and (d / "keypoints").exists():
                subdirs.append(d)

    if not subdirs:
        raise RuntimeError(f"No valid annotation folders found in {data_dir}")

    images = []
    player_lbls = {}
    kp_lbls = {}

    for d in subdirs:
        images.extend(d.glob("images/*.jpg"))
        for p in d.glob("players/*.txt"):
            player_lbls[p.stem] = p
        for p in d.glob("keypoints/*.txt"):
            kp_lbls[p.stem] = p

    # Only keep frames that have both label files
    stems = sorted(s for s in [p.stem for p in images] if s in player_lbls and s in kp_lbls)
    if not stems:
        raise RuntimeError(f"No complete samples (image + both label files) found in {data_dir}")

    random.seed(42)
    random.shuffle(stems)
    n_val = max(1, int(len(stems) * val_frac))
    val = set(stems[:n_val])
    train = [s for s in stems if s not in val]
    val = [s for s in stems if s in val]
    print(f"  Split: {len(train)} train / {len(val)} val  ({len(stems)} total)")

    img_map = {p.stem: p for p in images}

    for model_type, lbl_map in [("detection", player_lbls), ("keypoints", kp_lbls)]:
        for split, split_stems in [("train", train), ("valid", val)]:
            img_out = out_dir / model_type / "images" / split
            lbl_out = out_dir / model_type / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for stem in split_stems:
                shutil.copy(img_map[stem], img_out / f"{stem}.jpg")
                shutil.copy(lbl_map[stem], lbl_out / f"{stem}.txt")

    return out_dir / "detection", out_dir / "keypoints"


def _write_yolo_yaml(detection_dir: Path, weights: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names  # e.g. {0: 'player'} or {0: 'player', 1: 'goalkeeper', ...}
    yaml_path = detection_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {detection_dir.resolve()}\n"
        "train: images/train\n"
        "val:   images/valid\n"
        f"nc: {len(names)}\n"
        f"names: {list(names.values())}\n"
    )
    return yaml_path


def finetune_detection(detection_dir: Path, weights: str, epochs: int, batch: int) -> Path:
    print(f"\n--- Fine-tuning player detector ---")
    print(f"  weights : {weights}")
    print(f"  data    : {detection_dir}")
    print(f"  epochs  : {epochs}  batch: {batch}")

    yaml = _write_yolo_yaml(detection_dir, weights)

    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.train(
        data=str(yaml),
        epochs=epochs,
        batch=batch,
        imgsz=640,
        mosaic=0.5,
        degrees=5.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        project=str(_ROOT / "models" / "players" / "finetune"),
        name="run",
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"  Done → {best}")
    return best


def finetune_keypoints(keypoints_dir: Path, weights: str, epochs: int, batch: int, lr: float) -> Path:
    print(f"\n--- Fine-tuning pitch keypoint detector ---")
    print(f"  weights : {weights}")
    print(f"  data    : {keypoints_dir}")
    print(f"  epochs  : {epochs}  batch: {batch}  lr: {lr}")

    from torchkick.training import train_pitch_heatmap

    best = train_pitch_heatmap(
        data_dir=str(keypoints_dir),
        base_model=weights,
        epochs=epochs,
        batch_size=batch,
        learning_rate=lr,
        min_keypoints=3,  # relaxed — small dataset may have partial views
        compile_model=False,  # skip compile for quick sanity runs
        save_dir=str(_ROOT / "models" / "keypoints" / "finetune"),
    )
    print(f"  Done → {best}")
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune torchkick models on annotation exports")
    parser.add_argument("--data-dir", required=True, type=Path, help="Annotation export directory")
    parser.add_argument(
        "--player-weights", default=str(_ROOT / "models/players/yolo11l.pt"), help="Base player detector weights"
    )
    parser.add_argument(
        "--pitch-weights", default=str(_ROOT / "models/keypoints/heatmap.pt"), help="Base pitch keypoint weights"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Fine-tuning epochs (default 20)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default 4 — safe for small datasets)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for keypoints (default 1e-3)")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of data to use for validation")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write converted datasets (default: <data-dir>/../finetune_tmp/)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--detection-only", action="store_true")
    group.add_argument("--keypoints-only", action="store_true")

    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise SystemExit(f"[error] data dir not found: {data_dir}")

    out_dir = args.out_dir or data_dir.parent / "finetune_tmp"
    out_dir = out_dir.resolve()

    print(f"Preparing dataset from {data_dir} ...")
    detection_dir, keypoints_dir = _prepare_dataset(data_dir, out_dir, val_frac=args.val_frac)

    if not args.keypoints_only:
        finetune_detection(detection_dir, args.player_weights, args.epochs, args.batch)

    if not args.detection_only:
        finetune_keypoints(keypoints_dir, args.pitch_weights, args.epochs, args.batch, args.lr)

    print("\nAll done.")


if __name__ == "__main__":
    main()
