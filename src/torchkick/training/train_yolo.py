"""
YOLO player detection training.

This module provides utilities for training YOLO models on football
player detection using SoccerNet tracking data.

Example:
    >>> from torchkick.training import train_yolo
    >>> 
    >>> # Train with default settings
    >>> train_yolo(
    ...     data_zip="soccernet/tracking/train.zip",
    ...     epochs=50,
    ...     use_colors=False,  # Single-class player detection
    ... )
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import fsspec
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_jersey_color_class(img_pil: Image.Image, box: Tuple[int, int, int, int]) -> int:
    """
    Determine jersey color class from player crop.

    Color classes:
        0: Other/Mixed
        1: White
        2: Black
        3: Red
        4: Blue
        5: Yellow
        6: Green

    Args:
        img_pil: PIL Image of the full frame.
        box: (x, y, w, h) bounding box in pixels.

    Returns:
        Color class ID (0-6).
    """
    img_np = np.array(img_pil)
    x, y, w, h = map(int, box)

    height, width = img_np.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)

    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0:
        return 0

    ch, cw = crop.shape[:2]
    crop_center = crop[int(ch * 0.2) : int(ch * 0.6), int(cw * 0.25) : int(cw * 0.75)]

    if crop_center.size == 0:
        return 0

    hsv = cv2.cvtColor(crop_center, cv2.COLOR_RGB2HSV)

    mean_h = np.mean(hsv[:, :, 0])  # 0-179
    mean_s = np.mean(hsv[:, :, 1])  # 0-255
    mean_v = np.mean(hsv[:, :, 2])  # 0-255

    # Low saturation = grayscale
    if mean_s < 50:
        if mean_v > 180:
            return 1  # White
        if mean_v < 60:
            return 2  # Black
        return 0  # Gray/Other

    # Classify by hue
    if (mean_h < 15) or (mean_h > 165):
        return 3  # Red
    if 95 < mean_h < 135:
        return 4  # Blue
    if 20 < mean_h < 40:
        return 5  # Yellow
    if 40 < mean_h < 85:
        return 6  # Green

    return 0  # Other


def convert_to_yolo_format(
    zip_path: str,
    output_dir: str,
    use_colors: bool = True,
) -> None:
    """
    Convert SoccerNet tracking data to YOLO format.

    Extracts images and labels from the tracking zip file,
    converting ground truth annotations to YOLO format.

    Args:
        zip_path: Path to SoccerNet tracking zip file.
        output_dir: Output directory for YOLO dataset.
        use_colors: If True, classify by jersey color (7 classes).
            If False, single "player" class.

    The output structure is:
        output_dir/
            images/train/
            labels/train/
            dataset.yaml
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {zip_path} to YOLO format in {output_dir}")
    print(f"  Color classification: {use_colors}")

    fs = fsspec.filesystem("zip", fo=zip_path)

    # Find root directories
    root_contents = fs.ls("")
    root_dirs = []
    for p in root_contents:
        path_str = p["name"] if isinstance(p, dict) else p
        if fs.isdir(path_str):
            root_dirs.append(path_str)

    if not root_dirs:
        raise ValueError("No directories found in zip file")

    root = root_dirs[0]

    # Find sequences
    seq_contents = fs.ls(root)
    sequences = []
    for p in seq_contents:
        path_str = p["name"] if isinstance(p, dict) else p
        if fs.isdir(path_str):
            sequences.append(path_str)

    # Cache track colors for consistency
    track_color_cache = {}

    for seq_path in tqdm(sequences, desc="Converting sequences"):
        seq_name = Path(seq_path).name
        gt_path = f"{seq_path}/gt/gt.txt"

        if not fs.exists(gt_path):
            continue

        with fs.open(gt_path, "rb") as f:
            df = pd.read_csv(
                f,
                header=None,
                names=["frame", "track_id", "x", "y", "w", "h", "conf", "class_id", "visibility", "unused"],
            )

        for frame_id in df["frame"].unique():
            img_path_zip = f"{seq_path}/img1/{frame_id:06d}.jpg"

            if not fs.exists(img_path_zip):
                continue

            with fs.open(img_path_zip, "rb") as f:
                img_bytes = f.read()
                img = Image.open(io.BytesIO(img_bytes))
                img_w, img_h = img.size

            save_name = f"{seq_name}_{frame_id:06d}"
            with open(images_dir / f"{save_name}.jpg", "wb") as f:
                f.write(img_bytes)

            frame_data = df[df["frame"] == frame_id]

            yolo_lines = []
            for _, row in frame_data.iterrows():
                track_id = row["track_id"]
                cache_key = (seq_name, track_id)

                x, y, w, h = row["x"], row["y"], row["w"], row["h"]

                if use_colors:
                    if cache_key in track_color_cache:
                        color_class = track_color_cache[cache_key]
                    else:
                        color_class = get_jersey_color_class(img, (x, y, w, h))
                        if color_class != 0:
                            track_color_cache[cache_key] = color_class
                    cls_assignment = color_class
                else:
                    cls_assignment = 0

                # Convert to YOLO format (center, normalized)
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h

                cx = np.clip(cx, 0, 1)
                cy = np.clip(cy, 0, 1)
                nw = np.clip(nw, 0, 1)
                nh = np.clip(nh, 0, 1)

                yolo_lines.append(f"{cls_assignment} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(labels_dir / f"{save_name}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

    # Write dataset YAML
    if use_colors:
        names_yaml = """names:
  0: player_other
  1: player_white
  2: player_black
  3: player_red
  4: player_blue
  5: player_yellow
  6: player_green"""
    else:
        names_yaml = """names:
  0: player"""

    yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/train
{names_yaml}
"""

    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Conversion complete. Dataset YAML: {output_dir / 'dataset.yaml'}")


def train_yolo(
    data_zip: Optional[str] = None,
    data_dir: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 256,
    imgsz: int = 640,
    use_colors: bool = False,
    base_model: str = "yolo11n.pt",
    device: int = 0,
    project: str = "player_tracker",
) -> str:
    """
    Train YOLO model for player detection.

    Args:
        data_zip: Path to SoccerNet tracking zip (will convert to YOLO format).
        data_dir: Pre-converted YOLO dataset directory.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        imgsz: Input image size.
        use_colors: Train with jersey color classification.
        base_model: Base YOLO model to finetune.
        device: CUDA device index.
        project: Project name for saving results.

    Returns:
        Path to best model weights.

    Example:
        >>> weights = train_yolo(
        ...     data_zip="soccernet/tracking/train.zip",
        ...     epochs=50,
        ...     batch_size=256,
        ... )
        >>> print(f"Best model: {weights}")
    """
    from ultralytics import YOLO

    # Determine dataset directory
    if data_dir is None:
        default_base = "yolo_dataset"
        if os.path.exists("/workspace"):
            default_base = "/workspace/yolo_dataset"
        data_dir = os.environ.get("YOLO_DATASET_DIR", default_base)
        if use_colors:
            data_dir += "_colors"

    # Convert data if needed
    if not os.path.exists(data_dir):
        if data_zip is None:
            data_zip = "soccernet/tracking/tracking/train.zip"

        if not os.path.exists(data_zip):
            print(f"Downloading SoccerNet tracking data...")
            from torchkick.soccernet import download_soccernet

            download_soccernet("tracking", "soccernet/tracking")

        convert_to_yolo_format(data_zip, data_dir, use_colors=use_colors)
    else:
        print(f"Using existing dataset: {data_dir}")

    # Load and train model
    model = YOLO(base_model)

    project_name = f"{project}_yolo11n"
    if use_colors:
        project_name += "_colors"

    results = model.train(
        data=f"{data_dir}/dataset.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project_name,
        name="yolo11n_football",
        exist_ok=True,
        plots=True,
    )

    best_weights = f"{results.save_dir}/weights/best.pt"
    print(f"Training complete. Best model: {best_weights}")
    return best_weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO player detector")
    parser.add_argument("--data", type=str, help="Path to SoccerNet zip or YOLO dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--colors", action="store_true", help="Train with jersey colors")
    args = parser.parse_args()

    train_yolo(
        data_zip=args.data if args.data and args.data.endswith(".zip") else None,
        data_dir=args.data if args.data and not args.data.endswith(".zip") else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_colors=args.colors,
    )
