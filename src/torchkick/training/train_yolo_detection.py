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
import random
import shutil
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
val: images/val
{names_yaml}
"""

    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Conversion complete. Dataset YAML: {output_dir / 'dataset.yaml'}")


def convert_dir_to_yolo_format(
    soccernet_dir: str,
    output_dir: str,
    val_ratio: float = 0.15,
    seed: int = 42,
    frame_stride: int = 1,
) -> None:
    """
    Convert a pre-extracted SoccerNet directory to YOLO format with train/val split.

    Expects layout: <soccernet_dir>/<seq_name>/img1/<frame>.jpg
                    <soccernet_dir>/<seq_name>/gt/gt.txt  (MOT format)

    Args:
        frame_stride: Keep every Nth frame (1 = all frames, 5 = every 5th).
            Use 5 to reduce a ~25 GB dataset to ~5 GB when disk space is limited.
    """
    output_dir = Path(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    root = Path(soccernet_dir)
    # Recursive search mirrors _load_soccernet_dir_source — handles nested layouts
    gt_files = sorted(root.glob("**/gt/gt.txt"))
    sequences = [gt.parent.parent for gt in gt_files]

    rng = random.Random(seed)
    rng.shuffle(sequences)
    n_val = max(1, int(len(sequences) * val_ratio))
    val_seqs = {s.name for s in sequences[:n_val]}

    print(f"Converting {len(sequences)} sequences → {len(sequences) - n_val} train / {n_val} val")

    for seq_dir in tqdm(sequences, desc="Converting"):
        seq_name = seq_dir.name
        split = "val" if seq_name in val_seqs else "train"
        out_images = output_dir / "images" / split
        out_labels = output_dir / "labels" / split

        gt_path = seq_dir / "gt" / "gt.txt"
        img_dir = seq_dir / "img1"

        df = pd.read_csv(
            gt_path,
            header=None,
            names=["frame", "track_id", "x", "y", "w", "h", "conf", "class_id", "visibility", "unused"],
        )

        frame_ids = sorted(df["frame"].unique())
        for i, frame_id in enumerate(frame_ids):
            if i % frame_stride != 0:
                continue

            img_path = img_dir / f"{frame_id:06d}.jpg"
            if not img_path.exists():
                continue

            img = Image.open(img_path)
            img_w, img_h = img.size
            save_name = f"{seq_name}_{frame_id:06d}"

            shutil.copy2(img_path, out_images / f"{save_name}.jpg")

            frame_data = df[df["frame"] == frame_id]
            yolo_lines = []
            for _, row in frame_data.iterrows():
                x, y, w, h = row["x"], row["y"], row["w"], row["h"]
                cx = float(np.clip((x + w / 2) / img_w, 0, 1))
                cy = float(np.clip((y + h / 2) / img_h, 0, 1))
                nw = float(np.clip(w / img_w, 0, 1))
                nh = float(np.clip(h / img_h, 0, 1))
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(out_labels / f"{save_name}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

    yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val
names:
  0: player
"""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Conversion complete. Dataset YAML: {output_dir / 'dataset.yaml'}")


def train_yolo(
    data_zip: Optional[str] = None,
    data_dir: Optional[str] = None,
    soccernet_dir: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 32,
    imgsz: int = 640,
    use_colors: bool = False,
    base_model: str = "yolo11l.pt",
    device: int = 0,
    project: str = "player_tracker",
    frame_stride: int = 1,
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

    # Determine dataset directory — prefer root FS over /workspace to avoid filling 20G volume
    if data_dir is None:
        if os.path.exists("/workspace"):
            default_base = "/yolo_dataset"
        else:
            default_base = "yolo_dataset"
        data_dir = os.environ.get("YOLO_DATASET_DIR", default_base)
        if use_colors:
            data_dir += "_colors"

    # Convert data if needed
    if not os.path.exists(data_dir):
        if soccernet_dir is not None:
            convert_dir_to_yolo_format(soccernet_dir, data_dir, frame_stride=frame_stride)
        else:
            if data_zip is None:
                data_zip = "soccernet/tracking/tracking/train.zip"
            if not os.path.exists(data_zip):
                print("Downloading SoccerNet tracking data...")
                from torchkick.soccernet import download_soccernet

                download_soccernet("tracking", "soccernet/tracking")
            convert_to_yolo_format(data_zip, data_dir, use_colors=use_colors)
    else:
        print(f"Using existing dataset: {data_dir}")

    # Load and train model
    model = YOLO(base_model)

    model_tag = Path(base_model).stem  # e.g. "yolo11l"
    project_name = f"{project}_{model_tag}"
    if use_colors:
        project_name += "_colors"

    # Support both SoccerNet-converted (dataset.yaml) and Roboflow (data.yaml) layouts
    yaml_path = Path(data_dir) / "dataset.yaml"
    if not yaml_path.exists():
        yaml_path = Path(data_dir) / "data.yaml"

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project_name,
        name=f"{model_tag}_football",
        exist_ok=True,
        plots=True,
    )

    best_weights = f"{results.save_dir}/weights/best.pt"
    print(f"Training complete. Best model: {best_weights}")
    return best_weights


# ---------------------------------------------------------------------------
# SoccerNet calibration → YOLO-pose (32-keypoint Roboflow schema)
# ---------------------------------------------------------------------------

# 26 of 32 Roboflow pitch keypoints are directly derivable from SoccerNet
# line-class endpoints (sorted by image x then y).
# Keys: (LINE_CLASS_NAME, sorted_endpoint_index)  →  RF vertex index (0-31)
# Missing RF indices: 8, 10, 11, 18, 19, 21 (penalty spots + inner box intersections)
_SOCCERNET_TO_RF_VERTEX: dict = {
    # Pitch corners / boundary lines
    ("Side line top", 0): 0,
    ("Side line top", 1): 24,
    ("Side line bottom", 0): 5,
    ("Side line bottom", 1): 29,
    ("Side line left", 0): 0,
    ("Side line left", 1): 5,
    ("Side line right", 0): 24,
    ("Side line right", 1): 29,
    # Halfway line
    ("Middle line", 0): 13,
    ("Middle line", 1): 16,
    # Left penalty area
    ("Big rect. left top", 0): 1,
    ("Big rect. left top", 1): 9,
    ("Big rect. left bottom", 0): 4,
    ("Big rect. left bottom", 1): 12,
    ("Big rect. left main", 0): 9,
    ("Big rect. left main", 1): 12,
    # Left goal area
    ("Small rect. left top", 0): 2,
    ("Small rect. left top", 1): 6,
    ("Small rect. left bottom", 0): 3,
    ("Small rect. left bottom", 1): 7,
    ("Small rect. left main", 0): 6,
    ("Small rect. left main", 1): 7,
    # Right penalty area (sorted by x: front end first, goal line end second)
    ("Big rect. right top", 0): 17,
    ("Big rect. right top", 1): 25,
    ("Big rect. right bottom", 0): 20,
    ("Big rect. right bottom", 1): 28,
    ("Big rect. right main", 0): 17,
    ("Big rect. right main", 1): 20,
    # Right goal area
    ("Small rect. right top", 0): 22,
    ("Small rect. right top", 1): 26,
    ("Small rect. right bottom", 0): 23,
    ("Small rect. right bottom", 1): 27,
    ("Small rect. right main", 0): 22,
    ("Small rect. right main", 1): 23,
}

# Symmetric flip pairs for horizontal augmentation (YOLO-pose flip_idx field)
_RF_FLIP_IDX: list = [
    24,
    25,
    26,
    27,
    28,
    29,  # 0-5 → 24-29
    22,
    23,
    21,  # 6-8 → 22, 23, 21
    17,
    18,
    19,
    20,  # 9-12 → 17-20
    13,
    14,
    15,
    16,  # 13-16 self (halfway + circle top/bottom)
    9,
    10,
    11,
    12,  # 17-20 → 9-12
    8,  # 21 → 8
    6,
    7,  # 22-23 → 6-7
    0,
    1,
    2,
    3,
    4,
    5,  # 24-29 → 0-5
    31,
    30,  # 30-31 → 31, 30 (circle left ↔ right)
]


def convert_soccernet_calibration_to_yolo_pose(
    zip_path: str,
    output_dir: str,
    split: str = "train",
    imgsz: int = 640,
    min_keypoints: int = 6,
) -> int:
    """
    Convert one SoccerNet calibration zip to YOLO-pose format (32-keypoint schema).

    26 of the 32 Roboflow pitch keypoints are derived from SoccerNet line
    endpoints.  The 6 unreachable ones (penalty spots RF[8,21] and inner box
    corners RF[10,11,18,19]) are written with visibility=0.

    When the same RF vertex appears in multiple line classes (e.g. RF[0] is
    both "Side line top" pt[0] and "Side line left" pt[0]), pixel coordinates
    are averaged across all contributing lines.

    Output layout::

        output_dir/
          images/<split>/<frame_id>.jpg
          labels/<split>/<frame_id>.txt  # one row: 0 0.5 0.5 1.0 1.0 kp0…kp31

    Args:
        zip_path: Path to the SoccerNet calibration zip for one split.
        output_dir: Root directory for the YOLO-pose dataset.
        split: Subfolder name, e.g. ``"train"`` or ``"valid"``.
        imgsz: Resize images to this square size (default 640).
        min_keypoints: Minimum number of visible keypoints required to keep a
            sample.  Close-up / goal-mouth shots typically yield 2-4 visible
            points; broadcast wide shots yield 8-15+.  Samples below this
            threshold are skipped (default 6).

    Returns:
        Number of successfully converted samples.
    """
    import json
    from collections import defaultdict
    from io import BytesIO

    import fsspec
    from PIL import Image

    out = Path(output_dir)
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    with fsspec.open(zip_path, "rb") as f:
        zip_fs = fsspec.filesystem("zip", fo=f)
        all_files = zip_fs.ls("", detail=False)
        root_dirs = [p for p in all_files if zip_fs.isdir(p)]
        root = root_dirs[0].rstrip("/") if root_dirs else ""
        prefix = f"{root}/" if root else ""
        entries = zip_fs.ls(prefix, detail=False)
        json_paths = sorted(e for e in entries if e.endswith(".json"))

        converted = 0
        for json_path in tqdm(json_paths, desc=f"  {split}", unit="img"):
            frame_id = Path(json_path).stem
            img_path = json_path.replace(".json", ".jpg")
            if not zip_fs.exists(img_path):
                continue

            with zip_fs.open(json_path, "r") as jf:
                annotations = json.load(jf)
            if not annotations:
                continue

            # Load + resize image
            with zip_fs.open(img_path, "rb") as imgf:
                img = Image.open(BytesIO(imgf.read())).convert("RGB")
            img = img.resize((imgsz, imgsz), Image.BILINEAR)
            img.save(img_dir / f"{frame_id}.jpg", quality=90)

            # Build [32, 3] keypoint array: (x_norm, y_norm, visibility)
            # Accumulate multiple contributions per RF index then average
            accum: dict = defaultdict(list)  # rf_idx → [(x, y), ...]

            for class_name, pts in annotations.items():
                class_name = class_name.strip()
                if not pts:
                    continue

                if class_name == "Circle central":
                    # Find the 4 cardinal points by position
                    sorted_pts = sorted(pts, key=lambda p: (p["x"], p["y"]))
                    if len(sorted_pts) >= 2:
                        # left-most x → RF[30], right-most x → RF[31]
                        accum[30].append((sorted_pts[0]["x"], sorted_pts[0]["y"]))
                        accum[31].append((sorted_pts[-1]["x"], sorted_pts[-1]["y"]))
                    by_y = sorted(pts, key=lambda p: p["y"])
                    if len(by_y) >= 2:
                        # top-most y → RF[14], bottom-most y → RF[15]
                        accum[14].append((by_y[0]["x"], by_y[0]["y"]))
                        accum[15].append((by_y[-1]["x"], by_y[-1]["y"]))
                    continue

                # Non-circle: sort by (x, y) for canonical endpoint order
                sorted_pts = sorted(pts, key=lambda p: (p["x"], p["y"]))
                for ep_idx in range(min(2, len(sorted_pts))):
                    key = (class_name, ep_idx)
                    rf_idx = _SOCCERNET_TO_RF_VERTEX.get(key)
                    if rf_idx is not None:
                        p = sorted_pts[ep_idx]
                        accum[rf_idx].append((p["x"], p["y"]))

            # Average duplicates, build flat label
            kp = np.zeros((32, 3), dtype=np.float32)  # (x, y, vis)
            for rf_idx, coords in accum.items():
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                kp[rf_idx, 0] = float(np.clip(np.mean(xs), 0.0, 1.0))
                kp[rf_idx, 1] = float(np.clip(np.mean(ys), 0.0, 1.0))
                kp[rf_idx, 2] = 2.0  # labeled and visible

            # Skip close-up shots with too few visible landmarks
            if int(np.sum(kp[:, 2] > 0)) < min_keypoints:
                continue

            # YOLO-pose row: class cx cy w h  kp0x kp0y kp0v ... kp31x kp31y kp31v
            row = [0, 0.5, 0.5, 1.0, 1.0]
            for i in range(32):
                row.extend([kp[i, 0], kp[i, 1], int(kp[i, 2])])

            with open(lbl_dir / f"{frame_id}.txt", "w") as lf:
                lf.write(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + "\n")

            converted += 1

    return converted


def build_soccernet_keypoint_dataset(
    calibration_dir: str,
    output_dir: str,
    splits: Optional[list] = None,
    imgsz: int = 640,
    min_keypoints: int = 6,
) -> str:
    """
    Convert SoccerNet calibration splits to a YOLO-pose dataset and write a
    ``dataset.yaml`` file ready for ``train_yolo_keypoints``.

    Looks for ``<calibration_dir>/<split>.zip`` for each requested split.
    Skips splits whose zip files don't exist.

    Args:
        calibration_dir: Directory containing ``train.zip``, ``valid.zip``, etc.
        output_dir: Root directory for the converted dataset.
        splits: List of split names to convert (default ``["train", "valid"]``).
        imgsz: Resize images to this square size.

    Returns:
        Absolute path to the generated ``dataset.yaml``.

    Example:
        >>> yaml_path = build_soccernet_keypoint_dataset(
        ...     calibration_dir="data/soccernet/calibration",
        ...     output_dir="data/pitch_keypoints",
        ... )
        >>> weights = train_yolo_keypoints(data_yaml=yaml_path, epochs=100)
    """
    if splits is None:
        splits = ["train", "valid"]

    calib = Path(calibration_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    yaml_splits: dict = {}
    total = 0
    for split in splits:
        zip_path = calib / f"{split}.zip"
        if not zip_path.exists():
            print(f"  [skip] {zip_path} not found")
            continue
        print(f"Converting {zip_path.name} …")
        n = convert_soccernet_calibration_to_yolo_pose(
            str(zip_path), str(out), split=split, imgsz=imgsz, min_keypoints=min_keypoints
        )
        print(f"  → {n} samples written to {out}/images/{split}/")
        yaml_splits[split] = f"images/{split}"
        total += n

    if not yaml_splits:
        raise RuntimeError(f"No calibration zips found in {calibration_dir}")

    # When no valid split exists, carve out 15% of train as a held-out val set.
    # Files are moved (not copied) so there is zero leakage.
    if "valid" not in yaml_splits and "val" not in yaml_splits and "train" in yaml_splits:
        import random as _random

        _random.seed(42)
        train_img_dir = out / "images" / "train"
        train_lbl_dir = out / "labels" / "train"
        val_img_dir = out / "images" / "valid"
        val_lbl_dir = out / "labels" / "valid"
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)

        all_imgs = sorted(train_img_dir.glob("*.jpg"))
        n_val = max(1, int(len(all_imgs) * 0.15))
        val_imgs = _random.sample(all_imgs, n_val)

        for img_p in val_imgs:
            lbl_p = train_lbl_dir / img_p.with_suffix(".txt").name
            img_p.rename(val_img_dir / img_p.name)
            if lbl_p.exists():
                lbl_p.rename(val_lbl_dir / lbl_p.name)

        yaml_splits["valid"] = "images/valid"
        print(f"  Auto-split: {n_val} samples → valid, {len(all_imgs) - n_val} remain in train")

    # Write dataset.yaml
    yaml_path = out / "dataset.yaml"
    train_key = yaml_splits.get("train", next(iter(yaml_splits.values())))
    val_key = yaml_splits.get("valid", yaml_splits.get("val", train_key))

    yaml_lines = [
        f"path: {out.resolve()}",
        f"train: {train_key}",
        f"val: {val_key}",
        "",
        f"kpt_shape: [32, 3]",
        f"flip_idx: {_RF_FLIP_IDX}",
        "",
        "nc: 1",
        "names: ['pitch']",
    ]
    yaml_path.write_text("\n".join(yaml_lines) + "\n")
    print(f"\nDataset YAML → {yaml_path}  ({total} total samples)")
    return str(yaml_path)


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
