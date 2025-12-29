import os
import io
import cv2
import fsspec
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image


def get_jersey_color_class(img_pil, box):
    """
    Determines the color class of a player crop.
    Classes:
    0: Other/Mixed
    1: White
    2: Black
    3: Red
    4: Blue
    5: Yellow
    6: Green
    """
    # Convert PIL to CV2 (RGB -> BGR -> HSV)
    img_np = np.array(img_pil)
    x, y, w, h = map(int, box)

    height, width = img_np.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)

    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0:
        return 0

    ch, cw = crop.shape[:2]
    # Take middle 50% of width, and top 20-60% of height (chest area)
    crop_center = crop[int(ch * 0.2) : int(ch * 0.6), int(cw * 0.25) : int(cw * 0.75)]

    if crop_center.size == 0:
        return 0

    hsv = cv2.cvtColor(crop_center, cv2.COLOR_RGB2HSV)

    mean_h = np.mean(hsv[:, :, 0])  # 0-179
    mean_s = np.mean(hsv[:, :, 1])  # 0-255
    mean_v = np.mean(hsv[:, :, 2])  # 0-255

    if mean_s < 50:
        if mean_v > 180:
            return 1  # White
        if mean_v < 60:
            return 2  # Black
        return 0  # Grey/Other

    if (mean_h < 15) or (mean_h > 165):
        return 3  # Red
    if 95 < mean_h < 135:
        return 4  # Blue
    if 20 < mean_h < 40:
        return 5  # Yellow
    if 40 < mean_h < 85:
        return 6  # Green

    return 0  # Other (Purple, Orange, etc.)


def convert_to_yolo_format(zip_path, output_dir, use_colors=True):
    """
    Extracts images from zip and converts GT to YOLO format.
    YOLO format: class_id x_center y_center width height (normalized 0-1)
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"

    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting data from {zip_path} to YOLO format in {output_dir} (use_colors={use_colors})...")

    # Use fsspec to read zip without extracting everything first
    fs = fsspec.filesystem("zip", fo=zip_path)

    # Find all sequences
    # Structure is usually root/sequence/img1/... and root/sequence/gt/gt.txt
    # fs.ls("") returns a list of strings or dicts depending on version
    root_contents = fs.ls("")
    root_dirs = []
    for p in root_contents:
        # Handle dict output from fs.ls
        path_str = p["name"] if isinstance(p, dict) else p
        if fs.isdir(path_str):
            root_dirs.append(path_str)

    if not root_dirs:
        print("Error: No root directories found in zip.")
        return

    root = root_dirs[0]  # e.g., "train"

    # List sequences inside root
    seq_contents = fs.ls(root)
    sequences = []
    for p in seq_contents:
        path_str = p["name"] if isinstance(p, dict) else p
        if fs.isdir(path_str):
            sequences.append(path_str)

    # Cache track colors to ensure a track ID always gets the same class
    # (Optional, but helps consistency)
    track_color_cache = {}  # { (seq_name, track_id): class_id }

    for seq_path in tqdm(sequences):
        seq_name = Path(seq_path).name
        gt_path = f"{seq_path}/gt/gt.txt"

        if not fs.exists(gt_path):
            continue

        # Read GT
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
                    cls = color_class
                else:
                    # Default to single class 'player'
                    cls = 0

                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h

                cx = np.clip(cx, 0, 1)
                cy = np.clip(cy, 0, 1)
                nw = np.clip(nw, 0, 1)
                nh = np.clip(nh, 0, 1)

                yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(labels_dir / f"{save_name}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

    if use_colors:
        names_yaml = """
        names:
          0: player_other
          1: player_white
          2: player_black
          3: player_red
          4: player_blue
          5: player_yellow
          6: player_green"""
    else:
        names_yaml = """
        names:
          0: player"""

    yaml_content = f"""
        path: {output_dir.absolute()}
        train: images/train
        val: images/train
{names_yaml}
    """

    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print("Conversion complete.")


def train_yolo():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-colors", action="store_true", help="Disable jersey color classification")
    args = parser.parse_args()

    use_colors = not args.no_colors

    # Allow user to specify dataset path via env var or argument, default to local conversion
    DATA_DIR = os.environ.get("YOLO_DATASET_DIR", "yolo_dataset")
    if use_colors:
        DATA_DIR += "_colors"

    ZIP_PATH = "soccernet/tracking/tracking/train.zip"

    if not os.path.exists(DATA_DIR):
        print(f"Dataset not found at {DATA_DIR}. Attempting to convert from {ZIP_PATH}...")
        if os.path.exists(ZIP_PATH):
            convert_to_yolo_format(ZIP_PATH, DATA_DIR, use_colors=use_colors)
        else:
            print(f"Warning: Zip file {ZIP_PATH} not found. Please ensure you have a dataset prepared in {DATA_DIR}")
            # We continue, assuming the user might have mounted data there or will provide it
    else:
        print(f"Dataset found at {DATA_DIR}, skipping conversion.")

    # Load YOLO11 model (nano version as requested, or use 'yolo11m.pt' for medium)
    try:
        model = YOLO("yolo11n.pt")
    except Exception as e:
        print(f"Could not load yolo11n.pt: {e}. Please ensure ultralytics is updated.")
        return

    project_name = "football_yolo_colors" if use_colors else "football_yolo_simple"

    # Train the model
    results = model.train(
        data=f"{DATA_DIR}/dataset.yaml",
        epochs=50,  # Increased epochs as per recommendation
        imgsz=640,
        batch=32,  # Adjusted batch size
        device=0,
        project=project_name,
        name="yolo11n_football",
        exist_ok=True,
        plots=True,
    )

    print(f"Best model saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    train_yolo()
