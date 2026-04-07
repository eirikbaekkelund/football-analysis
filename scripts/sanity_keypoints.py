"""
Sanity-check YOLO-pose pitch keypoint predictions on sample images.

Runs the model on N random images, draws the 32 predicted keypoints
(index + dot) and saves annotated images to an output directory.

Usage:
    python scripts/sanity_keypoints.py \
        --weights /workspace/weights/keypoints/train/weights/best.pt \
        --image-dir /workspace/weights/soccernet_kp_dataset/images/train \
        --output-dir /workspace/logs/keypoint_sanity \
        --n 16
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


# Brief label for each of the 32 RF keypoints
_KP_LABELS = [
    "TL",  # 0  top-left corner
    "LP1",  # 1  left pen box top (goal line)
    "LG1",  # 2  left goal box top (goal line)
    "LG2",  # 3  left goal box bottom (goal line)
    "LP2",  # 4  left pen box bottom (goal line)
    "BL",  # 5  bottom-left corner
    "LGF1",  # 6  left goal box front top
    "LGF2",  # 7  left goal box front bottom
    "LPS",  # 8  left penalty spot
    "LPF1",  # 9  left pen box front top
    "LPI1",  # 10 left pen box inner top
    "LPI2",  # 11 left pen box inner bottom
    "LPF2",  # 12 left pen box front bottom
    "HT",  # 13 halfway line top
    "CCT",  # 14 centre circle top
    "CCB",  # 15 centre circle bottom
    "HB",  # 16 halfway line bottom
    "RPF1",  # 17 right pen box front top
    "RPI1",  # 18 right pen box inner top
    "RPI2",  # 19 right pen box inner bottom
    "RPF2",  # 20 right pen box front bottom
    "RPS",  # 21 right penalty spot
    "RGF1",  # 22 right goal box front top
    "RGF2",  # 23 right goal box front bottom
    "TR",  # 24 top-right corner
    "RP1",  # 25 right pen box top (goal line)
    "RG1",  # 26 right goal box top (goal line)
    "RG2",  # 27 right goal box bottom (goal line)
    "RP2",  # 28 right pen box bottom (goal line)
    "BR",  # 29 bottom-right corner
    "CCL",  # 30 centre circle left
    "CCR",  # 31 centre circle right
]

# Color by zone (BGR)
_KP_COLORS = (
    [(0, 255, 0)] * 6  # 0-5  corners / boundary
    + [(0, 200, 255)] * 7  # 6-12 left box
    + [(255, 255, 0)] * 4  # 13-16 centre
    + [(0, 100, 255)] * 7  # 17-23 right box
    + [(0, 255, 0)] * 6  # 24-29 corners / boundary (right)
    + [(255, 0, 255)] * 2  # 30-31 circle left/right
)


def draw_keypoints(img: np.ndarray, kps: np.ndarray, confs: np.ndarray, conf_thr: float = 0.3) -> np.ndarray:
    """Draw 32 keypoints on img. kps: [32,2] pixel coords, confs: [32]."""
    out = img.copy()
    h, w = out.shape[:2]
    for i, ((x, y), c) in enumerate(zip(kps, confs)):
        if c < conf_thr:
            continue
        px, py = int(x), int(y)
        if not (0 <= px < w and 0 <= py < h):
            continue
        color = _KP_COLORS[i]
        cv2.circle(out, (px, py), 6, color, -1)
        cv2.circle(out, (px, py), 6, (0, 0, 0), 1)
        cv2.putText(out, _KP_LABELS[i], (px + 7, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="/workspace/logs/keypoint_sanity")
    parser.add_argument("--n", type=int, default=16, help="Number of sample images")
    parser.add_argument("--conf", type=float, default=0.3, help="Keypoint confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list(Path(args.image_dir).glob("*.jpg"))
    if not img_paths:
        print(f"No .jpg images found in {args.image_dir}")
        return

    samples = random.sample(img_paths, min(args.n, len(img_paths)))
    print(f"Running on {len(samples)} images → {out_dir}")

    for img_path in samples:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = model.predict(frame, imgsz=args.imgsz, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            cv2.imwrite(str(out_dir / img_path.name), frame)
            continue

        kps = results[0].keypoints.xy[0].cpu().numpy()  # [32, 2]
        confs = results[0].keypoints.conf[0].cpu().numpy()  # [32]

        annotated = draw_keypoints(frame, kps, confs, conf_thr=args.conf)
        cv2.imwrite(str(out_dir / img_path.name), annotated)

    print(f"Saved {len(samples)} annotated images to {out_dir}")


if __name__ == "__main__":
    main()
