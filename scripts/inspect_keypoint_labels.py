"""
Visualize ground-truth YOLO-pose labels from the converted SoccerNet dataset.

Reads the .txt label files directly (no model needed) and overlays the 32
keypoints on the source images so we can verify the converter mapping is correct.

Usage:
    python scripts/inspect_keypoint_labels.py \
        --image-dir /workspace/weights/soccernet_kp_dataset/images/train \
        --label-dir /workspace/weights/soccernet_kp_dataset/labels/train \
        --output-dir /workspace/logs/label_inspect \
        --n 20
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

_KP_LABELS = [
    "TL", "LP1", "LG1", "LG2", "LP2", "BL",
    "LGF1", "LGF2", "LPS", "LPF1", "LPI1", "LPI2", "LPF2",
    "HT", "CCT", "CCB", "HB",
    "RPF1", "RPI1", "RPI2", "RPF2", "RPS", "RGF1", "RGF2",
    "TR", "RP1", "RG1", "RG2", "RP2", "BR",
    "CCL", "CCR",
]

_KP_COLORS = (
    [(0, 255, 0)] * 6       # 0-5  corners / boundary
    + [(0, 200, 255)] * 7   # 6-12 left box
    + [(255, 255, 0)] * 4   # 13-16 centre
    + [(0, 100, 255)] * 7   # 17-23 right box
    + [(0, 255, 0)] * 6     # 24-29 corners / boundary (right)
    + [(255, 0, 255)] * 2   # 30-31 circle left/right
)


def draw_gt_keypoints(img: np.ndarray, label_path: Path) -> np.ndarray:
    """Read a YOLO-pose .txt label and draw all visible keypoints."""
    if not label_path.exists():
        return img

    h, w = img.shape[:2]
    out = img.copy()

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 + 32 * 3:
                continue
            # parts: class cx cy bw bh  kp0x kp0y kp0v  kp1x kp1y kp1v  ...
            kps_flat = list(map(float, parts[5:]))
            for i in range(32):
                x_n, y_n, vis = kps_flat[i*3], kps_flat[i*3+1], kps_flat[i*3+2]
                if vis < 1:
                    continue
                px, py = int(x_n * w), int(y_n * h)
                if not (0 <= px < w and 0 <= py < h):
                    continue
                color = _KP_COLORS[i]
                cv2.circle(out, (px, py), 7, color, -1)
                cv2.circle(out, (px, py), 7, (0, 0, 0), 1)
                cv2.putText(out, _KP_LABELS[i], (px + 8, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", default="/workspace/logs/label_inspect")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    img_dir = Path(args.image_dir)
    lbl_dir = Path(args.label_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = list(img_dir.glob("*.jpg"))
    samples = random.sample(imgs, min(args.n, len(imgs)))
    print(f"Inspecting {len(samples)} ground-truth labels → {out_dir}")

    for img_path in samples:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        annotated = draw_gt_keypoints(frame, lbl_path)
        cv2.imwrite(str(out_dir / img_path.name), annotated)

    print("Done.")


if __name__ == "__main__":
    main()
