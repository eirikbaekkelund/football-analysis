"""
Pitch heatmap dataset for DINOv2-based keypoint detector training.

Reads YOLO-pose format label files (32-keypoint Roboflow schema) and converts
them to Gaussian heatmap targets for heatmap regression training.

Label format (one line per image):
    0 cx cy w h  kp0x kp0y kp0v  kp1x kp1y kp1v  ... kp31x kp31y kp31v
    (all coords normalised [0,1], visibility v ∈ {0, 2})

Example:
    >>> ds = PitchHeatmapDataset("data/soccernet_kp_dataset", split="train")
    >>> img, heatmaps, visibility, pitch_label = ds[0]
    >>> # img:         torch.Tensor [3, 560, 560]
    >>> # heatmaps:    torch.Tensor [32, 320, 320]
    >>> # visibility:  torch.Tensor [32]   binary
    >>> # pitch_label: torch.Tensor [1]    broadcast-view flag
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 32-keypoint Roboflow horizontal-flip permutation.
# flip_idx[i] = j means "after H-flip, keypoint i should be at slot j".
# Copied verbatim from train_yolo_detection.py:_RF_FLIP_IDX
# ---------------------------------------------------------------------------
_RF_FLIP_IDX: List[int] = [
    24,
    25,
    26,
    27,
    28,
    29,  # 0-5  → 24-29
    22,
    23,
    21,  # 6-8  → 22, 23, 21
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
    8,  # 21   → 8
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

# ImageNet normalization (DINOv2 pretraining statistics)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_KEYPOINTS = 32
INPUT_SIZE = 560  # must be multiple of 14 (DINOv2 patch size)
HEATMAP_SIZE = 320  # decoder output size (INPUT_SIZE / 8 * 4 = 560/14*8 ≈ 320)


def _gaussian_heatmap(cx: float, cy: float, sigma: float, size: int) -> np.ndarray:
    """Generate a 2-D Gaussian peak at (cx, cy) in a [size × size] heatmap."""
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    hm = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    return hm.astype(np.float32)


def _parse_label_line(line: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Parse one YOLO-pose label line into keypoint coords and visibility.

    Returns:
        kp_xy:  [32, 2]  normalised [0,1] coordinates
        kp_vis: [32]     binary visibility (1 = visible, 0 = invisible)
        or None if the line is malformed.
    """
    parts = line.strip().split()
    if len(parts) < 5 + NUM_KEYPOINTS * 3:
        return None

    flat = list(map(float, parts[5:]))
    kp_xy = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    kp_vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

    for i in range(NUM_KEYPOINTS):
        x, y, v = flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]
        kp_xy[i] = [x, y]
        kp_vis[i] = 1.0 if v > 0 else 0.0

    return kp_xy, kp_vis


class PitchHeatmapDataset(Dataset):
    """
    Pitch keypoint dataset producing Gaussian heatmap targets.

    Reads YOLO-pose label files and emits:
        - image tensor          [3, INPUT_SIZE, INPUT_SIZE]
        - heatmap targets       [32, HEATMAP_SIZE, HEATMAP_SIZE]
        - visibility mask       [32]
        - pitch presence label  [1]  (1.0 = broadcast view, 0.0 = close-up)

    Args:
        data_dir:      Root of the converted YOLO-pose dataset (contains
                       images/{split}/ and labels/{split}/).
        split:         "train" or "valid".
        augment:       Apply photometric augmentation pipeline.
        sigma:         Gaussian sigma in heatmap pixels (default 2.5).
        min_keypoints: Skip samples with fewer visible keypoints.  Applied
                       at construction time so __len__ / __getitem__ are stable.
        input_size:    Square model input dimension (must be multiple of 14).
        heatmap_size:  Square heatmap output dimension.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        augment: bool = True,
        sigma: float = 2.5,
        min_keypoints: int = 6,
        input_size: int = INPUT_SIZE,
        heatmap_size: int = HEATMAP_SIZE,
    ) -> None:
        self.augment = augment
        self.sigma = sigma
        self.min_kp = min_keypoints
        self.input_size = input_size
        self.heatmap_size = heatmap_size

        root = Path(data_dir)
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {lbl_dir}")

        # Build (image_path, label_path) pairs, filtering on min_keypoints
        self._samples: List[Tuple[Path, Path]] = []
        skipped = 0
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                skipped += 1
                continue
            with open(lbl_path) as f:
                line = f.readline()
            parsed = _parse_label_line(line)
            if parsed is None:
                skipped += 1
                continue
            _, kp_vis = parsed
            if int(kp_vis.sum()) < min_keypoints:
                skipped += 1
                continue
            self._samples.append((img_path, lbl_path))

        if not self._samples:
            raise RuntimeError(f"No valid samples found in {img_dir} after min_keypoints={min_keypoints} filter.")
        print(
            f"PitchHeatmapDataset [{split}]: {len(self._samples)} samples "
            f"({skipped} skipped by min_keypoints filter)"
        )

        # Photometric augmentation only — no geometric transforms
        self._aug = A.Compose(
            [
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08, p=0.7),
                A.RandomGamma(gamma_limit=(75, 130), p=0.4),
                A.GaussNoise(std_range=(0.02, 0.11), p=0.3),
                A.MotionBlur(blur_limit=7, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            ]
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self._samples[idx]

        # --- load image ---
        frame = cv2.imread(str(img_path))
        if frame is None:
            frame = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- load keypoints ---
        with open(lbl_path) as f:
            line = f.readline()
        parsed = _parse_label_line(line)
        if parsed is None:
            kp_xy = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
            kp_vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        else:
            kp_xy, kp_vis = parsed

        # --- photometric augmentation ---
        if self.augment:
            frame = self._aug(image=frame)["image"]

        # --- resize to model input size ---
        frame = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        # --- normalise to tensor ---
        img_f = frame.astype(np.float32) / 255.0
        img_f = (img_f - _IMAGENET_MEAN) / _IMAGENET_STD
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1))  # [3, H, W]

        # --- build Gaussian heatmap targets ---
        hm_size = self.heatmap_size
        sigma = self.sigma
        heatmaps = np.zeros((NUM_KEYPOINTS, hm_size, hm_size), dtype=np.float32)

        for i in range(NUM_KEYPOINTS):
            if kp_vis[i] < 0.5:
                continue
            cx = kp_xy[i, 0] * hm_size  # position in heatmap space
            cy = kp_xy[i, 1] * hm_size
            # skip if centre falls outside the heatmap
            if not (0.0 <= cx < hm_size and 0.0 <= cy < hm_size):
                kp_vis[i] = 0.0
                continue
            heatmaps[i] = _gaussian_heatmap(cx, cy, sigma, hm_size)

        hm_t = torch.from_numpy(heatmaps)  # [32, H, W]
        vis_t = torch.from_numpy(kp_vis)  # [32]
        pitch_label = torch.tensor([1.0], dtype=torch.float32)  # always 1 after filter

        return img_t, hm_t, vis_t, pitch_label
