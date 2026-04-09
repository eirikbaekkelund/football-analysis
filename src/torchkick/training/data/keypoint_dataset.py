"""
Keypoint dataset with augmentation for ViTPose-L training.

Wraps the existing LineKeypointDataset from soccernet/ with augmentations
suitable for pitch landmark detection:
    - Horizontal flip with symmetric keypoint mirroring
    - Random crop (simulate partial field views)
    - Color jitter / brightness variation
    - Resize to ViTPose input size (192x256 default)

Example:
    >>> dataset = KeypointAugDataset("data/soccernet/calibration/train.zip", augment=True)
    >>> image, keypoints, visibility = dataset[0]
    >>> # image: torch.Tensor [3, 256, 192]
    >>> # keypoints: torch.Tensor [29, 2]  pixel coords normalized to [0,1]
    >>> # visibility: torch.Tensor [29]  binary
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Standard ViTPose input size: (width, height)
DEFAULT_INPUT_SIZE = (192, 256)


class KeypointAugDataset(Dataset):
    """
    Augmented pitch keypoint dataset for ViTPose-L training.

    Wraps torchkick.soccernet.LineKeypointDataset with albumentations-based
    augmentation including horizontal flip with keypoint mirroring.

    Args:
        data_zip: Path to SoccerNet calibration zip file.
        input_size: (width, height) for the model input.
        augment: Apply augmentation pipeline.
        split: Dataset split prefix ("train", "test", "challenge").
    """

    # Pairs of keypoints that are mirrored on horizontal flip.
    # Format: (left_idx, right_idx) — indices into the 29-keypoint array.
    # These correspond to LINE_CLASSES in soccernet/calibration_data.py.
    # Left/right is from the broadcast camera perspective.
    FLIP_PAIRS = [
        (0, 1),  # Left/right corner flags
        (2, 3),  # Left/right goal posts (near)
        (4, 5),  # Left/right goal posts (far)
        (6, 7),  # Left/right penalty spots
        (8, 9),  # Left/right 18-yard box corners (near)
        (10, 11),  # Left/right 18-yard box corners (far)
        (12, 13),  # Left/right 6-yard box corners (near)
        (14, 15),  # Left/right 6-yard box corners (far)
    ]

    def __init__(
        self,
        data_zip: str,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        augment: bool = True,
        split: str = "train",
    ) -> None:
        self.input_size = input_size  # (W, H)
        self.augment = augment

        from torchkick.soccernet import LineKeypointDataset

        self._base = LineKeypointDataset(data_zip, split=split)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self._base[idx]
        # item: {"image": ndarray HxWx3, "keypoints": ndarray [29,2], "visibility": ndarray [29]}
        image = item["image"]
        kps = item["keypoints"].copy()  # [29, 2] in pixel coords
        vis = item["visibility"].copy()  # [29] binary

        if self.augment:
            image, kps, vis = self._augment(image, kps, vis)

        # Resize to model input size
        h_orig, w_orig = image.shape[:2]
        image = cv2.resize(image, self.input_size)  # (W, H)
        w_new, h_new = self.input_size
        if h_orig > 0 and w_orig > 0:
            kps[:, 0] *= w_new / w_orig
            kps[:, 1] *= h_new / h_orig

        # Normalize keypoints to [0, 1]
        kps_normalized = kps.copy()
        kps_normalized[:, 0] /= w_new
        kps_normalized[:, 1] /= h_new
        kps_normalized = np.clip(kps_normalized, 0, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        kps_tensor = torch.from_numpy(kps_normalized.astype(np.float32))
        vis_tensor = torch.from_numpy(vis.astype(np.float32))

        return image_tensor, kps_tensor, vis_tensor

    def _augment(
        self,
        image: np.ndarray,
        kps: np.ndarray,
        vis: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            import albumentations as A

            transform = A.Compose(
                [
                    A.RandomCrop(
                        height=int(image.shape[0] * 0.9),
                        width=int(image.shape[1] * 0.9),
                        p=0.3,
                    ),
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                    A.GaussNoise(var_limit=(5, 25), p=0.2),
                    A.MotionBlur(blur_limit=5, p=0.2),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
            kps_list = [(float(kps[i, 0]), float(kps[i, 1])) for i in range(len(kps))]
            result = transform(image=image, keypoints=kps_list)
            image = result["image"]
            result_kps = result["keypoints"]
            h_new, w_new = image.shape[:2]
            for i, (x, y) in enumerate(result_kps):
                kps[i, 0] = x
                kps[i, 1] = y
                if x < 0 or x >= w_new or y < 0 or y >= h_new:
                    vis[i] = 0
        except ImportError:
            pass

        return image, kps, vis
