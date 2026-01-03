"""
PyTorch Dataset for SoccerNet player tracking data.

This module provides Dataset classes for loading player tracking annotations
from SoccerNet zip files, with support for multiple bounding box formats
and optional jersey color extraction.

Example:
    >>> from torchkick.soccernet import PlayerTrackingDataset, tracking_collate_fn
    >>> from torch.utils.data import DataLoader
    >>> 
    >>> dataset = PlayerTrackingDataset(
    ...     "soccernet/tracking/train.zip",
    ...     bbox_format="xyxy",
    ...     extract_colors=True,
    ... )
    >>> loader = DataLoader(dataset, batch_size=8, collate_fn=tracking_collate_fn)
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Literal, Optional

import cv2
import fsspec
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PlayerTrackingDataset(Dataset):
    """
    PyTorch Dataset for SoccerNet player tracking data.

    Loads frames and annotations from SoccerNet tracking zip files without
    extracting to disk. Supports multiple bounding box output formats.

    Args:
        zip_path: Path to the SoccerNet tracking zip file.
        transform: Optional transform to apply to images.
        bbox_format: Output bounding box format:
            - "xywh": [x, y, width, height] - top-left corner + dimensions (MOT format)
            - "xyxy": [x1, y1, x2, y2] - top-left + bottom-right corners
            - "cxcywh": [cx, cy, width, height] - center + dimensions
        debug: Print debug information during loading.
        extract_colors: Extract jersey color features (slower but useful for team ID).

    Attributes:
        samples: List of (sequence_name, frame_number) tuples.
        annotations: Dict mapping sequence names to annotation DataFrames.

    Example:
        >>> dataset = PlayerTrackingDataset("train.zip", bbox_format="xyxy")
        >>> sample = dataset[0]
        >>> print(sample["boxes"].shape)  # (N, 4)
        >>> print(sample["track_ids"].shape)  # (N,)
    """

    def __init__(
        self,
        zip_path: str,
        transform: Optional[Any] = None,
        bbox_format: Literal["xywh", "xyxy", "cxcywh"] = "xyxy",
        debug: bool = False,
        extract_colors: bool = False,
    ) -> None:
        self.zip_path = zip_path
        self.transform = transform
        self.bbox_format = bbox_format
        self.debug = debug
        self.extract_colors = extract_colors

        self.samples: List[tuple[str, int]] = []
        self.annotations: Dict[str, pd.DataFrame] = {}

        self._build_index()

    def _build_index(self) -> None:
        """
        Build sample index and load annotations from the zip file.

        This approach reduces repeated file I/O during training and avoids
        extracting zip contents to disk.
        """
        fs, _, _ = fsspec.get_fs_token_paths(self.zip_path)

        with fs.open(self.zip_path, "rb") as f:
            zip_fs = fsspec.filesystem("zip", fo=f)

            # Find root folder (train/, test/, val/)
            root_dirs = [p["name"] if isinstance(p, dict) else p for p in zip_fs.ls("")]
            root = root_dirs[0].rstrip("/")
            sequences_dir = f"{root}/"

            for seq in zip_fs.ls(sequences_dir):
                seq_name = seq["name"] if isinstance(seq, dict) else seq
                seq_name = seq_name.rstrip("/")

                gt_path = f"{seq_name}/gt/gt.txt"
                if not zip_fs.exists(gt_path):
                    continue

                with zip_fs.open(gt_path) as gt:
                    # MOT format: frame,track_id,x,y,w,h,conf,class_id,visibility,unused
                    df = pd.read_csv(
                        gt,
                        header=None,
                        names=[
                            "frame",
                            "track_id",
                            "x",
                            "y",
                            "w",
                            "h",
                            "conf",
                            "class_id",
                            "visibility",
                            "unused",
                        ],
                    )

                self.annotations[seq_name] = df

                for frame in df["frame"].unique():
                    self.samples.append((seq_name, frame))

    def __len__(self) -> int:
        """Return total number of samples (frames) in dataset."""
        return len(self.samples)

    def _load_image(self, seq: str, frame: int) -> torch.Tensor:
        """Load image from zip archive."""
        img_path = f"zip://{seq}/img1/{frame:06d}.jpg::{self.zip_path}"
        with fsspec.open(img_path, "rb") as f:
            img: Image.Image = Image.open(BytesIO(f.read())).convert("RGB")

        # Convert to tensor: [H, W, C] -> [C, H, W]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        return img_tensor

    def _convert_bbox_format(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes from MOT format (xywh) to desired format.

        Args:
            boxes: Array of shape (N, 4) in [x, y, w, h] format.

        Returns:
            Converted boxes in the specified format.
        """
        if len(boxes) == 0:
            return boxes

        boxes = boxes.astype(np.float32).copy()

        if self.bbox_format == "xywh":
            return boxes

        elif self.bbox_format == "xyxy":
            converted = np.zeros_like(boxes)
            converted[:, 0] = boxes[:, 0]  # x1
            converted[:, 1] = boxes[:, 1]  # y1
            converted[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
            converted[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
            return converted

        elif self.bbox_format == "cxcywh":
            converted = np.zeros_like(boxes)
            converted[:, 0] = boxes[:, 0] + boxes[:, 2] / 2  # cx = x + w/2
            converted[:, 1] = boxes[:, 1] + boxes[:, 3] / 2  # cy = y + h/2
            converted[:, 2] = boxes[:, 2]  # w
            converted[:, 3] = boxes[:, 3]  # h
            return converted

        else:
            raise ValueError(f"Unknown bbox format: {self.bbox_format}")

    def _extract_jersey_color(self, image: torch.Tensor, box: np.ndarray) -> torch.Tensor:
        """
        Extract jersey color features from a bounding box region.

        Extracts the upper 60% of the bounding box (torso area) and
        computes mean HSV color values.

        Args:
            image: Image tensor of shape (C, H, W).
            box: Bounding box in [x, y, w, h] format (original MOT format).

        Returns:
            3D tensor with mean [H, S, V] values.
        """
        x, y, w, h = box.astype(int)
        _, H, W = image.size()

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)

        if x2 <= x1 or y2 <= y1:
            return torch.zeros(3)

        crop = image[:, y1:y2, x1:x2]
        if crop.numel() == 0:
            return torch.zeros(3)

        # Convert to numpy and take upper 60% (torso)
        crop_np = crop.permute(1, 2, 0).cpu().numpy()
        h_crop = crop_np.shape[0]
        crop_np = crop_np[: int(0.6 * h_crop)]

        if crop_np.size == 0:
            return torch.zeros(3)

        # Normalize to 0-255 if needed
        if crop_np.max() <= 1.0:
            crop_np = (crop_np * 255).astype(np.uint8)
        else:
            crop_np = crop_np.astype(np.uint8)

        # Convert to HSV and compute mean
        hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
        mean_hsv = hsv.reshape(-1, 3).mean(axis=0)

        return torch.tensor(mean_hsv, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
                - "image": Tensor of shape (C, H, W)
                - "boxes": Tensor of shape (N, 4) in specified format
                - "track_ids": Tensor of shape (N,) with track IDs
                - "color_features": Tensor of shape (N, 3) if extract_colors=True
                - "seq_name": Sequence name string
                - "frame": Frame number
        """
        seq, frame = self.samples[idx]
        image = self._load_image(seq, frame)

        ann = self.annotations[seq]
        ann = ann[ann.frame == frame].copy()

        boxes_mot = ann[["x", "y", "w", "h"]].values.astype(np.float32)
        track_ids = ann["track_id"].values.astype(np.int64)

        if self.transform:
            image = self.transform(image)

        if self.extract_colors:
            color_features = [self._extract_jersey_color(image, box) for box in boxes_mot]
        else:
            color_features = []

        boxes = self._convert_bbox_format(boxes_mot)

        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "track_ids": torch.tensor(track_ids, dtype=torch.long),
            "color_features": torch.stack(color_features) if color_features else torch.zeros((0, 3)),
            "seq_name": seq,
            "frame": frame,
        }


def tracking_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching tracking data.

    Handles variable numbers of detections per frame by keeping boxes
    and track_ids as lists rather than stacking.

    Args:
        batch: List of sample dictionaries from PlayerTrackingDataset.

    Returns:
        Batched dictionary with:
            - "images": Stacked tensor of shape (B, C, H, W)
            - "boxes": List of B tensors, each (N_i, 4)
            - "track_ids": List of B tensors, each (N_i,)
            - "color_features": List of B tensors, each (N_i, 3)
            - "seq_names": List of B sequence name strings
            - "frames": List of B frame numbers
    """
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "track_ids": [b["track_ids"] for b in batch],
        "color_features": [b["color_features"] for b in batch],
        "seq_names": [b["seq_name"] for b in batch],
        "frames": [b["frame"] for b in batch],
    }


def visualize_sample(sample: Dict[str, Any], bbox_format: str = "xyxy") -> None:
    """
    Visualize a tracking sample with bounding boxes using matplotlib.

    Args:
        sample: Sample dictionary from PlayerTrackingDataset.
        bbox_format: Format of bounding boxes in the sample.

    Example:
        >>> sample = dataset[0]
        >>> visualize_sample(sample, bbox_format="xyxy")
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    image = sample["image"].permute(1, 2, 0).cpu().numpy()
    boxes = sample["boxes"].cpu().numpy()
    track_ids = sample["track_ids"].cpu().numpy()

    img_h, img_w = image.shape[:2]

    _, ax = plt.subplots(1, figsize=(16, 10))
    ax.imshow(image)

    for box, track_id in zip(boxes, track_ids):
        if bbox_format == "xywh":
            x, y, w, h = box
        elif bbox_format == "xyxy":
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
        elif bbox_format == "cxcywh":
            cx, cy, w, h = box
            x, y = cx - w / 2, cy - h / 2
        else:
            raise ValueError(f"Unknown bbox format: {bbox_format}")

        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x,
            y - 10,
            f"ID: {track_id}",
            color="yellow",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
        )

    ax.set_title(
        f"Frame {sample.get('frame', 'N/A')} - Seq: {sample.get('seq_name', 'N/A')}\n" f"Image: {img_w}x{img_h}"
    )
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    plt.tight_layout()
    plt.show()


__all__ = [
    "PlayerTrackingDataset",
    "tracking_collate_fn",
    "visualize_sample",
]
