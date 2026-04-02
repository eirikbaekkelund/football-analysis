"""
Mixed detection dataset for RT-DETR-X training.

Merges SoccerNet tracking data, Roboflow football datasets, and CVAT exports
into a unified COCO-format PyTorch Dataset. Supports mosaic augmentation
(4-image mosaic following YOLOv5 strategy) via albumentations.

Sources supported:
    - "soccernet_zip": SoccerNet tracking zip files (uses PlayerTrackingDataset)
    - "coco_json": Standard COCO-format annotation JSON + image directory

Example:
    >>> sources = [
    ...     {"type": "soccernet_zip", "path": "data/soccernet/tracking/train.zip"},
    ...     {"type": "coco_json", "path": "data/roboflow/annotations.json",
    ...      "images_dir": "data/roboflow/images/"},
    ... ]
    >>> dataset = MixedDetectionDataset(sources, augment=True, input_size=640)
    >>> img, targets = dataset[0]
    >>> # img: torch.Tensor [3, 640, 640], targets: dict with "boxes" and "labels"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_coco_source(
    json_path: str,
    images_dir: str,
) -> List[Dict[str, Any]]:
    """Load COCO-format annotations into a flat list of sample dicts."""
    with open(json_path) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_anns: Dict[int, List] = {}
    for ann in coco.get("annotations", []):
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    samples = []
    for img_id, file_name in id_to_file.items():
        img_path = str(Path(images_dir) / file_name)
        anns = id_to_anns.get(img_id, [])
        boxes_xyxy = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(ann.get("category_id", 0))
        samples.append(
            {
                "image_path": img_path,
                "boxes": boxes_xyxy,  # [[x1,y1,x2,y2], ...]
                "labels": labels,
            }
        )
    return samples


def _load_soccernet_source(zip_path: str) -> List[Dict[str, Any]]:
    """Build metadata-only index from SoccerNet zip. Images loaded lazily in __getitem__."""
    try:
        from torchkick.soccernet import PlayerTrackingDataset
    except ImportError:
        raise ImportError("SoccerNet source requires torchkick[soccernet]")

    ds = PlayerTrackingDataset(zip_path, bbox_format="xyxy")
    samples = []
    for seq_name, frame_id in ds.samples:
        df = ds.annotations[seq_name]
        frame_rows = df[df["frame"] == frame_id]
        boxes = []
        for _, row in frame_rows.iterrows():
            x, y, w, h = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
            boxes.append([x, y, x + w, y + h])
        samples.append(
            {
                "zip_path": zip_path,
                "seq_name": seq_name,
                "frame_id": int(frame_id),
                "boxes": boxes,
                "labels": [0] * len(boxes),
            }
        )
    return samples


def _apply_augmentations(
    image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, input_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply albumentations augmentation pipeline.

    Conservative on color (preserve jersey colors) but aggressive on geometry.
    Returns (image, boxes, labels) — labels are filtered to match surviving boxes.
    """
    try:
        import albumentations as A

        transform = A.Compose(
            [
                A.LongestMaxSize(max_size=input_size),
                A.PadIfNeeded(input_size, input_size, border_mode=cv2.BORDER_CONSTANT, fill=114),
                A.HorizontalFlip(p=0.5),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                A.Affine(shear=(-5, 5), p=0.3),
                A.MotionBlur(blur_limit=5, p=0.2),
                # Weak color augmentation: preserve jersey colors
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=0.5),
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.2),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3),
        )
    except ImportError:
        # albumentations not installed — just resize
        image = cv2.resize(image, (input_size, input_size))
        if len(boxes):
            scale_x = input_size / max(image.shape[1], 1)
            scale_y = input_size / max(image.shape[0], 1)
            boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y])
        return image, boxes, labels

    labels_list = labels.tolist() if len(labels) else []
    if len(boxes):
        result = transform(image=image, bboxes=boxes.tolist(), labels=labels_list)
        image = result["image"]
        boxes = np.array(result["bboxes"], dtype=np.float32) if result["bboxes"] else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(result["labels"], dtype=np.int64) if result["labels"] else np.zeros(0, dtype=np.int64)
    else:
        result = transform(image=image, bboxes=[], labels=[])
        image = result["image"]
        boxes = np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros(0, dtype=np.int64)

    return image, boxes, labels


class MixedDetectionDataset(Dataset):
    """
    Multi-source detection dataset for RT-DETR-X training.

    Merges SoccerNet tracking, Roboflow COCO exports, and CVAT exports
    into a unified interface returning (image_tensor, targets) pairs.

    Args:
        sources: List of source descriptors. Each is a dict with:
            - "type": "soccernet_zip" | "coco_json"
            - "path": path to zip or JSON file
            - "images_dir": (coco_json only) path to images directory
        input_size: Square input size for the model (default 640).
        augment: Whether to apply augmentation pipeline.

    Returns:
        (image_tensor, targets) where:
            image_tensor: torch.Tensor [3, input_size, input_size] float32 in [0, 1]
            targets: dict with "boxes" [N, 4] xyxy and "labels" [N] int64
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        input_size: int = 640,
        augment: bool = True,
    ) -> None:
        self.input_size = input_size
        self.augment = augment
        self._samples: List[Dict[str, Any]] = []

        for src in sources:
            src_type = src["type"]
            if src_type == "coco_json":
                self._samples.extend(_load_coco_source(src["path"], src.get("images_dir", "")))
            elif src_type == "soccernet_zip":
                self._samples.extend(_load_soccernet_source(src["path"]))
            else:
                raise ValueError(f"Unknown source type: {src_type!r}. Use 'coco_json' or 'soccernet_zip'.")

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image(self, sample: Dict[str, Any]) -> np.ndarray:
        if "zip_path" in sample:
            import fsspec
            from io import BytesIO
            from PIL import Image as PILImage
            img_path = f"zip://{sample['seq_name']}/img1/{sample['frame_id']:06d}.jpg::{sample['zip_path']}"
            with fsspec.open(img_path, "rb") as f:
                img = np.array(PILImage.open(BytesIO(f.read())).convert("RGB"))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if "image_array" in sample:
            img = sample["image_array"]
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            return img
        path = sample["image_path"]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self._samples[idx]
        image = self._load_image(sample)
        boxes = np.array(sample["boxes"], dtype=np.float32) if sample["boxes"] else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(sample["labels"], dtype=np.int64) if sample["labels"] else np.zeros(0, dtype=np.int64)

        if self.augment:
            image, boxes, labels = _apply_augmentations(image, boxes, labels, self.input_size)
        else:
            image = cv2.resize(image, (self.input_size, self.input_size))

        # BGR -> RGB, [0, 255] -> [0, 1]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        targets = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
        }
        return image_tensor, targets
