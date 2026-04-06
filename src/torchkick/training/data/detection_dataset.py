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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_coco_source(
    json_path: str,
    images_dir: str,
    label_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Load COCO-format annotations into a flat list of sample dicts.

    Args:
        label_map: Optional remapping of category IDs. E.g. to collapse
            player/goalkeeper/referee → 0 (person) and ball → 1:
            ``{1: 0, 2: 0, 3: 0, 4: 1}``
            Annotations whose category maps to -1 are dropped.
    """
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
            cat = ann.get("category_id", 0)
            if label_map is not None:
                cat = label_map.get(cat, -1)
                if cat == -1:
                    continue
            x, y, w, h = ann["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(cat)
        samples.append(
            {
                "image_path": img_path,
                "boxes": boxes_xyxy,
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

    # SoccerNet uses class_id=-1 for all tracks (no per-class annotation in this split)
    # Standard MOT class IDs if present: 1=player, 2=goalkeeper, 3=referee → 0; 4=ball → 1
    _CLASS_TO_LABEL = {-1: 0, 1: 0, 2: 0, 3: 0, 4: 1}
    ds = PlayerTrackingDataset(zip_path, bbox_format="xyxy")
    samples = []
    for seq_name, frame_id in ds.samples:
        df = ds.annotations[seq_name]
        frame_rows = df[(df["frame"] == frame_id) & (df["class_id"].isin(_CLASS_TO_LABEL.keys()))]
        boxes = []
        labels = []
        for _, row in frame_rows.iterrows():
            x, y, w, h = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
            boxes.append([x, y, x + w, y + h])
            labels.append(_CLASS_TO_LABEL[int(row["class_id"])])
        if not boxes:
            continue
        samples.append(
            {
                "zip_path": zip_path,
                "seq_name": seq_name,
                "frame_id": int(frame_id),
                "boxes": boxes,
                "labels": labels,
            }
        )
    return samples


def _load_soccernet_dir_source(root_dir: str) -> List[Dict[str, Any]]:
    """Build index from a pre-extracted SoccerNet directory (much faster than zip).

    Expects the same layout as the zip: <root>/<seq_name>/img1/<frame>.jpg
    and <root>/<seq_name>/gt/gt.txt (MOT-format ground truth).
    """
    root = Path(root_dir)
    samples = []
    for gt_file in sorted(root.glob("**/gt/gt.txt")):
        seq_name = gt_file.parts[-3]
        img_dir = gt_file.parent.parent / "img1"
        # Parse MOT gt.txt: frame,id,x,y,w,h,conf,class,visibility
        # SoccerNet uses class_id=-1 for all tracks (no per-class annotation in this split)
        # Standard MOT class IDs if present: 1=player, 2=goalkeeper, 3=referee → 0; 4=ball → 1
        _CLASS_TO_LABEL = {"-1": 0, "1": 0, "2": 0, "3": 0, "4": 1}
        frames: Dict[int, List] = {}
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                fid = int(parts[0])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                # conf=0 means "ignore" region in MOT format — skip
                if len(parts) >= 7 and parts[6].strip() == "0":
                    continue
                class_str = parts[7].strip() if len(parts) >= 8 else "-1"
                if class_str not in _CLASS_TO_LABEL:
                    continue
                label = _CLASS_TO_LABEL[class_str]
                frames.setdefault(fid, []).append([x, y, x + w, y + h, label])
        for fid, entries in frames.items():
            valid = [(e[:4], e[4]) for e in entries if e[2] > e[0] and e[3] > e[1]]
            if not valid:
                continue
            img_path = str(img_dir / f"{fid:06d}.jpg")
            valid_boxes, valid_labels = zip(*valid)
            samples.append({"image_path": img_path, "boxes": list(valid_boxes), "labels": list(valid_labels)})
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
                # Moderate color augmentation: stronger than before but still jersey-safe
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05, p=0.8),
                A.CoarseDropout(num_holes_range=(2, 8), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.3),
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

    import warnings

    labels_list = labels.tolist() if len(labels) else []
    if len(boxes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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


def _apply_color_augmentations(
    image: np.ndarray, boxes: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Colour-only augmentations applied on top of mosaic (geometry already handled)."""
    try:
        import albumentations as A

        transform = A.Compose(
            [
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05, p=0.8),
                A.MotionBlur(blur_limit=5, p=0.2),
                A.CoarseDropout(num_holes_range=(2, 8), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.3),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3),
        )
    except ImportError:
        return image, boxes, labels

    import warnings

    labels_list = labels.tolist() if len(labels) else []
    if len(boxes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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

        try:
            import albumentations as A

            self._letterbox = A.Compose(
                [
                    A.LongestMaxSize(max_size=input_size),
                    A.PadIfNeeded(input_size, input_size, border_mode=cv2.BORDER_CONSTANT, fill=114),
                ],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1),
            )
        except ImportError:
            self._letterbox = None

        for src in sources:
            src_type = src["type"]
            if src_type == "coco_json":
                self._samples.extend(_load_coco_source(src["path"], src.get("images_dir", ""), src.get("label_map")))
            elif src_type == "soccernet_zip":
                self._samples.extend(_load_soccernet_source(src["path"]))
            elif src_type == "soccernet_dir":
                self._samples.extend(_load_soccernet_dir_source(src["path"]))
            else:
                raise ValueError(
                    f"Unknown source type: {src_type!r}. Use 'coco_json', 'soccernet_zip', or 'soccernet_dir'."
                )

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

    def _apply_mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """4-image mosaic augmentation (YOLOv5-style).

        Stitches 4 images on a 2×input_size canvas around a random centre point,
        then crops back to input_size×input_size. Effectively 4× dataset variety.
        """
        size = self.input_size
        canvas = np.full((size * 2, size * 2, 3), 114, dtype=np.uint8)
        # Random centre, biased toward the middle half to avoid degenerate crops
        cx = int(np.random.uniform(size * 0.5, size * 1.5))
        cy = int(np.random.uniform(size * 0.5, size * 1.5))

        indices = [idx] + [np.random.randint(0, len(self._samples)) for _ in range(3)]
        all_boxes: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for tile_idx, sample_idx in enumerate(indices):
            sample = self._samples[sample_idx]
            img = self._load_image(sample)
            boxes = (
                np.array(sample["boxes"], dtype=np.float32) if sample["boxes"] else np.zeros((0, 4), dtype=np.float32)
            )
            labels = np.array(sample["labels"], dtype=np.int64) if sample["labels"] else np.zeros(0, dtype=np.int64)

            # Scale image so longest side = input_size
            h, w = img.shape[:2]
            scale = min(size / h, size / w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            if len(boxes):
                boxes = boxes * scale

            # Anchor each tile to the mosaic centre (cx, cy)
            if tile_idx == 0:  # top-left: bottom-right corner → (cx, cy)
                x1c, y1c = max(cx - new_w, 0), max(cy - new_h, 0)
                x2c, y2c = cx, cy
                img_x1, img_y1 = max(new_w - cx, 0), max(new_h - cy, 0)
            elif tile_idx == 1:  # top-right: bottom-left corner → (cx, cy)
                x1c, y1c = cx, max(cy - new_h, 0)
                x2c, y2c = min(cx + new_w, size * 2), cy
                img_x1, img_y1 = 0, max(new_h - cy, 0)
            elif tile_idx == 2:  # bottom-left: top-right corner → (cx, cy)
                x1c, y1c = max(cx - new_w, 0), cy
                x2c, y2c = cx, min(cy + new_h, size * 2)
                img_x1, img_y1 = max(new_w - cx, 0), 0
            else:  # bottom-right: top-left corner → (cx, cy)
                x1c, y1c = cx, cy
                x2c, y2c = min(cx + new_w, size * 2), min(cy + new_h, size * 2)
                img_x1, img_y1 = 0, 0

            pw, ph = x2c - x1c, y2c - y1c
            canvas[y1c:y2c, x1c:x2c] = img[img_y1 : img_y1 + ph, img_x1 : img_x1 + pw]

            if len(boxes):
                offset_boxes = boxes.copy()
                offset_boxes[:, [0, 2]] += x1c - img_x1
                offset_boxes[:, [1, 3]] += y1c - img_y1
                offset_boxes[:, [0, 2]] = offset_boxes[:, [0, 2]].clip(x1c, x2c)
                offset_boxes[:, [1, 3]] = offset_boxes[:, [1, 3]].clip(y1c, y2c)
                valid = (offset_boxes[:, 2] - offset_boxes[:, 0] > 4) & (offset_boxes[:, 3] - offset_boxes[:, 1] > 4)
                all_boxes.append(offset_boxes[valid])
                all_labels.append(labels[valid])

        # Crop size×size centred at (cx, cy), clamped to canvas bounds
        sx = int(np.clip(cx - size // 2, 0, size))
        sy = int(np.clip(cy - size // 2, 0, size))
        cropped = canvas[sy : sy + size, sx : sx + size]

        merged_boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), dtype=np.float32)
        merged_labels = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.int64)

        if len(merged_boxes):
            merged_boxes[:, [0, 2]] -= sx
            merged_boxes[:, [1, 3]] -= sy
            merged_boxes[:, [0, 2]] = merged_boxes[:, [0, 2]].clip(0, size)
            merged_boxes[:, [1, 3]] = merged_boxes[:, [1, 3]].clip(0, size)
            valid = (merged_boxes[:, 2] - merged_boxes[:, 0] > 4) & (merged_boxes[:, 3] - merged_boxes[:, 1] > 4)
            merged_boxes = merged_boxes[valid]
            merged_labels = merged_labels[valid]

        return cropped, merged_boxes, merged_labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self._samples[idx]
        image = self._load_image(sample)
        boxes = np.array(sample["boxes"], dtype=np.float32) if sample["boxes"] else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(sample["labels"], dtype=np.int64) if sample["labels"] else np.zeros(0, dtype=np.int64)

        if self.augment:
            if np.random.random() < 0.5:
                # Mosaic path: composite 4 images, then apply colour aug on top
                image, boxes, labels = self._apply_mosaic(idx)
                image, boxes, labels = _apply_color_augmentations(image, boxes, labels)
            else:
                image, boxes, labels = _apply_augmentations(image, boxes, labels, self.input_size)
        else:
            if self._letterbox is not None:
                result = self._letterbox(image=image, bboxes=boxes.tolist(), labels=labels.tolist())
                image = result["image"]
                boxes = (
                    np.array(result["bboxes"], dtype=np.float32)
                    if result["bboxes"]
                    else np.zeros((0, 4), dtype=np.float32)
                )
                labels = np.array(result["labels"], dtype=np.int64) if result["labels"] else np.zeros(0, dtype=np.int64)
            else:
                orig_h, orig_w = image.shape[:2]
                image = cv2.resize(image, (self.input_size, self.input_size))
                if len(boxes):
                    scale_x = self.input_size / max(orig_w, 1)
                    scale_y = self.input_size / max(orig_h, 1)
                    boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)

        # BGR -> RGB, normalize with ImageNet stats (RT-DETR backbone pretrained on ImageNet)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        targets = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
        }
        return image_tensor, targets
