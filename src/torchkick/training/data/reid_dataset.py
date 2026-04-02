"""
ReID dataset for DINOv2+ArcFace training.

Provides player crop datasets from three sources:
    1. SoccerNet GT bounding boxes with team labels
    2. CVAT human-corrected exports with team/role labels
    3. ByteTrack pseudo-track crops (same track_id = same person)

Augmentation strategy: WEAK color jitter (jersey color IS the signal),
strong geometric augmentation (simulate broadcast camera angles).

Classes: 0 = home team, 1 = away team, 2 = referee

Example:
    >>> sources = [
    ...     {"type": "soccernet_zip", "path": "data/train.zip"},
    ...     {"type": "coco_json", "path": "data/cvat_export.json", "images_dir": "data/images/"},
    ... ]
    >>> dataset = ReIDDataset(sources, augment=True)
    >>> crop, label = dataset[0]
    >>> # crop: torch.Tensor [3, 256, 128], label: int (0/1/2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _weak_color_augment(image: np.ndarray) -> np.ndarray:
    """
    Apply weak color + strong geometric augmentation.

    Color jitter is intentionally weak — jersey color is the discriminative
    signal for team classification. Strong geometric augmentation simulates
    broadcast camera angles and player poses.
    """
    try:
        import albumentations as A

        transform = A.Compose(
            [
                # Weak color: don't destroy jersey colors
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.01, p=0.5),
                A.RandomGamma(gamma_limit=(85, 115), p=0.3),
                # Strong geometric
                A.HorizontalFlip(p=0.5),
                A.Perspective(scale=(0.03, 0.08), p=0.3),
                A.Affine(scale=(0.9, 1.1), shear=(-8, 8), p=0.4),
                A.MotionBlur(blur_limit=(3, 7), p=0.3),
                # Occlusion patches (simulate partial occlusion)
                A.CoarseDropout(max_holes=3, max_height=24, max_width=24, min_holes=1, p=0.3),
            ]
        )
        return transform(image=image)["image"]
    except ImportError:
        return image


class ReIDDataset(Dataset):
    """
    Player crop dataset for ReID (DINOv2+ArcFace) training.

    Can be constructed two ways:

    **Simple (used by train_reid.py)**::

        # supervised — class sub-folders named 0/1/2 or home/away/referee
        ds = ReIDDataset("data/reid/", source_type="label_dir", augment=True)

        # BYOL SSL — track-id sub-folders; same folder = same identity
        ds = ReIDDataset("data/reid/", source_type="tracklet_dir",
                         augment=True, byol_mode=True)

    **Legacy (list of source descriptors)**::

        sources = [
            {"type": "soccernet_zip", "path": "data/train.zip"},
            {"type": "coco_json", "path": "data/export.json", "images_dir": "data/images/"},
            {"type": "tracklet_dir", "path": "data/tracks/", "label": 0},
        ]
        ds = ReIDDataset(sources, augment=True)

    Args:
        data_dir_or_sources: Either a directory path string (used with
            ``source_type``) or a list of source descriptor dicts (legacy).
        source_type: ``"label_dir"`` or ``"tracklet_dir"``. Only used when
            ``data_dir_or_sources`` is a string.
        input_size: (width, height) for player crops. Default (128, 256).
        augment: Apply augmentation pipeline.
        min_crop_size: Skip crops smaller than this in pixels (noisy GT).
        byol_mode: When True, ``__getitem__`` returns
            ``{"view1": tensor, "view2": tensor}`` (two independent
            augmentations of the same crop) for BYOL SSL training.
            When False, returns ``{"crops": tensor, "labels": int}``.
    """

    LABEL_NAMES = {0: "home", 1: "away", 2: "referee"}
    _LABEL_DIR_ALIASES = {"home": 0, "away": 1, "referee": 2, "ref": 2}

    def __init__(
        self,
        data_dir_or_sources,
        source_type: Optional[str] = None,
        input_size: Tuple[int, int] = (128, 256),
        augment: bool = True,
        min_crop_size: int = 16,
        byol_mode: bool = False,
    ) -> None:
        self.input_size = input_size  # (W, H)
        self.augment = augment
        self.min_crop_size = min_crop_size
        self.byol_mode = byol_mode
        self._items: List[Dict[str, Any]] = []

        if isinstance(data_dir_or_sources, (str, Path)):
            # Simple directory-based construction
            data_dir = str(data_dir_or_sources)
            if source_type == "label_dir":
                self._load_label_dir(data_dir)
            elif source_type == "tracklet_dir":
                self._load_tracklet_dir_nested(data_dir)
            else:
                raise ValueError(
                    f"source_type must be 'label_dir' or 'tracklet_dir' when "
                    f"data_dir_or_sources is a path, got {source_type!r}"
                )
        else:
            # Legacy list-of-dicts construction
            sources = data_dir_or_sources
            for src in sources:
                src_type = src["type"]
                if src_type == "soccernet_zip":
                    self._load_soccernet(src["path"])
                elif src_type == "coco_json":
                    self._load_coco(src["path"], src.get("images_dir", ""), src.get("team_attribute", "team"))
                elif src_type == "tracklet_dir":
                    self._load_tracklet_dir(src["path"], src.get("label", 0))
                else:
                    raise ValueError(f"Unknown source type: {src_type!r}")

    def _load_label_dir(self, data_dir: str) -> None:
        """Load crops from ``data_dir/<label>/crop.jpg`` structure."""
        root = Path(data_dir)
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            # Parse label from folder name
            name = sub.name.lower()
            if name.isdigit():
                label = int(name)
            elif name in self._LABEL_DIR_ALIASES:
                label = self._LABEL_DIR_ALIASES[name]
            else:
                continue  # skip unknown sub-folders
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in sub.glob(ext):
                    self._items.append({"image_path": str(p), "box": None, "label": label})

    def _load_tracklet_dir_nested(self, data_dir: str) -> None:
        """
        Load crops from ``data_dir/<track_id>/crop.jpg`` structure.

        All crops in the same sub-folder share the same track_id, used as
        the identity label for BYOL positive-pair mining.
        """
        root = Path(data_dir)
        label = 0
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            crops = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                crops.extend(sub.glob(ext))
            for p in crops:
                self._items.append({"image_path": str(p), "box": None, "label": label})
            label += 1

    def _load_soccernet(self, zip_path: str) -> None:
        try:
            from torchkick.soccernet import PlayerTrackingDataset
        except ImportError:
            raise ImportError("SoccerNet source requires torchkick[soccernet]")

        ds = PlayerTrackingDataset(zip_path)
        for i in range(len(ds)):
            item = ds[i]
            img = item["image"]
            for j, box in enumerate(item["boxes"]):
                x1, y1, x2, y2 = [int(v) for v in box]
                team_label = item.get("team_labels", [0] * len(item["boxes"]))[j]
                self._items.append(
                    {
                        "image_array": img,
                        "box": (x1, y1, x2, y2),
                        "label": int(team_label),
                    }
                )

    def _load_coco(self, json_path: str, images_dir: str, team_attr: str) -> None:
        with open(json_path) as f:
            coco = json.load(f)

        id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in id_to_file:
                continue
            img_path = str(Path(images_dir) / id_to_file[img_id])
            x, y, w, h = ann["bbox"]
            team_label = ann.get("attributes", {}).get(team_attr, ann.get("category_id", 0))
            self._items.append(
                {
                    "image_path": img_path,
                    "box": (int(x), int(y), int(x + w), int(y + h)),
                    "label": int(team_label),
                }
            )

    def _load_tracklet_dir(self, dir_path: str, label: int) -> None:
        """Load pre-extracted player crops from directory. All get the same label."""
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for p in Path(dir_path).glob(ext):
                self._items.append(
                    {
                        "image_path": str(p),
                        "box": None,  # entire image is the crop
                        "label": label,
                    }
                )

    def __len__(self) -> int:
        return len(self._items)

    def _load_crop(self, item: Dict[str, Any]) -> np.ndarray:
        """Load and crop image to (W, H), returned as BGR uint8."""
        if "image_array" in item:
            img = item["image_array"]
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = cv2.imread(item["image_path"])
            if img is None:
                img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)

        box = item["box"]
        if box is not None:
            x1, y1, x2, y2 = box
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < self.min_crop_size or y2 - y1 < self.min_crop_size:
                img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
            else:
                img = img[y1:y2, x1:x2]

        return cv2.resize(img, self.input_size)

    def _to_tensor(self, crop_bgr: np.ndarray) -> torch.Tensor:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._items[idx]
        crop = self._load_crop(item)

        if self.byol_mode:
            # Return two independently augmented views for BYOL SSL
            view1 = _weak_color_augment(crop) if self.augment else crop.copy()
            view2 = _weak_color_augment(crop) if self.augment else crop.copy()
            return {
                "view1": self._to_tensor(view1),
                "view2": self._to_tensor(view2),
            }

        if self.augment:
            crop = _weak_color_augment(crop)

        return {
            "crops": self._to_tensor(crop),
            "labels": item["label"],
        }
