"""
FIFA Skeletal Tracking Light 2026 dataset loader.

Supports two modes:
  - "pose_2d": (crop_image [3,256,192], kp2d [17,2] in crop pixels, vis [17])
               for ViTPose fine-tuning.
  - "lift_3d": (kp2d_norm [17,3], kp3d_root [17,3]) for 3D lifter training.
               kp2d_norm channels are (x_rootrel, y_rootrel, visibility).
               kp3d_root is root-relative in metres.

Body25→COCO-17 keypoint mapping is applied automatically.
NaN detections are dropped.

Data must be downloaded before use:
    huggingface-cli download tijiang13/FIFA-Skeletal-Tracking-Light-2026 \\
        --repo-type dataset --local-dir data/fifa/

Example:
    >>> ds = FIFAPoseDataset("data/fifa/", split="train", mode="pose_2d")
    >>> img, kp2d, vis = ds[0]

    >>> ds3 = FIFAPoseDataset("data/fifa/", split="train", mode="lift_3d")
    >>> kp2d_norm, kp3d = ds3[0]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Body25 index → COCO-17 index mapping.
# Usage: coco17_kps[i] = body25_kps[BODY25_TO_COCO17[i]]
BODY25_TO_COCO17: List[int] = [
    0,  # COCO  0  nose          ← Body25  0  Nose
    16,  # COCO  1  left_eye      ← Body25 16  Left eye
    15,  # COCO  2  right_eye     ← Body25 15  Right eye
    18,  # COCO  3  left_ear      ← Body25 18  Left ear
    17,  # COCO  4  right_ear     ← Body25 17  Right ear
    5,  # COCO  5  left_shoulder ← Body25  5  Left shoulder
    2,  # COCO  6  right_shoulder← Body25  2  Right shoulder
    6,  # COCO  7  left_elbow    ← Body25  6  Left elbow
    3,  # COCO  8  right_elbow   ← Body25  3  Right elbow
    7,  # COCO  9  left_wrist    ← Body25  7  Left wrist
    4,  # COCO 10  right_wrist   ← Body25  4  Right wrist
    12,  # COCO 11  left_hip      ← Body25 12  Left hip
    9,  # COCO 12  right_hip     ← Body25  9  Right hip
    13,  # COCO 13  left_knee     ← Body25 13  Left knee
    10,  # COCO 14  right_knee    ← Body25 10  Right knee
    14,  # COCO 15  left_ankle    ← Body25 14  Left ankle
    11,  # COCO 16  right_ankle   ← Body25 11  Right ankle
]

# Root joint index in COCO-17 (used to centre the 2D/3D poses)
_LEFT_HIP_IDX = 11
_RIGHT_HIP_IDX = 12

# Shoulder indices used for scale normalisation
_LEFT_SHOULDER_IDX = 5
_RIGHT_SHOULDER_IDX = 6

# Minimum non-NaN keypoints required to include a sample
_MIN_VISIBLE = 6

# Default crop size for ViTPose (width, height)
_INPUT_W = 192
_INPUT_H = 256


def _body25_to_coco17(kps: np.ndarray) -> np.ndarray:
    """
    Convert Body25 keypoints to COCO-17 subset.

    Args:
        kps: [..., 25, D] array (D = 2 or 3).

    Returns:
        [..., 17, D] array.
    """
    return kps[..., BODY25_TO_COCO17, :]


def _pad_box_to_aspect(box: np.ndarray, img_h: int, img_w: int, ratio: float = 3 / 4) -> np.ndarray:
    """
    Expand box to target width:height aspect ratio (symmetric padding).

    Args:
        box: [x1, y1, x2, y2].
        img_h, img_w: Image dimensions for clamping.
        ratio: target w/h ratio (ViTPose default 3:4).

    Returns:
        Padded [x1, y1, x2, y2] clamped to image bounds.
    """
    x1, y1, x2, y2 = box.astype(float)
    bw, bh = x2 - x1, y2 - y1
    if bh == 0:
        bh = 1.0

    target_w = bh * ratio
    target_h = bw / ratio

    if bw / bh < ratio:
        # Too narrow — pad width
        pad = (target_w - bw) / 2
        x1 -= pad
        x2 += pad
    else:
        # Too short — pad height
        pad = (target_h - bh) / 2
        y1 -= pad
        y2 += pad

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _skeleton_box(kp2d: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute tight bbox from visible (non-NaN) keypoints with 20% padding.

    Args:
        kp2d: [17, 2] float (NaN for invisible).

    Returns:
        [x1, y1, x2, y2] or None if no visible keypoints.
    """
    vis = ~np.any(np.isnan(kp2d), axis=-1)
    if vis.sum() == 0:
        return None
    pts = kp2d[vis]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pw, ph = (x2 - x1) * 0.2 + 5, (y2 - y1) * 0.2 + 5
    return np.array([x1 - pw, y1 - ph, x2 + pw, y2 + ph], dtype=np.float32)


class FIFAPoseDataset(Dataset):
    """
    FIFA Skeletal Tracking Light 2026 dataset.

    Builds a flat index of (sequence, frame, person) samples at init time.
    Lazy-loads images and numpy arrays on demand.

    Args:
        root: Path to downloaded dataset root (contains skel_2d/, cameras/, …).
        split: "train", "val", or "test" — selects the corresponding sequences_*.txt.
        mode: "pose_2d" for ViTPose fine-tuning; "lift_3d" for lifter training.
        augment: Apply colour/flip augmentation (pose_2d mode only).
        load_images: Whether to load image crops (required for pose_2d, optional for lift_3d).
        max_samples: Cap the dataset at this many samples (for quick experiments).
    """

    FLIP_PAIRS: List[Tuple[int, int]] = [
        (1, 2),  # eyes
        (3, 4),  # ears
        (5, 6),  # shoulders
        (7, 8),  # elbows
        (9, 10),  # wrists
        (11, 12),  # hips
        (13, 14),  # knees
        (15, 16),  # ankles
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "pose_2d",
        augment: bool = True,
        load_images: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        assert mode in ("pose_2d", "lift_3d"), f"Unknown mode: {mode}"
        self.root = Path(root)
        self.mode = mode
        self.augment = augment and mode == "pose_2d"
        self.load_images = load_images or mode == "pose_2d"

        seq_file = self.root / f"sequences_{split}.txt"
        if not seq_file.exists():
            # Fall back to sequences_full.txt for splits that might not exist
            seq_file = self.root / "sequences_full.txt"
        with open(seq_file) as f:
            sequences = [l.strip() for l in f if l.strip()]

        self._index: List[Tuple[str, int, int]] = []  # (seq, frame_idx, person_idx)
        self._skel2d_cache: dict = {}
        self._skel3d_cache: dict = {}
        self._boxes_cache: dict = {}

        for seq in sequences:
            skel2d_path = self.root / "skel_2d" / f"{seq}.npy"
            skel3d_path = self.root / "skel_3d" / f"{seq}.npy"
            if not skel2d_path.exists():
                continue

            skel2d = np.load(skel2d_path, allow_pickle=True)  # (F, P, 25, 2)
            skel3d = np.load(skel3d_path, allow_pickle=True) if skel3d_path.exists() else None

            # skel2d might be (F, P, 25, 2) or (F, P, 25, 3) with confidence
            if skel2d.ndim == 4 and skel2d.shape[2] == 25:
                self._skel2d_cache[seq] = skel2d
                self._skel3d_cache[seq] = skel3d

                n_frames, n_persons = skel2d.shape[:2]
                for f in range(n_frames):
                    for p in range(n_persons):
                        kp = skel2d[f, p, :, :2]  # [25, 2]
                        vis = ~np.any(np.isnan(kp), axis=-1)
                        if vis.sum() < _MIN_VISIBLE:
                            continue
                        if mode == "lift_3d" and skel3d is not None:
                            kp3 = skel3d[f, p]  # [25, 3]
                            if np.any(np.isnan(kp3)):
                                continue
                        self._index.append((seq, f, p))

                # Try to load boxes
                box_path = self.root / "boxes" / f"{seq}.npy"
                if box_path.exists():
                    self._boxes_cache[seq] = np.load(box_path, allow_pickle=True)

            if max_samples and len(self._index) >= max_samples:
                break

        if max_samples:
            self._index = self._index[:max_samples]

        print(f"FIFAPoseDataset [{split}/{mode}]: {len(self._index)} samples from {len(sequences)} sequences")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq, frame_idx, person_idx = self._index[idx]

        skel2d = self._skel2d_cache[seq]  # (F, P, 25, 2)
        kp25_2d = skel2d[frame_idx, person_idx, :, :2].astype(np.float32)  # [25, 2]
        kp17_2d = _body25_to_coco17(kp25_2d)  # [17, 2]
        vis17 = (~np.any(np.isnan(kp17_2d), axis=-1)).astype(np.float32)  # [17]
        kp17_2d = np.nan_to_num(kp17_2d, nan=0.0)

        if self.mode == "pose_2d":
            return self._get_pose2d(seq, frame_idx, kp17_2d, vis17)
        else:
            skel3d = self._skel3d_cache[seq]  # (F, P, 25, 3)
            kp25_3d = skel3d[frame_idx, person_idx].astype(np.float32)  # [25, 3]
            kp17_3d = _body25_to_coco17(kp25_3d)  # [17, 3]
            return self._get_lift3d(kp17_2d, vis17, kp17_3d)

    def _get_pose2d(
        self,
        seq: str,
        frame_idx: int,
        kp17_2d: np.ndarray,  # [17, 2] full-frame pixel coords
        vis17: np.ndarray,  # [17]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load crop and remap keypoints to crop space for ViTPose fine-tuning."""
        img_bgr = self._load_image(seq, frame_idx)
        img_h, img_w = img_bgr.shape[:2]

        # Get bounding box
        box = self._get_box(seq, frame_idx, person_idx=None, kp17_2d=kp17_2d, img_h=img_h, img_w=img_w)
        if box is None:
            # Fallback: use full frame
            box = np.array([0, 0, img_w, img_h], dtype=np.float32)

        x1, y1, x2, y2 = map(int, box)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img_bgr

        crop_h, crop_w = crop.shape[:2]

        # Remap keypoints to crop coords
        kp_crop = kp17_2d.copy()
        kp_crop[:, 0] -= x1
        kp_crop[:, 1] -= y1

        if self.augment:
            crop, kp_crop, vis17 = self._augment(crop, kp_crop, vis17)
            crop_h, crop_w = crop.shape[:2]

        # Clamp keypoints out of crop bounds
        out_of_bounds = (
            (kp_crop[:, 0] < 0) | (kp_crop[:, 0] >= crop_w) | (kp_crop[:, 1] < 0) | (kp_crop[:, 1] >= crop_h)
        )
        vis17 = vis17 * (~out_of_bounds).astype(np.float32)

        # Resize to ViTPose input
        crop_resized = cv2.resize(crop, (_INPUT_W, _INPUT_H))
        scale_x = _INPUT_W / max(crop_w, 1)
        scale_y = _INPUT_H / max(crop_h, 1)
        kp_resized = kp_crop.copy()
        kp_resized[:, 0] *= scale_x
        kp_resized[:, 1] *= scale_y
        kp_resized = np.clip(kp_resized, 0, [_INPUT_W - 1, _INPUT_H - 1])

        img_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        kp_tensor = torch.from_numpy(kp_resized.astype(np.float32))
        vis_tensor = torch.from_numpy(vis17.astype(np.float32))

        return img_tensor, kp_tensor, vis_tensor

    def _get_lift3d(
        self,
        kp17_2d: np.ndarray,  # [17, 2]
        vis17: np.ndarray,  # [17]
        kp17_3d: np.ndarray,  # [17, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalise 2D and 3D poses for lifter training.

        2D: root-relative (hip midpoint subtracted), scale-normalised
            by torso height, returned as [17, 3] with visibility as 3rd channel.
        3D: root-relative in metres, returned as [17, 3].
        """
        root_2d = (kp17_2d[_LEFT_HIP_IDX] + kp17_2d[_RIGHT_HIP_IDX]) / 2.0
        # Torso scale: shoulder midpoint to hip midpoint
        shoulder_mid = (kp17_2d[_LEFT_SHOULDER_IDX] + kp17_2d[_RIGHT_SHOULDER_IDX]) / 2.0
        scale = np.linalg.norm(shoulder_mid - root_2d)
        scale = max(scale, 1.0)  # avoid division by zero

        kp2d_norm = (kp17_2d - root_2d) / scale  # [17, 2]
        # Zero out invisible keypoints
        kp2d_norm *= vis17[:, np.newaxis]

        # Stack visibility as 3rd channel
        kp2d_input = np.concatenate([kp2d_norm, vis17[:, np.newaxis]], axis=-1).astype(np.float32)  # [17, 3]

        root_3d = (kp17_3d[_LEFT_HIP_IDX] + kp17_3d[_RIGHT_HIP_IDX]) / 2.0
        kp3d_root = (kp17_3d - root_3d).astype(np.float32)  # [17, 3] root-relative metres
        kp3d_root *= vis17[:, np.newaxis]

        return torch.from_numpy(kp2d_input), torch.from_numpy(kp3d_root)

    def _load_image(self, seq: str, frame_idx: int) -> np.ndarray:
        img_path = self.root / "images" / seq / f"{frame_idx:05d}.jpg"
        if not img_path.exists():
            # Try zero-padded variants
            img_path = self.root / "images" / seq / f"{frame_idx:06d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            # Return black frame if image missing
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        return img

    def _get_box(
        self,
        seq: str,
        frame_idx: int,
        person_idx,
        kp17_2d: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> Optional[np.ndarray]:
        boxes = self._boxes_cache.get(seq)
        if boxes is not None and boxes.ndim >= 3:
            try:
                box_raw = boxes[frame_idx, person_idx]
                if not np.any(np.isnan(box_raw)) and len(box_raw) == 4:
                    return _pad_box_to_aspect(box_raw, img_h, img_w)
            except (IndexError, TypeError):
                pass

        # Fall back to skeleton bounding box
        box_skel = _skeleton_box(kp17_2d)
        if box_skel is not None:
            return _pad_box_to_aspect(box_skel, img_h, img_w)
        return None

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
            for i, (x, y) in enumerate(result["keypoints"]):
                kps[i, 0] = x
                kps[i, 1] = y
        except ImportError:
            pass

        if np.random.random() < 0.5:
            h, w = image.shape[:2]
            image = cv2.flip(image, 1)
            kps[:, 0] = w - kps[:, 0]
            for left_idx, right_idx in self.FLIP_PAIRS:
                kps[[left_idx, right_idx]] = kps[[right_idx, left_idx]]
                vis[[left_idx, right_idx]] = vis[[right_idx, left_idx]]

        return image, kps, vis


__all__ = [
    "FIFAPoseDataset",
    "BODY25_TO_COCO17",
    "_body25_to_coco17",
]
