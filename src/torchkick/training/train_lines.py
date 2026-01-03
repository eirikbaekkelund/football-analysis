"""
Pitch line detection training.

This module provides training utilities for HRNet-based pitch line
detection models. These models detect line keypoints and extremities
for camera calibration and homography estimation.

Example:
    >>> from torchkick.training import train_lines, LineDetectionDataset
    >>> 
    >>> # Train line detector
    >>> train_lines(
    ...     data_dir="soccernet/calibration",
    ...     output_dir="weights/pitch",
    ...     epochs=100,
    ... )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import albumentations as A

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# Constants
NUM_LINE_CLASSES = 23
HEATMAP_SIZE = (135, 240)
INPUT_SIZE = (540, 960)

LINE_NAMES = [
    "left_penalty_top",
    "left_penalty_left",
    "left_penalty_bottom",
    "right_penalty_top",
    "right_penalty_right",
    "right_penalty_bottom",
    "left_goal_top",
    "left_goal_left_top",
    "left_goal_left_bottom",
    "right_goal_top",
    "right_goal_right_bottom",
    "right_goal_right_top",
    "center_line",
    "top_touchline",
    "left_sideline",
    "right_sideline",
    "bottom_touchline",
    "left_small_box_top",
    "left_small_box_left",
    "left_small_box_bottom",
    "right_small_box_top",
    "right_small_box_right",
    "right_small_box_bottom",
]


class LineDetectionDataset(Dataset):
    """
    Dataset for pitch line detection training.

    Loads images and line annotations, generating heatmaps for each
    line class for training.

    Args:
        data_dir: Directory containing images and labels.
        split: "train" or "val".
        augment: Apply data augmentation.

    The expected directory structure is:
        data_dir/
            images/train/
            images/val/
            labels/train/
            labels/val/
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        augment: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and HAS_ALBUMENTATIONS

        self.samples = self._load_samples()
        self.transform = self._build_transforms()

    def _load_samples(self) -> List[Dict]:
        """Load sample paths and annotations."""
        samples = []
        annotations_file = self.data_dir / f"{self.split}_annotations.json"

        if annotations_file.exists():
            with open(annotations_file, "r") as f:
                annotations = json.load(f)

            for item in annotations:
                image_path = self.data_dir / "images" / item["image"]
                if image_path.exists():
                    samples.append(
                        {
                            "image_path": str(image_path),
                            "lines": item["lines"],
                            "extremities": item.get("extremities", {}),
                        }
                    )
        else:
            # Alternative structure
            images_dir = self.data_dir / "images" / self.split
            labels_dir = self.data_dir / "labels" / self.split

            if images_dir.exists():
                for img_file in images_dir.glob("*.jpg"):
                    label_file = labels_dir / f"{img_file.stem}.json"
                    if label_file.exists():
                        with open(label_file, "r") as f:
                            label_data = json.load(f)
                        samples.append(
                            {
                                "image_path": str(img_file),
                                "lines": label_data.get("lines", {}),
                                "extremities": label_data.get("extremities", {}),
                            }
                        )

        return samples

    def _build_transforms(self):
        """Build augmentation transforms."""
        if not self.augment or not HAS_ALBUMENTATIONS:
            return None

        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.MotionBlur(blur_limit=5, p=0.1),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-5, 5), shear=(-3, 3), p=0.3),
            ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        )

    def _generate_line_heatmap(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        heatmap_h: int,
        heatmap_w: int,
        sigma: float = 3.0,
    ) -> np.ndarray:
        """Generate heatmap for a line segment."""
        heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

        x1, y1 = point1
        x2, y2 = point2

        length = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        if length == 0:
            return heatmap

        for i in range(length + 1):
            t = i / max(length, 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            if 0 <= x < heatmap_w and 0 <= y < heatmap_h:
                for dy in range(-int(3 * sigma), int(3 * sigma) + 1):
                    for dx in range(-int(3 * sigma), int(3 * sigma) + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < heatmap_w and 0 <= ny < heatmap_h:
                            dist_sq = dx * dx + dy * dy
                            val = np.exp(-dist_sq / (2 * sigma * sigma))
                            heatmap[ny, nx] = max(heatmap[ny, nx], val)

        return heatmap

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # Collect keypoints for augmentation
        keypoints = []
        keypoint_labels = []

        for line_name, line_data in sample["lines"].items():
            if line_name in LINE_NAMES:
                line_idx = LINE_NAMES.index(line_name)
                if "points" in line_data:
                    for pt in line_data["points"]:
                        keypoints.append((pt[0], pt[1]))
                        keypoint_labels.append(line_idx)

        # Apply augmentation
        if self.transform and keypoints:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed["image"]
            keypoints = transformed["keypoints"]

        # Resize image
        image = cv2.resize(image, (INPUT_SIZE[1], INPUT_SIZE[0]))

        # Generate heatmaps
        scale_x = HEATMAP_SIZE[1] / orig_w
        scale_y = HEATMAP_SIZE[0] / orig_h

        heatmaps = np.zeros((NUM_LINE_CLASSES + 1, HEATMAP_SIZE[0], HEATMAP_SIZE[1]), dtype=np.float32)

        for line_name, line_data in sample["lines"].items():
            if line_name not in LINE_NAMES:
                continue

            line_idx = LINE_NAMES.index(line_name)

            if "extremities" in line_data:
                ext = line_data["extremities"]
                if len(ext) >= 2:
                    p1 = (int(ext[0][0] * scale_x), int(ext[0][1] * scale_y))
                    p2 = (int(ext[1][0] * scale_x), int(ext[1][1] * scale_y))
                    heatmaps[line_idx] = self._generate_line_heatmap(p1, p2, HEATMAP_SIZE[0], HEATMAP_SIZE[1])

        # Background class
        heatmaps[-1] = 1.0 - np.max(heatmaps[:-1], axis=0)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = image.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(image).float(),
            "heatmaps": torch.from_numpy(heatmaps).float(),
        }


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class LineHeatmapLoss(nn.Module):
    """Loss for line heatmap prediction."""

    def __init__(self, use_focal: bool = True) -> None:
        super().__init__()
        self.use_focal = use_focal
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_focal:
            return self.focal_loss(pred, target)
        return self.mse_loss(torch.sigmoid(pred), target)


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        targets = batch["heatmaps"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            outputs = nn.functional.interpolate(
                outputs,
                size=(HEATMAP_SIZE[0], HEATMAP_SIZE[1]),
                mode='bilinear',
                align_corners=False,
            )
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def _validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            targets = batch["heatmaps"].to(device)

            outputs = model(images)
            outputs = nn.functional.interpolate(
                outputs,
                size=(HEATMAP_SIZE[0], HEATMAP_SIZE[1]),
                mode='bilinear',
                align_corners=False,
            )
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_lines(
    data_dir: str,
    output_dir: str = "weights/pitch",
    config_path: str = "models/pitch/config/hrnetv2_w48_l.yaml",
    pretrained_weights: Optional[str] = None,
    batch_size: int = 8,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    num_workers: int = 4,
    use_focal_loss: bool = True,
    save_every: int = 10,
    device: Optional[str] = None,
) -> str:
    """
    Train pitch line detection model.

    Args:
        data_dir: Directory with training data.
        output_dir: Directory to save weights.
        config_path: Path to HRNet config YAML.
        pretrained_weights: Optional path to pretrained weights.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay for optimizer.
        num_workers: DataLoader workers.
        use_focal_loss: Use focal loss for class imbalance.
        save_every: Save checkpoint every N epochs.
        device: Device string.

    Returns:
        Path to best model weights.

    Example:
        >>> train_lines(
        ...     data_dir="soccernet/calibration",
        ...     output_dir="weights/pitch",
        ...     epochs=100,
        ... )
    """
    import yaml

    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Training pitch line detector on {dev}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Import model creation (deferred to avoid circular imports)
    try:
        from models.pitch.model.cls_hrnet_l import get_cls_net as get_line_model
    except ImportError:
        raise ImportError(
            "HRNet model not found. Ensure models/pitch/model/ is in path. "
            "See README for model download instructions."
        )

    model = get_line_model(config)

    # Replace final layer for our number of classes
    in_channels = model.last_layer.in_channels
    model.last_layer = nn.Conv2d(in_channels, NUM_LINE_CLASSES + 1, kernel_size=1, stride=1, padding=0)

    if pretrained_weights and Path(pretrained_weights).exists():
        print(f"Loading pretrained weights from {pretrained_weights}")
        state_dict = torch.load(pretrained_weights, map_location=dev)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(dev)

    # Load datasets
    train_dataset = LineDetectionDataset(data_dir, split="train", augment=True)
    val_dataset = LineDetectionDataset(data_dir, split="val", augment=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    criterion = LineHeatmapLoss(use_focal=use_focal_loss)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    best_model_path = output_path / "best_line_detector.pth"

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = _train_epoch(model, train_loader, optimizer, criterion, scaler, dev)
        val_loss = _validate(model, val_loader, criterion, dev)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model")

        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), output_path / f"line_detector_epoch_{epoch + 1}.pth")

    torch.save(model.state_dict(), output_path / "final_line_detector.pth")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return str(best_model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train pitch line detector")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="weights/pitch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_lines(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
