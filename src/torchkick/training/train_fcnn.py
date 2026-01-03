"""
Faster R-CNN player detection training.

This module provides training utilities for Faster R-CNN (FCNN)
based player detection using SoccerNet tracking data.

Faster R-CNN provides higher accuracy for small objects (distant players)
compared to YOLO, with a latency trade-off.

Example:
    >>> from torchkick.training import train_fcnn, get_player_detector_model
    >>> 
    >>> # Get model
    >>> model = get_player_detector_model(num_classes=2)
    >>> 
    >>> # Train
    >>> train_fcnn(
    ...     data_zip="soccernet/tracking/train.zip",
    ...     epochs=10,
    ...     batch_size=64,
    ... )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_player_detector_model(num_classes: int = 2) -> torch.nn.Module:
    """
    Get a pre-trained Faster R-CNN model for player detection.

    The model is initialized with COCO pre-trained weights and the
    final prediction layer is replaced for the target number of classes.

    Args:
        num_classes: Number of output classes including background.
            Use 2 for (background, player).

    Returns:
        Faster R-CNN model ready for training.

    Example:
        >>> model = get_player_detector_model(num_classes=2)
        >>> model.to("cuda")
        >>> model.eval()
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def _adapt_batch_for_torchvision(
    batch: Dict,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Convert batch to torchvision format.

    Torchvision detection models expect:
    - images: List[Tensor[C, H, W]]
    - targets: List[Dict{'boxes': Tensor, 'labels': Tensor}]

    Args:
        batch: Batch from PlayerTrackingDataset.
        device: Target device.

    Returns:
        (images, targets) in torchvision format.
    """
    images = [img.to(device).float() / 255.0 for img in batch['images']]

    targets = []
    for i in range(len(images)):
        boxes = batch['boxes'][i].to(device)

        # Filter invalid boxes
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]

        num_objs = boxes.shape[0]
        labels = torch.ones((num_objs,), dtype=torch.int64, device=device)

        targets.append(
            {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([i], device=device),
            }
        )

    return images, targets


def _train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler,
    print_freq: int = 10,
) -> float:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(data_loader):
        images, targets = _adapt_batch_for_torchvision(batch, device)

        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += losses.item()

        if batch_idx % print_freq == 0:
            cls_loss = loss_dict['loss_classifier'].item()
            box_loss = loss_dict['loss_box_reg'].item()
            print(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(data_loader)}] "
                f"Loss: {losses.item():.4f} (Cls: {cls_loss:.3f} | Box: {box_loss:.3f})"
            )

    return running_loss / len(data_loader)


def train_fcnn(
    data_zip: str = "soccernet/tracking/tracking/train.zip",
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    num_workers: int = 8,
    save_path: str = "fcnn_player_tracker.pth",
    device: Optional[str] = None,
) -> str:
    """
    Train Faster R-CNN model for player detection.

    Args:
        data_zip: Path to SoccerNet tracking zip file.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Base learning rate (scaled by batch size).
        num_workers: DataLoader workers.
        save_path: Path to save final model weights.
        device: Device string ("cuda" or "cpu").

    Returns:
        Path to saved model weights.

    Example:
        >>> weights = train_fcnn(
        ...     data_zip="soccernet/tracking/train.zip",
        ...     epochs=10,
        ...     batch_size=64,
        ... )
    """
    from torchkick.soccernet import PlayerTrackingDataset, tracking_collate_fn

    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    lr = learning_rate * (batch_size / 32)

    print(f"Training Faster R-CNN on {dev}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")

    # Load data
    train_dataset = PlayerTrackingDataset(
        zip_path=data_zip,
        bbox_format="xyxy",
        extract_colors=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=tracking_collate_fn,
    )

    # Initialize model
    model = get_player_detector_model(num_classes=2)
    model.to(dev)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler('cuda')
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    for epoch in range(epochs):
        avg_loss = _train_one_epoch(model, optimizer, train_loader, dev, epoch, scaler)
        lr_scheduler.step()

        checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            },
            checkpoint_path,
        )
        print(f"Epoch {epoch} complete. Loss: {avg_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    train_fcnn()
