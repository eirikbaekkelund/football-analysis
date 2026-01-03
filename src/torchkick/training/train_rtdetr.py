"""
RT-DETR player detection training.

RT-DETR (Real-Time Detection Transformer) is a transformer-based
detection model that achieves YOLO-like speed with higher accuracy.

Key benefits:
- Apache 2.0 license (commercially friendly)
- ~25-35 FPS on RTX 2000 Ada
- Better handling of occlusions due to transformer attention

Reference: https://arxiv.org/abs/2304.08069

Example:
    >>> from torchkick.training import train_rtdetr
    >>> 
    >>> train_rtdetr(
    ...     data_zip="soccernet/tracking/train.zip",
    ...     epochs=20,
    ...     batch_size=32,
    ... )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import box_convert


def get_rtdetr_model(
    num_classes: int = 2,
    model_name: str = "PekingU/rtdetr_r50vd",
) -> Tuple:
    """
    Get RT-DETR model and processor.

    Args:
        num_classes: Number of classes (excluding background).
        model_name: HuggingFace model identifier.

    Returns:
        (model, processor) tuple.

    Example:
        >>> model, processor = get_rtdetr_model(num_classes=2)
        >>> model.to("cuda")
    """
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    model = RTDetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    processor = RTDetrImageProcessor.from_pretrained(model_name)

    return model, processor


def _adapt_batch_for_rtdetr(
    batch: Dict,
    processor,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Prepare batch for RT-DETR.

    Args:
        batch: Batch from PlayerTrackingDataset.
        processor: RTDetrImageProcessor.
        device: Target device.

    Returns:
        (pixel_values, targets) for RT-DETR.
    """
    # Convert tensors to numpy for processor
    images = []
    for img_tensor in batch['images']:
        img_np = img_tensor.permute(1, 2, 0).numpy()
        images.append(img_np)

    # Prepare targets in DETR format
    targets = []
    for i in range(len(images)):
        boxes = batch['boxes'][i]

        # Filter invalid boxes
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]

        if len(boxes) == 0:
            targets.append(
                {
                    "boxes": torch.zeros((0, 4)),
                    "class_labels": torch.zeros((0,), dtype=torch.long),
                }
            )
            continue

        # Normalize boxes to [0, 1] and convert to cxcywh
        h, w = images[i].shape[:2]
        boxes_norm = boxes.clone().float()
        boxes_norm[:, [0, 2]] /= w
        boxes_norm[:, [1, 3]] /= h

        boxes_cxcywh = box_convert(boxes_norm, "xyxy", "cxcywh")
        labels = torch.zeros(boxes_cxcywh.shape[0], dtype=torch.long)

        targets.append(
            {
                "boxes": boxes_cxcywh.to(device),
                "class_labels": labels.to(device),
            }
        )

    # Process images
    encoding = processor(images=images, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)

    return pixel_values, targets


def _train_one_epoch(
    model,
    processor,
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
        pixel_values, targets = _adapt_batch_for_rtdetr(batch, processor, device)

        with torch.amp.autocast('cuda'):
            outputs = model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if batch_idx % print_freq == 0:
            loss_dict = outputs.loss_dict if hasattr(outputs, 'loss_dict') else {}
            loss_str = " | ".join(f"{k}: {v.item():.3f}" for k, v in loss_dict.items()) if loss_dict else ""
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(data_loader)}] Loss: {loss.item():.4f} {loss_str}")

    return running_loss / len(data_loader)


def train_rtdetr(
    data_zip: str = "soccernet/tracking/tracking/train.zip",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_workers: int = 8,
    save_path: str = "rtdetr_player_tracker.pth",
    device: Optional[str] = None,
) -> str:
    """
    Train RT-DETR model for player detection.

    Args:
        data_zip: Path to SoccerNet tracking zip file.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        num_workers: DataLoader workers.
        save_path: Path to save final model weights.
        device: Device string ("cuda" or "cpu").

    Returns:
        Path to saved model weights.

    Example:
        >>> weights = train_rtdetr(
        ...     data_zip="soccernet/tracking/train.zip",
        ...     epochs=20,
        ... )
    """
    from torchkick.soccernet import PlayerTrackingDataset, tracking_collate_fn

    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"Training RT-DETR on {dev}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")

    # Load model
    model, processor = get_rtdetr_model(num_classes=2)
    model.to(dev)

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

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    for epoch in range(epochs):
        avg_loss = _train_one_epoch(model, processor, optimizer, train_loader, dev, epoch, scaler)
        lr_scheduler.step()

        checkpoint_path = f"rtdetr_checkpoint_epoch_{epoch}.pth"
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
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
    train_rtdetr()
