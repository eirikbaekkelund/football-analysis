"""
RT-DETR Player Detection Training Script

RT-DETR (Real-Time Detection Transformer) is a commercial-friendly (Apache 2.0)
alternative to YOLO that achieves similar speed with transformer-based architecture.

Expected performance on RTX 2000 Ada: ~25-35 FPS (7-10x faster than Faster-RCNN)

Reference: https://arxiv.org/abs/2304.08069
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

# RT-DETR from HuggingFace transformers (Apache 2.0)
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from soccernet.tracking_data import PlayerTrackingDataset, tracking_collate_fn


def get_rtdetr_model(num_classes: int = 2, model_size: str = "l"):
    """
    Returns RT-DETR model configured for player detection.

    Args:
        num_classes: Number of classes (excluding background). 1 = player only
        model_size: "l" (large, more accurate) or "x" (extra large)

    Returns:
        RT-DETR model and processor
    """
    model_name = f"PekingU/rtdetr_r50vd"  # ResNet-50 backbone

    # Load pretrained model
    model = RTDetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    processor = RTDetrImageProcessor.from_pretrained(model_name)

    return model, processor


def adapt_batch_for_rtdetr(batch, processor, device):
    """
    Prepare batch for RT-DETR using HuggingFace processor.
    """
    # Convert tensors to PIL images for processor
    images = []
    for img_tensor in batch['images']:
        # [C, H, W] -> [H, W, C] numpy
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
            # Empty target
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

        # xyxy -> cxcywh
        boxes_cxcywh = box_convert(boxes_norm, "xyxy", "cxcywh")

        num_objs = boxes_cxcywh.shape[0]
        labels = torch.zeros(num_objs, dtype=torch.long)  # Class 0 = player

        targets.append(
            {
                "boxes": boxes_cxcywh,
                "class_labels": labels,
            }
        )

    # Process images
    encoding = processor(
        images=images,
        return_tensors="pt",
    )

    pixel_values = encoding["pixel_values"].to(device)

    # Move targets to device
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["class_labels"] = t["class_labels"].to(device)

    return pixel_values, targets


def train_one_epoch(model, processor, optimizer, data_loader, device, epoch, scaler, print_freq=10):
    """Train RT-DETR for one epoch."""
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(data_loader):
        pixel_values, targets = adapt_batch_for_rtdetr(batch, processor, device)

        with torch.amp.autocast('cuda'):
            outputs = model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if batch_idx % print_freq == 0:
            # Get individual losses if available
            loss_dict = outputs.loss_dict if hasattr(outputs, 'loss_dict') else {}
            loss_str = " | ".join(f"{k}: {v.item():.3f}" for k, v in loss_dict.items()) if loss_dict else ""
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(data_loader)}] " f"Loss: {loss.item():.4f} {loss_str}")

    return running_loss / len(data_loader)


def run_training_pipeline(
    path_to_data_zip: str = "soccernet/tracking/tracking/train.zip",
):
    """
    Main training pipeline for RT-DETR player detection.

    Args:
        path_to_data_zip: Path to SoccerNet tracking data
    """
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # RT-DETR is more memory efficient, can use larger batches
    BATCH_SIZE = 16  # Adjust based on VRAM
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.0001  # Lower LR for transformer (AdamW)
    SAVE_PATH = "rtdetr_player_tracker.pth"

    print(f"Starting RT-DETR Training on {DEVICE}")
    print(f"  Model: PekingU/rtdetr_r50vd (HuggingFace)")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  License: Apache 2.0 (commercial-safe)")

    # Model and processor
    model, processor = get_rtdetr_model(num_classes=1)  # 1 class = player
    model.to(DEVICE)

    # Dataset
    train_dataset = PlayerTrackingDataset(
        zip_path=path_to_data_zip,
        bbox_format="xyxy",
        extract_colors=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=tracking_collate_fn,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    scaler = torch.amp.GradScaler('cuda')

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )

    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(model, processor, optimizer, train_loader, DEVICE, epoch, scaler)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # Save checkpoint
        model.save_pretrained(f"rtdetr_checkpoint_epoch_{epoch}")
        processor.save_pretrained(f"rtdetr_checkpoint_epoch_{epoch}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(SAVE_PATH.replace('.pth', ''))
            processor.save_pretrained(SAVE_PATH.replace('.pth', ''))
            print(f"  -> Saved best model with loss {best_loss:.4f}")

    print(f"\nTraining complete! Best model saved to {SAVE_PATH.replace('.pth', '')}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RT-DETR for player detection")
    parser.add_argument(
        "--data", type=str, default="soccernet/tracking/tracking/train.zip", help="Path to training data zip"
    )

    args = parser.parse_args()
    run_training_pipeline(args.data)
