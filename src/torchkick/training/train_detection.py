"""
RT-DETR-X player/ball/referee detection trainer.

Replaces train_fcnn.py and train_rtdetr.py. Trains PekingU/rtdetr_r101vd
(RT-DETR-X variant) with torch.compile, cosine LR with warmup, and optional
FSDP for multi-GPU runs.

Data sources (via MixedDetectionDataset):
    - SoccerNet tracking ZIP files
    - Roboflow football COCO JSON exports
    - CVAT human-annotated COCO JSON exports

Usage:
    >>> from torchkick.training.train_detection import train_detection
    >>> weights = train_detection(
    ...     data_config=[
    ...         {"type": "soccernet_zip", "path": "data/train.zip"},
    ...         {"type": "coco_json", "path": "data/roboflow.json", "images_dir": "data/images/"},
    ...     ],
    ...     epochs=50,
    ...     save_dir="weights/detection/",
    ... )

CLI:
    $ torchkick train detection --data data/train.zip --epochs 50
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, random_split


def _get_model_and_processor(model_name: str):
    try:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        processor = RTDetrImageProcessor.from_pretrained(model_name)
        model = RTDetrForObjectDetection.from_pretrained(model_name)
        return model, processor
    except ImportError:
        raise ImportError("transformers>=4.35.0 required. Install: pip install torchkick[reid]")


def _collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def train_detection(
    data_config: List[Dict[str, Any]],
    model_name: str = "PekingU/rtdetr_r101vd",
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.05,
    grad_accumulation: int = 4,
    val_split: float = 0.1,
    use_fsdp: bool = False,
    compile_model: bool = False,
    save_dir: str = "weights/detection/",
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
) -> str:
    """
    Train RT-DETR-X on mixed soccer detection data.

    Args:
        data_config: List of source dicts for MixedDetectionDataset.
        model_name: HuggingFace model identifier.
        epochs: Number of training epochs.
        batch_size: Per-GPU batch size.
        learning_rate: Peak learning rate.
        warmup_ratio: Fraction of steps used for LR warmup.
        grad_accumulation: Gradient accumulation steps.
        val_split: Fraction of data held out for validation.
        use_fsdp: Enable PyTorch FSDP for multi-GPU training.
        compile_model: Apply torch.compile (recommended for A100/H100).
        save_dir: Directory to save checkpoints.
        device: Device string. Auto-detects CUDA if None.
        wandb_project: W&B project name. Skips W&B logging if None.

    Returns:
        Path to best model checkpoint.
    """
    from torchkick.training.data.detection_dataset import MixedDetectionDataset

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # W&B logging (optional)
    run = None
    if wandb_project:
        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                config={
                    "model_name": model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                },
            )
        except ImportError:
            print("W&B not installed; skipping logging.")

    # Dataset — build index once, split with fixed seed, wrap train subset for augmentation
    print("Loading dataset...")
    input_size = 640
    full_dataset = MixedDetectionDataset(data_config, input_size=input_size, augment=False)
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    val_ds, _ = random_split(full_dataset, [n_val, n_train], generator=generator)

    # Re-use same indices for training but with augmentation enabled
    aug_dataset = MixedDetectionDataset(data_config, input_size=input_size, augment=True)
    generator2 = torch.Generator().manual_seed(42)
    _, train_ds = random_split(aug_dataset, [n_val, n_train], generator=generator2)

    import os
    n_cpu = os.cpu_count() or 4
    n_workers = min(n_cpu, 12)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=4, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=max(n_workers // 2, 2),
        pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=_collate_fn,
    )

    print(f"Train: {n_train} samples | Val: {n_val} samples")

    # Model
    model, processor = _get_model_and_processor(model_name)

    if use_fsdp:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            model = FSDP(model)
        except Exception as e:
            print(f"FSDP init failed ({e}); falling back to single-GPU.")
    else:
        model = model.to(dev)

    if compile_model and not use_fsdp:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    total_steps = (n_train // (batch_size * grad_accumulation)) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    try:
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    except ImportError:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None

    best_val_loss = float("inf")
    best_ckpt = str(save_path / "rtdetr_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(dev)
            # Build HuggingFace-compatible labels
            hf_labels = []
            for t in targets:
                boxes = t["boxes"].to(dev)
                labels_t = t["labels"].to(dev)
                # Convert xyxy to cxcywh normalized
                if len(boxes):
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / input_size
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / input_size
                    bw = (boxes[:, 2] - boxes[:, 0]) / input_size
                    bh = (boxes[:, 3] - boxes[:, 1]) / input_size
                    valid = (bw > 0) & (bh > 0)
                    boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=1)[valid]
                    labels_t = labels_t[valid]
                else:
                    boxes_cxcywh = boxes
                hf_labels.append({"class_labels": labels_t, "boxes": boxes_cxcywh})

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                outputs = model(pixel_values=images, labels=hf_labels)
                loss = outputs.loss / grad_accumulation

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accumulation == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                if hasattr(scheduler, "step") and isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
                    pass  # CosineAnnealingLR steps per epoch
                else:
                    scheduler.step()
                optimizer.zero_grad()

                update = (step + 1) // grad_accumulation
                if update % 100 == 0:
                    print(f"  [{epoch}/{epochs}] update {update} loss={loss.item() * grad_accumulation:.4f}", flush=True)

            train_loss += loss.item() * grad_accumulation

        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(dev)
                hf_labels = []
                for t in targets:
                    boxes = t["boxes"].to(dev)
                    labels_t = t["labels"].to(dev)
                    if len(boxes):
                        cx = (boxes[:, 0] + boxes[:, 2]) / 2 / input_size
                        cy = (boxes[:, 1] + boxes[:, 3]) / 2 / input_size
                        bw = (boxes[:, 2] - boxes[:, 0]) / input_size
                        bh = (boxes[:, 3] - boxes[:, 1]) / input_size
                        valid = (bw > 0) & (bh > 0)
                        boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=1)[valid]
                        labels_t = labels_t[valid]
                    else:
                        boxes_cxcywh = boxes
                    hf_labels.append({"class_labels": labels_t, "boxes": boxes_cxcywh})
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    out = model(pixel_values=images, labels=hf_labels)
                val_loss += out.loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} | train={avg_train:.4f} val={avg_val:.4f} | {elapsed:.1f}s")

        if run:
            run.log({"train_loss": avg_train, "val_loss": avg_val, "epoch": epoch})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Unwrap compiled/FSDP model for saving
            save_model = model
            if hasattr(model, "_orig_mod"):
                save_model = model._orig_mod  # type: ignore[attr-defined]
            torch.save({"model_state_dict": save_model.state_dict(), "epoch": epoch, "val_loss": avg_val}, best_ckpt)
            print(f"  Saved best checkpoint → {best_ckpt}")

    if run:
        run.finish()

    print(f"\nTraining complete. Best checkpoint: {best_ckpt}")
    return best_ckpt
