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


def _get_model_and_processor(model_name: str, num_labels: int = 2):
    try:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        processor = RTDetrImageProcessor.from_pretrained(model_name)
        model = RTDetrForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
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
    num_labels: int = 1,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    lr_backbone_scale: float = 0.1,
    warmup_ratio: float = 0.05,
    grad_accumulation: int = 2,
    val_split: float = 0.15,
    use_fsdp: bool = False,
    compile_model: bool = False,
    save_dir: str = "weights/detection/",
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
    conf_threshold: float = 0.3,
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
    n_workers = min(n_cpu, 16)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(n_workers // 2, 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=_collate_fn,
    )

    print(f"Train: {n_train} samples | Val: {n_val} samples")

    # Model
    model, processor = _get_model_and_processor(model_name, num_labels=num_labels)

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

    # Optimizer & Scheduler — backbone gets 10x lower LR to prevent catastrophic forgetting
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    other_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": learning_rate * lr_backbone_scale},
            {"params": other_params, "lr": learning_rate},
        ],
        weight_decay=1e-4,
    )
    total_steps = (n_train // (batch_size * grad_accumulation)) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    try:
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    except ImportError:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None

    best_map50 = 0.0
    best_ckpt = str(save_path / "rtdetr_best.pth")
    try:
        from torchmetrics.detection import MeanAveragePrecision

        map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(dev)
    except ImportError:
        map_metric = None
        print("torchmetrics not found; checkpoint will fall back to val_loss. Install: pip install torchkick[training]")

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

            # One-time sanity check after first forward pass
            if epoch == 1 and step == 0:
                print("\n=== SANITY CHECK (epoch 1, step 0) ===")
                # --- Inputs ---
                print(f"  [input] images: {images.shape} {images.dtype}")
                print(f"  [input] mean/std: {images.mean().item():.3f}/{images.std().item():.3f}  (expect ~0.0/1.0)")
                print(f"  [input] min/max:  {images.min().item():.3f}/{images.max().item():.3f}  (expect ~-2.1/+2.6)")
                total_boxes = sum(len(lbl["boxes"]) for lbl in hf_labels)
                all_labels_list = [lbl["class_labels"] for lbl in hf_labels if len(lbl["class_labels"])]
                if all_labels_list:
                    all_cls = torch.cat(all_labels_list)
                    unique, counts = all_cls.unique(return_counts=True)
                    print(
                        f"  [input] boxes total: {total_boxes}  label dist: { {int(k): int(v) for k, v in zip(unique, counts)} }  (expect {{0: N}})"
                    )
                if total_boxes:
                    all_boxes_cxcywh = torch.cat([lbl["boxes"] for lbl in hf_labels if len(lbl["boxes"])])
                    print(
                        f"  [input] boxes cxcywh range: cx=[{all_boxes_cxcywh[:,0].min():.2f},{all_boxes_cxcywh[:,0].max():.2f}] "
                        f"cy=[{all_boxes_cxcywh[:,1].min():.2f},{all_boxes_cxcywh[:,1].max():.2f}] "
                        f"w=[{all_boxes_cxcywh[:,2].min():.3f},{all_boxes_cxcywh[:,2].max():.3f}]  (expect all in 0-1)"
                    )
                # --- Outputs ---
                print(f"  [output] loss: {outputs.loss.item():.4f}")
                loss_dict = getattr(outputs, "loss_dict", {})
                if loss_dict:
                    print(f"  [output] loss_dict keys: {list(loss_dict.keys())}")
                    for k, v in loss_dict.items():
                        print(f"    {k}: {v.item():.4f}")
                # Predicted logits: shape [B, num_queries, num_classes]
                if hasattr(outputs, "logits"):
                    logits = outputs.logits  # [B, Q, C]
                    scores = logits.sigmoid().max(dim=-1).values  # [B, Q]
                    print(f"  [output] logits shape: {logits.shape}  (expect [B, ~460, {num_labels}] in train mode)")
                    print(f"  [output] max pred score per image: {scores.max(dim=-1).values.tolist()}")
                print("=======================================\n", flush=True)

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
                if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    import warnings

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step()`")
                        scheduler.step()
                optimizer.zero_grad()

                update = (step + 1) // grad_accumulation
                if update % 100 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    loss_dict = getattr(outputs, "loss_dict", {})
                    loss_cls = loss_dict.get("loss_vfl", loss_dict.get("loss_ce", float("nan")))
                    loss_bbox = loss_dict.get("loss_bbox", float("nan"))
                    loss_giou = loss_dict.get("loss_giou", float("nan"))
                    if hasattr(loss_cls, "item"):
                        loss_cls = loss_cls.item()
                    if hasattr(loss_bbox, "item"):
                        loss_bbox = loss_bbox.item()
                    if hasattr(loss_giou, "item"):
                        loss_giou = loss_giou.item()
                    print(
                        f"  [{epoch}/{epochs}] update {update} | "
                        f"loss={loss.item() * grad_accumulation:.4f} "
                        f"cls={loss_cls:.3f} bbox={loss_bbox:.3f} giou={loss_giou:.3f} | "
                        f"lr={cur_lr:.2e}",
                        flush=True,
                    )

            train_loss += loss.item() * grad_accumulation

        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_sanity_done = False
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

                # Accumulate mAP — use class-0 (person) score directly; force predicted label=0
                if map_metric is not None and hasattr(out, "logits") and hasattr(out, "pred_boxes"):
                    scores_batch = out.logits.sigmoid()  # [B, 300, num_labels]
                    boxes_batch = out.pred_boxes  # [B, 300, 4] normalized cxcywh
                    preds_map = []
                    targets_map = []
                    for t, scores_img, boxes_img in zip(targets, scores_batch, boxes_batch):
                        cx, cy, bw, bh = boxes_img.unbind(-1)
                        x1 = (cx - bw / 2) * input_size
                        y1 = (cy - bh / 2) * input_size
                        x2 = (cx + bw / 2) * input_size
                        y2 = (cy + bh / 2) * input_size
                        abs_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                        person_scores = scores_img[:, 0]  # class-0 (person) confidence
                        keep = person_scores > conf_threshold
                        preds_map.append(
                            {
                                "boxes": abs_boxes[keep].cpu(),
                                "scores": person_scores[keep].cpu(),
                                "labels": torch.zeros(keep.sum(), dtype=torch.long),
                            }
                        )
                        targets_map.append(
                            {
                                "boxes": t["boxes"].cpu(),
                                "labels": t["labels"].cpu(),
                            }
                        )
                    map_metric.update(preds_map, targets_map)

                if epoch == 1 and not val_sanity_done:
                    val_sanity_done = True
                    print("\n=== VAL SANITY CHECK (epoch 1, first val batch) ===")
                    print(
                        f"  [input] images: {images.shape}  mean/std: {images.mean().item():.3f}/{images.std().item():.3f}"
                    )
                    total_boxes = sum(len(lbl["boxes"]) for lbl in hf_labels)
                    all_labels_list = [lbl["class_labels"] for lbl in hf_labels if len(lbl["class_labels"])]
                    if all_labels_list:
                        all_cls = torch.cat(all_labels_list)
                        unique, counts = all_cls.unique(return_counts=True)
                        print(
                            f"  [input] boxes: {total_boxes}  label dist: { {int(k): int(v) for k, v in zip(unique, counts)} }"
                        )
                    print(f"  [output] val loss: {out.loss.item():.4f}")
                    loss_dict = getattr(out, "loss_dict", {})
                    if loss_dict:
                        for k, v in loss_dict.items():
                            print(f"    {k}: {v.item():.4f}")
                    if hasattr(out, "logits"):
                        person_scores_val = out.logits.sigmoid()[:, :, 0]  # class-0 (person)
                        print(
                            f"  [output] logits: {out.logits.shape}  (expect [B, 300, {num_labels}])  "
                            f"max person score per image: {person_scores_val.max(dim=-1).values.tolist()}"
                        )
                    print("====================================================\n", flush=True)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        map50 = 0.0
        map_all = 0.0
        if map_metric is not None:
            map_result = map_metric.compute()
            map50 = map_result["map_50"].item()
            map_all = map_result["map"].item()
            map_metric.reset()

        print(
            f"Epoch {epoch}/{epochs} | train={avg_train:.4f} val={avg_val:.4f} "
            f"map@0.5={map50:.4f} map={map_all:.4f} | lr={cur_lr:.2e} | {elapsed:.1f}s"
        )

        if run:
            run.log(
                {
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "map50": map50,
                    "map": map_all,
                    "lr": cur_lr,
                    "epoch": epoch,
                }
            )

        is_best = (map50 > best_map50) if map_metric is not None else False
        if is_best:
            best_map50 = map50
            # Unwrap compiled/FSDP model for saving
            save_model = model
            if hasattr(model, "_orig_mod"):
                save_model = model._orig_mod  # type: ignore[attr-defined]
            torch.save(
                {"model_state_dict": save_model.state_dict(), "epoch": epoch, "val_loss": avg_val, "map50": map50},
                best_ckpt,
            )
            print(f"  Saved best checkpoint → {best_ckpt}  (map@0.5={map50:.4f})")

    if run:
        run.finish()

    print(f"\nTraining complete. Best checkpoint: {best_ckpt}")
    return best_ckpt
