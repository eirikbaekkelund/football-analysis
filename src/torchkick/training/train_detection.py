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

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, random_split


def _make_phased_lr_lambda(
    phase1_steps: int,
    ramp_steps: int,
    total_steps: int,
    phase1_ratio: float = 0.01,
    min_lr_ratio: float = 0.01,
):
    """
    Three-phase LR schedule:
      1. Hold at phase1_ratio for phase1_steps steps  (near-frozen / protect pretrained weights).
      2. Linear ramp from phase1_ratio → 1.0 over ramp_steps steps.
      3. Cosine decay from 1.0 → min_lr_ratio over remaining steps.
    Set phase1_steps=0 and phase1_ratio=0.0 to get a plain warmup + cosine schedule.
    """
    ramp_end = phase1_steps + ramp_steps

    def lr_lambda(step: int) -> float:
        if step < phase1_steps:
            return phase1_ratio
        if step < ramp_end:
            t = (step - phase1_steps) / max(1, ramp_steps)
            return phase1_ratio + (1.0 - phase1_ratio) * t
        progress = (step - ramp_end) / max(1, total_steps - ramp_end)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return lr_lambda


_BACKBONE_HF_MAP = {
    "r50": "PekingU/rtdetr_r50vd",
    "r101": "PekingU/rtdetr_r101vd",
}


def _build_swin_t_rtdetr(num_labels: int):
    from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor

    config = RTDetrConfig(
        backbone="swin_tiny_patch4_window7_224",
        use_timm_backbone=True,
        backbone_kwargs={"pretrained": True, "out_indices": (1, 2, 3)},
        encoder_in_channels=[192, 384, 768],
        num_labels=num_labels,
    )
    model = RTDetrForObjectDetection(config)
    processor = RTDetrImageProcessor()
    print("Backbone: Swin-T (ImageNet-22k pretrained via timm)")
    return model, processor


def _get_model_and_processor(num_labels: int, backbone: str = "r101"):
    try:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        if backbone == "swin_t":
            return _build_swin_t_rtdetr(num_labels)

        hf_name = _BACKBONE_HF_MAP.get(backbone, backbone)
        processor = RTDetrImageProcessor.from_pretrained(hf_name)
        model = RTDetrForObjectDetection.from_pretrained(
            hf_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        print(f"Backbone: {backbone} ({hf_name})")
        return model, processor
    except ImportError:
        raise ImportError("transformers>=4.35.0 required. Install: pip install torchkick[reid]")


def _replace_cls_heads_with_mlp(model, num_labels: int, hidden_dim: int = 256):
    """Replace linear class_embed + enc_score_head with 2-layer MLP heads."""
    import torch.nn as nn

    def make_mlp():
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

    inner = getattr(model, "model", model)
    decoder = getattr(inner, "decoder", None)
    if decoder is not None and hasattr(decoder, "class_embed"):
        n = len(decoder.class_embed)
        decoder.class_embed = nn.ModuleList([make_mlp() for _ in range(n)])
    if hasattr(inner, "enc_score_head"):
        inner.enc_score_head = make_mlp()
    print(f"cls heads → MLP({hidden_dim}→{hidden_dim}→{num_labels}) + LayerNorm")
    return model


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
    warmup_ratio: float = 0.02,
    backbone_phase1_epochs: int = 5,
    backbone_ramp_epochs: int = 5,
    grad_accumulation: int = 2,
    val_split: float = 0.15,
    use_fsdp: bool = False,
    compile_model: bool = False,
    save_dir: str = "weights/detection/",
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
    focal_gamma: float = 3.0,
    resume_from: Optional[str] = None,
    cls_head_lr_scale: float = 20.0,
    cls_head_decay_epochs: int = 30,
    backbone: str = "r101",
    use_mlp_head: bool = False,
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
    model, _ = _get_model_and_processor(num_labels=num_labels, backbone=backbone)

    # Override VFL focal gamma
    if hasattr(model, "config") and hasattr(model.config, "focal_loss_gamma"):
        model.config.focal_loss_gamma = focal_gamma
        print(f"Focal gamma → {focal_gamma}")

    # Replace heads BEFORE loading checkpoint so key names align for MLP-to-MLP resumes.
    if use_mlp_head:
        model = _replace_cls_heads_with_mlp(model, num_labels=num_labels)

    start_epoch = 1
    best_map50 = 0.0
    ckpt_has_mlp_heads = False
    if resume_from:
        ckpt = torch.load(resume_from, map_location="cpu")
        # Detect whether checkpoint already has trained MLP heads (Sequential keys contain ".0.weight")
        ckpt_has_mlp_heads = any(
            ("class_embed" in k or "enc_score_head" in k) and ".0.weight" in k
            for k in ckpt["model_state_dict"]
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_epoch = ckpt["epoch"] + 1
        best_map50 = ckpt.get("map50", 0.0)
        print(f"Resumed from {resume_from} (epoch {ckpt['epoch']}, mAP@0.5={best_map50:.4f})")

    # Initialize classification head biases to focal-loss prior (≈ -4.6).
    # Skipped when resuming a checkpoint that already has trained MLP heads.
    if not resume_from or (use_mlp_head and not ckpt_has_mlp_heads):
        prior_bias = -math.log((1 - 0.01) / 0.01)  # ≈ -4.6
        for name, module in model.named_modules():
            if hasattr(module, "bias") and module.bias is not None:
                if ("class_embed" in name or "enc_score_head" in name) and module.bias.shape[0] == num_labels:
                    torch.nn.init.constant_(module.bias, prior_bias)
        print(f"Classification head biases initialised to {prior_bias:.3f} (focal prior p=0.01)")

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

    # Three-group optimizer:
    # - backbone:   0.1x LR  — prevent catastrophic forgetting of pretrained features
    # - cls heads:  10x LR   — re-initialized from scratch, needs fast convergence
    # - everything else: 1x  — pretrained encoder/decoder, moderate update
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    cls_head_params = [
        p for n, p in model.named_parameters() if ("class_embed" in n or "enc_score_head" in n) and "backbone" not in n
    ]
    other_params = [
        p
        for n, p in model.named_parameters()
        if "backbone" not in n and "class_embed" not in n and "enc_score_head" not in n
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": learning_rate * lr_backbone_scale},
            {"params": other_params, "lr": learning_rate},
            {"params": cls_head_params, "lr": learning_rate * cls_head_lr_scale},
        ],
        weight_decay=1e-4,
    )
    print(
        f"Optimizer groups: backbone={sum(p.numel() for p in backbone_params)/1e6:.1f}M@{learning_rate*lr_backbone_scale:.0e}  "
        f"cls_head={sum(p.numel() for p in cls_head_params)/1e6:.2f}M@{learning_rate*cls_head_lr_scale:.0e}  "
        f"other={sum(p.numel() for p in other_params)/1e6:.1f}M@{learning_rate:.0e}"
    )
    remaining_epochs = epochs - start_epoch + 1
    steps_per_epoch = n_train // (batch_size * grad_accumulation)
    total_steps = steps_per_epoch * remaining_epochs

    # Per-group phased schedules:
    #   backbone  — hold near-frozen for phase1 epochs, ramp over ramp epochs, cosine decay rest
    #   other     — short warmup then full cosine decay
    #   cls_head  — short warmup then aggressive cosine decay (cls_head_decay_epochs)
    backbone_phase1_steps = backbone_phase1_epochs * steps_per_epoch
    backbone_ramp_steps = backbone_ramp_epochs * steps_per_epoch
    other_warmup_steps = int(total_steps * warmup_ratio)
    cls_head_total_steps = other_warmup_steps + steps_per_epoch * min(cls_head_decay_epochs, remaining_epochs)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            _make_phased_lr_lambda(backbone_phase1_steps, backbone_ramp_steps, total_steps),          # backbone
            _make_phased_lr_lambda(0, other_warmup_steps, total_steps, phase1_ratio=0.0),             # other
            _make_phased_lr_lambda(0, other_warmup_steps, cls_head_total_steps, phase1_ratio=0.0),    # cls_head
        ],
    )
    print(
        f"Scheduler: backbone phase1={backbone_phase1_epochs}ep → ramp={backbone_ramp_epochs}ep → cosine | "
        f"cls_head warmup → decay over {min(cls_head_decay_epochs, remaining_epochs)}ep | "
        f"other warmup → cosine over {remaining_epochs}ep"
    )

    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None

    best_ckpt = str(save_path / "rtdetr_best.pth")
    try:
        from torchmetrics.detection import MeanAveragePrecision

        map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(dev)
    except ImportError:
        map_metric = None
        print("torchmetrics not found; checkpoint will fall back to val_loss. Install: pip install torchkick[training]")

    for epoch in range(start_epoch, epochs + 1):
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
            if epoch == start_epoch and step == 0:
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
                        f"  [input] boxes total: {total_boxes}  label dist: { {int(k): int(v) for k, v in zip(unique, counts)} }  (expect {{0: players, 1: balls}})"
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
                    print(f"  [output] logits shape: {logits.shape}  (expect [B, ~400, {num_labels}] in train mode)")
                    print(f"  [output] max pred score per image: {scores.max(dim=-1).values.tolist()}")
                print("=======================================\n", flush=True)

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accumulation == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
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
                        preds_map.append(
                            {
                                "boxes": abs_boxes.cpu(),
                                "scores": person_scores.cpu(),
                                "labels": torch.zeros(len(person_scores), dtype=torch.long),
                            }
                        )
                        targets_map.append(
                            {
                                "boxes": t["boxes"].cpu(),
                                "labels": t["labels"].cpu(),
                            }
                        )
                    map_metric.update(preds_map, targets_map)

                if not val_sanity_done:
                    val_sanity_done = True
                    if hasattr(out, "logits"):
                        s = out.logits.sigmoid()[:, :, 0]  # [B, 300]
                        top25 = s.topk(25, dim=-1).values  # [B, 25]
                        print(
                            f"  [val ep{epoch}] score dist — "
                            f"top1: {s.max(dim=-1).values.mean():.4f}  "
                            f"top25_mean: {top25.mean():.4f}  "
                            f"above_0.01: {(s > 0.01).float().sum(dim=-1).mean():.1f}/img  "
                            f"above_0.1: {(s > 0.1).float().sum(dim=-1).mean():.1f}/img",
                            flush=True,
                        )

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
            f"map@0.5={map50:.4f} map@1.0={map_all:.4f} | lr={cur_lr:.2e} | {elapsed:.1f}s"
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
