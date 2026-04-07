"""
ViTPose fine-tuning on FIFA Skeletal Tracking 2026 body pose data.

Fine-tunes usyd-community/vitpose-base-simple (COCO-17 joints) on broadcast
soccer frames using heatmap-based MSE supervision.  Target heatmaps are
generated as 2-D Gaussians centred at each visible keypoint.

The FIFA dataset provides Body25 annotations — these are remapped to COCO-17
automatically by FIFAPoseDataset.

Data must be downloaded before use:
    huggingface-cli download tijiang13/FIFA-Skeletal-Tracking-Light-2026 \\
        --repo-type dataset --local-dir data/fifa/

Example:
    >>> from torchkick.training.train_body_pose import train_body_pose
    >>> weights = train_body_pose(
    ...     data_dir="data/fifa/",
    ...     epochs=30,
    ...     save_dir="weights/body_pose/",
    ... )

CLI:
    $ torchkick train body-pose --data data/fifa/ --epochs 30
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ViTPose input / heatmap sizes
_INPUT_H = 256
_INPUT_W = 192
_HM_H = 64  # _INPUT_H // 4
_HM_W = 48  # _INPUT_W // 4
_N_JOINTS = 17
_HM_SIGMA = 2.0  # Gaussian σ in heatmap pixels


def _make_heatmaps(
    kp2d: np.ndarray,  # [17, 2] pixel coords in input image space
    vis: np.ndarray,  # [17] visibility
) -> np.ndarray:
    """
    Convert keypoint coordinates to Gaussian heatmaps.

    Args:
        kp2d: [17, 2] pixel coordinates in _INPUT_H×_INPUT_W space.
        vis:  [17] visibility flags (1=visible, 0=hidden).

    Returns:
        [17, _HM_H, _HM_W] float32 heatmaps with values in [0, 1].
    """
    heatmaps = np.zeros((_N_JOINTS, _HM_H, _HM_W), dtype=np.float32)
    scale_x = _HM_W / _INPUT_W
    scale_y = _HM_H / _INPUT_H

    # Pre-build coordinate grids (reused for every joint)
    xs = np.arange(_HM_W, dtype=np.float32)
    ys = np.arange(_HM_H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)  # [HM_H, HM_W]

    for i in range(_N_JOINTS):
        if vis[i] < 0.5:
            continue
        cx = kp2d[i, 0] * scale_x
        cy = kp2d[i, 1] * scale_y
        heatmaps[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * _HM_SIGMA**2))

    return heatmaps


def _collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    kp2d = [b[1].numpy() for b in batch]
    vis = [b[2].numpy() for b in batch]
    heatmaps = torch.from_numpy(np.stack([_make_heatmaps(k, v) for k, v in zip(kp2d, vis)]))
    vis_t = torch.stack([b[2] for b in batch])
    return images, heatmaps, vis_t


def train_body_pose(
    data_dir: str,
    model_id: str = "usyd-community/vitpose-base-simple",
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    warmup_epochs: int = 3,
    val_split: float = 0.1,
    save_dir: str = "weights/body_pose/",
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> str:
    """
    Fine-tune ViTPose on FIFA body pose data.

    Args:
        data_dir: Root of the downloaded FIFA dataset.
        model_id: HuggingFace model ID or local path for ViTPose.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate (backbone uses 0.1× this).
        warmup_epochs: Linear LR warmup epochs.
        val_split: Fraction of training data used for validation.
        save_dir: Directory to save checkpoints.
        device: "cuda" / "cpu" / "mps". Auto-detects if None.
        wandb_project: W&B project name (optional).
        max_samples: Cap dataset size for smoke tests.

    Returns:
        Path to best model checkpoint.
    """
    try:
        from transformers import ViTPoseForPoseEstimation, ViTPoseImageProcessor
    except ImportError:
        raise ImportError("transformers>=4.46.0 required. Install: pip install torchkick[reid]")

    from torchkick.training.data.fifa_pose_dataset import FIFAPoseDataset

    dev = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Device: {dev}")

    run = None
    if wandb_project:
        try:
            import wandb

            run = wandb.init(project=wandb_project, config={"model": model_id, "epochs": epochs})
        except ImportError:
            pass

    # ------------------------------------------------------------------ Data
    train_ds = FIFAPoseDataset(data_dir, split="train", mode="pose_2d", augment=True, max_samples=max_samples)
    val_ds = FIFAPoseDataset(
        data_dir, split="val", mode="pose_2d", augment=False, max_samples=max_samples and max_samples // 5
    )

    # If no val split file exists, carve from training set
    if len(val_ds) == 0:
        from torch.utils.data import random_split

        n_val = max(1, int(len(train_ds) * val_split))
        train_ds, val_ds = random_split(train_ds, [len(train_ds) - n_val, n_val])
        print(f"Carved val set: {len(train_ds)} train | {len(val_ds)} val")

    n_workers = min(4, torch.get_num_threads())
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=_collate_fn
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ----------------------------------------------------------------- Model
    print(f"Loading ViTPose: {model_id}")
    processor = ViTPoseImageProcessor.from_pretrained(model_id)
    model = ViTPoseForPoseEstimation.from_pretrained(model_id)
    model = model.to(dev)

    # ViTPose preprocessing normalisation constants (ImageNet)
    _MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    # 2-group optimizer: backbone at 0.1× LR
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": head_params, "lr": learning_rate},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler() if dev.type == "cuda" else None
    loss_fn = nn.MSELoss(reduction="none")

    best_val_loss = float("inf")
    best_ckpt = str(save_path / "vitpose_body_best.pth")

    # dataset_index selects the ViTPose head (0 = COCO)
    def _dataset_index(b: int) -> torch.Tensor:
        return torch.zeros(b, dtype=torch.long, device=dev)

    for epoch in range(1, epochs + 1):
        # Linear LR warmup
        if epoch <= warmup_epochs:
            scale = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * scale if "initial_lr" in pg else learning_rate * scale

        model.train()
        train_loss = 0.0
        t0 = time.time()

        for images, hm_gt, vis_gt in train_loader:
            # images: [B, 3, H, W] in [0,1] (RGB), already resized to 256×192
            images = images.to(dev)
            hm_gt = hm_gt.to(dev)  # [B, 17, 64, 48]
            vis_gt = vis_gt.to(dev)  # [B, 17]

            # Normalise for ImageNet backbone
            images_norm = (images - _MEAN) / _STD

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=dev.type, enabled=scaler is not None):
                outputs = model(pixel_values=images_norm, dataset_index=_dataset_index(images.size(0)))
                hm_pred = outputs.heatmaps  # [B, 17, 64, 48]

                # Weight loss by visibility: invisible joints don't contribute
                vis_mask = vis_gt.unsqueeze(-1).unsqueeze(-1)  # [B, 17, 1, 1]
                loss = (loss_fn(hm_pred, hm_gt) * vis_mask).sum() / (vis_mask.sum() * _HM_H * _HM_W + 1e-6)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        if epoch > warmup_epochs:
            scheduler.step()

        # ---------------------------------------------------------------- Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, hm_gt, vis_gt in val_loader:
                images = (images.to(dev) - _MEAN) / _STD
                hm_gt = hm_gt.to(dev)
                vis_gt = vis_gt.to(dev)
                with torch.amp.autocast(device_type=dev.type, enabled=scaler is not None):
                    outputs = model(pixel_values=images, dataset_index=_dataset_index(images.size(0)))
                    hm_pred = outputs.heatmaps
                    vis_mask = vis_gt.unsqueeze(-1).unsqueeze(-1)
                    val_loss += (loss_fn(hm_pred, hm_gt) * vis_mask).sum().item() / (
                        vis_mask.sum().item() * _HM_H * _HM_W + 1e-6
                    )

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | train={avg_train:.5f} val={avg_val:.5f} | {elapsed:.1f}s")

        if run:
            run.log({"train_loss": avg_train, "val_loss": avg_val, "epoch": epoch})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_obj = model if not hasattr(model, "_orig_mod") else model._orig_mod
            torch.save({"model_state_dict": save_obj.state_dict(), "epoch": epoch, "val_loss": avg_val}, best_ckpt)
            print(f"  Saved → {best_ckpt}")

    if run:
        run.finish()

    print(f"\nTraining complete. Best checkpoint: {best_ckpt}")
    return best_ckpt
