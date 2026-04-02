"""
ViTPose-L pitch landmark keypoint trainer.

Trains a ViTPose-L model on SoccerNet pitch calibration data to detect
29 pitch landmarks (center circle, penalty spots, corner flags, 18-yard box
corners, etc.). Replaces train_lines.py.

Loss:
    - SmoothL1Loss on (x, y) regression for visible keypoints
    - BCEWithLogitsLoss on visibility prediction

Example:
    >>> from torchkick.training.train_keypoints import train_keypoints
    >>> weights = train_keypoints(
    ...     data_zip="data/soccernet/calibration/train.zip",
    ...     epochs=100,
    ...     save_dir="weights/keypoints/",
    ... )

CLI:
    $ torchkick train keypoints --data data/calibration/train.zip --epochs 100
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


class KeypointHead(nn.Module):
    """
    Simple regression head on top of a ViT backbone.

    Predicts (x, y) normalized coordinates and visibility for each keypoint.
    Uses the CLS token from the ViT backbone.
    """

    def __init__(self, in_features: int, num_keypoints: int = 29) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.coord_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_keypoints * 2),
            # No Sigmoid — unbounded regression avoids gradient saturation at pitch edges
        )
        self.vis_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_keypoints),
        )

    def forward(self, cls_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = self.coord_head(cls_token).view(-1, self.num_keypoints, 2)
        visibility = self.vis_head(cls_token)
        return coords, visibility


def _collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    keypoints = torch.stack([b[1] for b in batch])
    visibility = torch.stack([b[2] for b in batch])
    return images, keypoints, visibility


def train_keypoints(
    data_zip: str,
    model_variant: str = "ViTPose-L",
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    warmup_epochs: int = 5,
    val_split: float = 0.1,
    compile_model: bool = True,
    save_dir: str = "weights/keypoints/",
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
    num_keypoints: int = 29,
) -> str:
    """
    Train ViTPose-L on SoccerNet pitch calibration data.

    Args:
        data_zip: Path to SoccerNet calibration zip file.
        model_variant: Model size. "ViTPose-L" uses vit_large_patch16_224.
        epochs: Number of training epochs.
        batch_size: Per-GPU batch size (small due to high-res images).
        learning_rate: Peak learning rate.
        warmup_epochs: Epochs for linear LR warmup.
        val_split: Validation fraction.
        compile_model: Apply torch.compile.
        save_dir: Directory to save checkpoints.
        device: Device string. Auto-detects CUDA if None.
        wandb_project: W&B project name.

    Returns:
        Path to best model checkpoint.
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm>=0.9.0 required. Install: pip install torchkick[reid]")

    from torchkick.training.data.keypoint_dataset import KeypointAugDataset

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    run = None
    if wandb_project:
        try:
            import wandb

            run = wandb.init(
                project=wandb_project, config={"model": model_variant, "epochs": epochs, "batch_size": batch_size}
            )
        except ImportError:
            pass

    # Dataset
    full_ds = KeypointAugDataset(data_zip, augment=True)
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=_collate_fn
    )
    print(f"Dataset: {n_train} train | {n_val} val")

    # Model: ViT backbone + keypoint head
    backbone_name = "vit_large_patch16_224" if "L" in model_variant else "vit_base_patch16_224"
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    in_features = backbone.num_features
    head = KeypointHead(in_features=in_features, num_keypoints=num_keypoints)

    class ViTPoseModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            features = self.backbone(x)  # [B, in_features] (CLS token)
            return self.head(features)

    model = ViTPoseModel().to(dev)

    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled.")

    # Loss functions
    coord_loss_fn = nn.SmoothL1Loss(reduction="none")
    vis_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler() if dev.type == "cuda" else None

    best_val_loss = float("inf")
    best_ckpt = str(save_path / "vitpose_best.pth")

    for epoch in range(1, epochs + 1):
        # Linear warmup
        if epoch <= warmup_epochs:
            lr_scale = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = learning_rate * lr_scale

        model.train()
        train_loss = 0.0
        t0 = time.time()

        for images, kps_gt, vis_gt in train_loader:
            images = images.to(dev)
            kps_gt = kps_gt.to(dev)  # [B, 29, 2] normalized
            vis_gt = vis_gt.to(dev)  # [B, 29] binary

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=dev.type, enabled=(scaler is not None)):
                kps_pred, vis_pred = model(images)  # [B,29,2], [B,29]
                # Coordinate loss: only on visible keypoints
                coord_loss = coord_loss_fn(kps_pred, kps_gt)  # [B, 29, 2]
                vis_mask = vis_gt.unsqueeze(-1)  # [B, 29, 1]
                coord_loss = (coord_loss * vis_mask).sum() / (vis_mask.sum() + 1e-6)
                # Visibility loss
                vis_loss = vis_loss_fn(vis_pred, vis_gt)
                loss = coord_loss + 0.5 * vis_loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        if epoch > warmup_epochs:
            scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, kps_gt, vis_gt in val_loader:
                images = images.to(dev)
                kps_gt = kps_gt.to(dev)
                vis_gt = vis_gt.to(dev)
                with torch.amp.autocast(device_type=dev.type, enabled=(scaler is not None)):
                    kps_pred, vis_pred = model(images)
                    coord_loss = coord_loss_fn(kps_pred, kps_gt)
                    vis_mask = vis_gt.unsqueeze(-1)
                    coord_loss = (coord_loss * vis_mask).sum() / (vis_mask.sum() + 1e-6)
                    vis_loss = vis_loss_fn(vis_pred, vis_gt)
                    val_loss += (coord_loss + 0.5 * vis_loss).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{epochs} | train={avg_train:.4f} val={avg_val:.4f} | {time.time()-t0:.1f}s")

        if run:
            run.log({"train_loss": avg_train, "val_loss": avg_val, "epoch": epoch})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_model = model if not hasattr(model, "_orig_mod") else model._orig_mod
            torch.save({"model_state_dict": save_model.state_dict(), "epoch": epoch, "val_loss": avg_val}, best_ckpt)
            print(f"  Saved → {best_ckpt}")

    if run:
        run.finish()

    print(f"\nTraining complete. Best: {best_ckpt}")
    return best_ckpt
