"""
Learned 2D→3D pose lifting trained on FIFA Skeletal Tracking 2026 data.

Trains a residual MLP that maps root-relative, scale-normalised 2D keypoints
(COCO-17 + visibility) to root-relative 3D world positions (metres).

The trained lifter can replace the analytical PoseLift3D at inference time
when `use_learned_lifter=True` is passed.  The checkpoint stores the model
weights alongside the training config so normalisation constants are always
applied consistently.

Architecture (PoseLiftMLP):
    Input:  [B, 17, 3]  — (x_rootrel, y_rootrel, visibility), scale-normalised
    Hidden: 3 × residual blocks of (Linear → BN → ReLU → Linear → BN) + skip
    Output: [B, 17, 3]  — root-relative xyz in metres

Loss: MSE on visible joints only.

Data must be downloaded before use:
    huggingface-cli download tijiang13/FIFA-Skeletal-Tracking-Light-2026 \\
        --repo-type dataset --local-dir data/fifa/

Example:
    >>> from torchkick.training.train_pose_lifter import train_pose_lifter
    >>> weights = train_pose_lifter(
    ...     data_dir="data/fifa/",
    ...     epochs=50,
    ...     save_dir="weights/lifter/",
    ... )

CLI:
    $ torchkick train pose-lifter --data data/fifa/ --epochs 50
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


_N_JOINTS = 17


class _ResBlock(nn.Module):
    """Linear residual block with BatchNorm and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class PoseLiftMLP(nn.Module):
    """
    Residual MLP for 2D→3D pose lifting.

    Args:
        n_joints: Number of keypoints (default 17 for COCO).
        hidden_dim: Width of each residual block.
        n_blocks: Number of residual blocks.
        dropout: Dropout probability within each block.
    """

    def __init__(
        self,
        n_joints: int = _N_JOINTS,
        hidden_dim: int = 1024,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = n_joints * 3  # (x, y, vis) per joint
        out_dim = n_joints * 3  # (x, y, z) per joint

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.n_joints = n_joints

    def forward(self, kp2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kp2d: [B, N_joints, 3] root-relative 2D + visibility.

        Returns:
            [B, N_joints, 3] root-relative 3D coordinates.
        """
        B = kp2d.size(0)
        x = kp2d.view(B, -1)  # [B, N*3]
        x = self.input_proj(x)  # [B, hidden]
        x = self.blocks(x)  # [B, hidden]
        x = self.output_proj(x)  # [B, N*3]
        return x.view(B, self.n_joints, 3)


def _collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    kp2d = torch.stack([b[0] for b in batch])  # [B, 17, 3]
    kp3d = torch.stack([b[1] for b in batch])  # [B, 17, 3]
    return kp2d, kp3d


def train_pose_lifter(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    warmup_epochs: int = 3,
    hidden_dim: int = 1024,
    n_blocks: int = 4,
    dropout: float = 0.1,
    save_dir: str = "weights/lifter/",
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> str:
    """
    Train the PoseLiftMLP on FIFA 2D/3D paired pose data.

    Args:
        data_dir: Root of the downloaded FIFA dataset.
        epochs: Training epochs.
        batch_size: Batch size (can be large — no image loading).
        learning_rate: Peak learning rate.
        warmup_epochs: Linear LR warmup epochs.
        hidden_dim: Hidden dimension for each residual block.
        n_blocks: Number of residual blocks.
        dropout: Dropout probability.
        save_dir: Directory to save checkpoint.
        device: "cuda" / "cpu" / "mps". Auto-detects if None.
        wandb_project: W&B project name (optional).
        max_samples: Cap dataset size for smoke tests.

    Returns:
        Path to best model checkpoint.
    """
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

            run = wandb.init(
                project=wandb_project,
                config={"epochs": epochs, "hidden_dim": hidden_dim, "n_blocks": n_blocks},
            )
        except ImportError:
            pass

    # ------------------------------------------------------------------ Data
    # No image loading needed for lifter — set load_images=False
    train_ds = FIFAPoseDataset(data_dir, split="train", mode="lift_3d", load_images=False, max_samples=max_samples)
    val_ds = FIFAPoseDataset(
        data_dir, split="val", mode="lift_3d", load_images=False, max_samples=max_samples and max_samples // 5
    )

    if len(val_ds) == 0:
        from torch.utils.data import random_split

        n_val = max(1, int(len(train_ds) * 0.1))
        train_ds, val_ds = random_split(train_ds, [len(train_ds) - n_val, n_val])
        print(f"Carved val set: {len(train_ds)} train | {len(val_ds)} val")

    n_workers = min(8, torch.get_num_threads())
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
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=_collate_fn
    )
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ----------------------------------------------------------------- Model
    model = PoseLiftMLP(hidden_dim=hidden_dim, n_blocks=n_blocks, dropout=dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PoseLiftMLP: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler() if dev.type == "cuda" else None
    loss_fn = nn.MSELoss(reduction="none")

    best_val_mpjpe = float("inf")
    best_ckpt = str(save_path / "pose_lifter_best.pth")

    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:
            scale = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = learning_rate * scale

        model.train()
        train_loss = 0.0
        t0 = time.time()

        for kp2d, kp3d_gt in train_loader:
            kp2d = kp2d.to(dev)  # [B, 17, 3]  (x, y, vis)
            kp3d_gt = kp3d_gt.to(dev)  # [B, 17, 3]  root-relative metres

            vis = kp2d[..., 2:3]  # [B, 17, 1] — visibility as weight

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=dev.type, enabled=scaler is not None):
                kp3d_pred = model(kp2d)
                per_joint = loss_fn(kp3d_pred, kp3d_gt)  # [B, 17, 3]
                loss = (per_joint * vis).sum() / (vis.sum() * 3 + 1e-6)

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
        val_mpjpe = 0.0
        n_val_joints = 0

        with torch.no_grad():
            for kp2d, kp3d_gt in val_loader:
                kp2d = kp2d.to(dev)
                kp3d_gt = kp3d_gt.to(dev)
                vis = kp2d[..., 2]  # [B, 17]

                with torch.amp.autocast(device_type=dev.type, enabled=scaler is not None):
                    kp3d_pred = model(kp2d)

                # MPJPE (mm): mean per-joint Euclidean error on visible joints
                errors = torch.norm(kp3d_pred - kp3d_gt, dim=-1)  # [B, 17] in metres
                val_mpjpe += (errors * vis).sum().item() * 1000  # → mm
                n_val_joints += vis.sum().item()

        avg_train = train_loss / max(len(train_loader), 1)
        mpjpe_mm = val_mpjpe / max(n_val_joints, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | train_mse={avg_train:.5f} val_mpjpe={mpjpe_mm:.1f}mm | {elapsed:.1f}s")

        if run:
            run.log({"train_loss": avg_train, "val_mpjpe_mm": mpjpe_mm, "epoch": epoch})

        if mpjpe_mm < best_val_mpjpe:
            best_val_mpjpe = mpjpe_mm
            save_obj = model if not hasattr(model, "_orig_mod") else model._orig_mod
            torch.save(
                {
                    "model_state_dict": save_obj.state_dict(),
                    "epoch": epoch,
                    "val_mpjpe_mm": mpjpe_mm,
                    "config": {
                        "hidden_dim": hidden_dim,
                        "n_blocks": n_blocks,
                        "dropout": dropout,
                        "n_joints": _N_JOINTS,
                    },
                },
                best_ckpt,
            )
            print(f"  Saved → {best_ckpt}  (MPJPE={mpjpe_mm:.1f}mm)")

    if run:
        run.finish()

    print(f"\nTraining complete. Best: {best_ckpt}  (MPJPE={best_val_mpjpe:.1f}mm)")
    return best_ckpt
