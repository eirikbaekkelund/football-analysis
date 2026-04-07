"""
Training script for the DINOv2+heatmap pitch keypoint detector.

Trains a DINOv2PitchModel (ViT-S/14 backbone + CNN decoder) on the
32-keypoint Roboflow pitch schema using weighted MSE heatmap loss plus
a lightweight broadcast-view BCE head.

No backbone freeze is applied — differential learning rates (backbone 10×
lower than decoder) allow the backbone to adapt to soccer-specific features
while preventing catastrophic forgetting of ImageNet pretraining.

Usage:
    torchkick train pitch-heatmap \\
        --data /workspace/weights/soccernet_kp_dataset \\
        --epochs 100 --batch-size 8 --wandb-project torchkick

    # or directly:
    python -c "
    from torchkick.training import train_pitch_heatmap
    train_pitch_heatmap(data_dir='/workspace/weights/soccernet_kp_dataset')
    "
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchkick.training.data.pitch_heatmap_dataset import PitchHeatmapDataset

# Zone index ranges for per-zone PCK (Roboflow 32-kp schema)
_ZONE_LEFT = list(range(0, 13))  # left boundary + left box
_ZONE_CENTER = list(range(13, 17)) + [30, 31]  # centre line + circle
_ZONE_RIGHT = list(range(17, 30))  # right box + right boundary


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    visibility: torch.Tensor,
    fg_weight: float = 100.0,
) -> torch.Tensor:
    """
    Weighted MSE on sigmoid-activated heatmaps.

    Foreground pixels (target > 0.01) receive fg_weight× higher loss weight
    so the optimiser cannot minimise loss by predicting all-zeros.
    Invisible keypoints (visibility == 0) are fully masked out.

    Args:
        logits:     [B, 32, H, W]  raw model output (no sigmoid)
        targets:    [B, 32, H, W]  Gaussian heatmap targets in [0, 1]
        visibility: [B, 32]        binary keypoint visibility mask
        fg_weight:  foreground pixel loss multiplier (default 100)

    Returns:
        Scalar loss.
    """
    pred = torch.sigmoid(logits)
    weight = torch.ones_like(targets)
    weight[targets > 0.01] = fg_weight

    # mask out invisible keypoints entirely
    vis_mask = visibility.view(visibility.shape[0], visibility.shape[1], 1, 1).expand_as(targets)
    weight = weight * vis_mask

    denom = weight.sum().clamp(min=1.0)
    return (weight * (pred - targets) ** 2).sum() / denom


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _decode_heatmaps(
    logits: torch.Tensor,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """
    Decode heatmap logits to pixel coordinates.

    Returns:
        kps: [B, 32, 2] pixel coordinates in [0, orig_w] × [0, orig_h]
    """
    hm_size = logits.shape[-1]
    heatmaps = torch.sigmoid(logits).float().cpu().numpy()  # [B, 32, H, W]
    B = heatmaps.shape[0]
    kps = np.zeros((B, 32, 2), dtype=np.float32)

    for b in range(B):
        for k in range(32):
            hm = heatmaps[b, k]
            hy, hx = np.unravel_index(np.argmax(hm), hm.shape)
            kps[b, k, 0] = (float(hx) + 0.5) / hm_size * orig_w
            kps[b, k, 1] = (float(hy) + 0.5) / hm_size * orig_h

    return kps


def _compute_pck(
    pred_kps: np.ndarray,
    gt_kps: np.ndarray,
    visibility: np.ndarray,
    threshold_px: float,
) -> dict:
    """
    Compute PCK (percentage correct keypoints) within threshold_px.

    Args:
        pred_kps:   [N, 32, 2]
        gt_kps:     [N, 32, 2]  (in same pixel space)
        visibility: [N, 32]
        threshold_px: distance threshold in pixels

    Returns:
        dict with keys: all, left, center, right
    """
    dist = np.linalg.norm(pred_kps - gt_kps, axis=-1)  # [N, 32]
    vis = visibility > 0.5

    def _zone_pck(indices):
        v = vis[:, indices]
        d = dist[:, indices]
        n = v.sum()
        if n == 0:
            return float("nan")
        return float(((d < threshold_px) & v).sum() / n)

    return {
        "all": _zone_pck(list(range(32))),
        "left": _zone_pck(_ZONE_LEFT),
        "center": _zone_pck(_ZONE_CENTER),
        "right": _zone_pck(_ZONE_RIGHT),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_pitch_heatmap(
    data_dir: Optional[str] = None,
    soccernet_calibration_dir: Optional[str] = None,
    base_model: Optional[str] = None,
    backbone_variant: str = "dinov2_vits14",
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    backbone_lr_scale: float = 0.1,
    warmup_epochs: int = 5,
    fg_weight: float = 100.0,
    min_keypoints: int = 6,
    sigma: float = 2.5,
    imgsz: int = 560,
    compile_model: bool = False,
    save_dir: str = "weights/pitch_heatmap/",
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
    num_workers: int = 4,
) -> str:
    """
    Train DINOv2+heatmap pitch keypoint detector.

    Args:
        data_dir:                  Root of pre-converted dataset (images/ + labels/ subdirs).
                                   Required unless soccernet_calibration_dir is given.
        soccernet_calibration_dir: Path to SoccerNet calibration directory containing
                                   train.zip / valid.zip.  Auto-converts to label format
                                   and sets data_dir automatically.
        base_model:                Optional path to existing checkpoint for fine-tuning.
        backbone_variant:   "dinov2_vits14" or "dinov2_vitb14".
        epochs:             Total training epochs.
        batch_size:         Per-GPU batch size.
        learning_rate:      Decoder / head learning rate.
        backbone_lr_scale:  Backbone LR = learning_rate × backbone_lr_scale.
        warmup_epochs:      Linear LR warmup epochs.
        fg_weight:          Foreground heatmap pixel loss weight.
        min_keypoints:      Minimum visible keypoints to include a sample.
        sigma:              Gaussian sigma in heatmap pixels.
        imgsz:              Model input size (must be multiple of 14).
        compile_model:      Apply torch.compile (speeds up by ~15-20%).
        save_dir:           Directory to save checkpoints.
        device:             Torch device ("cuda", "cuda:0", "cpu", …).
        wandb_project:      If set, log metrics to this WandB project.
        num_workers:        DataLoader worker count.

    Returns:
        Path to best checkpoint (.pt).
    """
    from torchkick.models.pitch import DINOv2PitchModel

    # --- auto-convert SoccerNet calibration data if provided ---
    if soccernet_calibration_dir is not None:
        from torchkick.training.train_yolo_detection import build_soccernet_keypoint_dataset

        converted_dir = str(Path(save_dir).parent / "soccernet_kp_dataset")
        print(f"Converting SoccerNet calibration data → {converted_dir}")
        build_soccernet_keypoint_dataset(
            calibration_dir=soccernet_calibration_dir,
            output_dir=converted_dir,
            min_keypoints=min_keypoints,
        )
        data_dir = converted_dir

    if data_dir is None:
        raise ValueError("Provide data_dir or soccernet_calibration_dir.")

    # --- device ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # --- datasets ---
    train_ds = PitchHeatmapDataset(
        data_dir,
        split="train",
        augment=True,
        sigma=sigma,
        min_keypoints=min_keypoints,
        input_size=imgsz,
    )
    val_ds = PitchHeatmapDataset(
        data_dir,
        split="valid",
        augment=False,
        sigma=sigma,
        min_keypoints=min_keypoints,
        input_size=imgsz,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- model ---
    model = DINOv2PitchModel(backbone_name=backbone_variant, num_keypoints=32)

    if base_model is not None:
        ckpt = torch.load(base_model, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"  Resumed from {base_model}")

    model = model.to(dev)

    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")

    # --- optimizer: differential LR (backbone gets 10× lower) ---
    backbone_params = (
        list(model.backbone.parameters()) if not compile_model else list(model._orig_mod.backbone.parameters())
    )
    other_params = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("backbone") and not n.startswith("_orig_mod.backbone")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": learning_rate * backbone_lr_scale},
            {"params": other_params, "lr": learning_rate},
        ],
        weight_decay=0.01,
    )

    # --- scheduler: linear warmup + cosine annealing ---
    def _warmup_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_lambda)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup_epochs, 1),
        eta_min=1e-6,
    )

    # --- WandB ---
    _wandb = None
    if wandb_project:
        try:
            import wandb

            _wandb = wandb
            wandb.init(
                project=wandb_project,
                config=dict(
                    backbone=backbone_variant,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=learning_rate,
                    backbone_lr_scale=backbone_lr_scale,
                    fg_weight=fg_weight,
                    sigma=sigma,
                    imgsz=imgsz,
                    min_keypoints=min_keypoints,
                ),
            )
            wandb.watch(model, log_freq=100)
        except ImportError:
            print("wandb not installed — skipping logging")
            _wandb = None

    # --- save dir ---
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_ckpt = str(save_path / "best.pt")
    last_ckpt = str(save_path / "last.pt")

    best_pck = -1.0
    scaler = torch.amp.GradScaler("cuda", enabled="cuda" in device)

    for epoch in range(epochs):
        t0 = time.perf_counter()

        # ---- train ----
        model.train()
        train_loss = 0.0

        for imgs, hm_targets, vis, pitch_labels in train_loader:
            imgs = imgs.to(dev)
            hm_targets = hm_targets.to(dev)
            vis = vis.to(dev)
            pitch_labels = pitch_labels.to(dev)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled="cuda" in device):
                logits, pitch_logit = model(imgs)
                loss_hm = _heatmap_loss(logits, hm_targets, vis, fg_weight)
                loss_pitch = F.binary_cross_entropy_with_logits(pitch_logit, pitch_labels)
                loss = loss_hm + 0.1 * loss_pitch

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss_hm.item()

        train_loss /= max(len(train_loader), 1)

        # ---- lr scheduler step ----
        if epoch < warmup_epochs:
            warmup_sched.step()
        else:
            cosine_sched.step()

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        all_pred = []
        all_gt = []
        all_vis = []
        n_visible = []

        with torch.no_grad():
            for imgs, hm_targets, vis, _ in val_loader:
                imgs = imgs.to(dev)
                hm_targets = hm_targets.to(dev)
                vis_dev = vis.to(dev)

                with torch.amp.autocast("cuda", enabled="cuda" in device):
                    logits, _ = model(imgs)
                    loss_hm = _heatmap_loss(logits, hm_targets, vis_dev, fg_weight)

                val_loss += loss_hm.item()

                # decode for PCK (use imgsz as reference pixel space)
                pred_kps = _decode_heatmaps(logits, imgsz, imgsz)
                gt_kps = _decode_gt(hm_targets, imgsz)

                all_pred.append(pred_kps)
                all_gt.append(gt_kps)
                all_vis.append(vis.numpy())
                n_visible.append(vis.sum(dim=1).float().mean().item())

        val_loss /= max(len(val_loader), 1)

        pred_arr = np.concatenate(all_pred, axis=0)
        gt_arr = np.concatenate(all_gt, axis=0)
        vis_arr = np.concatenate(all_vis, axis=0)
        avg_vis = float(np.mean(n_visible))

        pck10 = _compute_pck(pred_arr, gt_arr, vis_arr, threshold_px=10.0)
        pck20 = _compute_pck(pred_arr, gt_arr, vis_arr, threshold_px=20.0)

        elapsed = time.perf_counter() - t0
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_dec = optimizer.param_groups[1]["lr"]

        print(
            f"Ep {epoch+1:3d}/{epochs} | "
            f"train {train_loss:.5f} | val {val_loss:.5f} | "
            f"PCK@10={pck10['all']:.3f} (L={pck10['left']:.3f} "
            f"C={pck10['center']:.3f} R={pck10['right']:.3f}) | "
            f"vis={avg_vis:.1f} | {elapsed:.1f}s"
        )

        # ---- WandB logging ----
        if _wandb is not None:
            log = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/pck_10px": pck10["all"],
                "val/pck_20px": pck20["all"],
                "val/pck_left_10px": pck10["left"],
                "val/pck_center_10px": pck10["center"],
                "val/pck_right_10px": pck10["right"],
                "val/n_visible_avg": avg_vis,
                "lr/backbone": lr_bb,
                "lr/decoder": lr_dec,
                "epoch": epoch + 1,
            }
            if (epoch + 1) % 10 == 0:
                log["val/heatmap_viz"] = _make_wandb_heatmap_images(_wandb, imgs[:4], logits[:4], hm_targets[:4])
            _wandb.log(log)

        # ---- checkpoint ----
        _save_ckpt(model, optimizer, epoch + 1, pck10["all"], val_loss, backbone_variant, last_ckpt)

        if pck10["all"] > best_pck or math.isnan(best_pck):
            best_pck = pck10["all"]
            _save_ckpt(model, optimizer, epoch + 1, pck10["all"], val_loss, backbone_variant, best_ckpt)
            print(f"  ✓ New best PCK@10: {best_pck:.4f} → {best_ckpt}")

    if _wandb is not None:
        _wandb.finish()

    print(f"\nTraining complete. Best PCK@10={best_pck:.4f}")
    print(f"Best checkpoint: {best_ckpt}")
    return best_ckpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_gt(hm_targets: torch.Tensor, imgsz: int) -> np.ndarray:
    """
    Decode GT heatmap targets to pixel coordinates for PCK evaluation.

    Args:
        hm_targets: [B, 32, H, W]  Gaussian targets in [0,1]
        imgsz:      reference pixel space size

    Returns:
        [B, 32, 2] pixel coordinates
    """
    hm_size = hm_targets.shape[-1]
    hm = hm_targets.float().cpu().numpy()
    B = hm.shape[0]
    kps = np.zeros((B, 32, 2), dtype=np.float32)

    for b in range(B):
        for k in range(32):
            h = hm[b, k]
            if h.max() < 1e-4:
                continue
            hy, hx = np.unravel_index(np.argmax(h), h.shape)
            kps[b, k, 0] = (float(hx) + 0.5) / hm_size * imgsz
            kps[b, k, 1] = (float(hy) + 0.5) / hm_size * imgsz

    return kps


def _save_ckpt(
    model,
    optimizer,
    epoch: int,
    val_pck: float,
    val_loss: float,
    backbone_variant: str,
    path: str,
) -> None:
    """Save model checkpoint with config dict for inference reconstruction."""
    # unwrap torch.compile if present
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_pck_10px": val_pck,
            "val_loss": val_loss,
            "config": {
                "backbone_variant": backbone_variant,
                "num_keypoints": 32,
                "input_size": 560,
                "heatmap_size": 320,
            },
        },
        path,
    )


def _make_wandb_heatmap_images(wandb, imgs, logits, targets):
    """Log a grid of GT vs predicted heatmap overlays for 4 samples."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        imgs_np = imgs[:4].float().cpu().numpy()
        pred_np = torch.sigmoid(logits[:4]).float().cpu().numpy()
        gt_np = targets[:4].float().cpu().numpy()

        # denormalise for display
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        imgs_np = (imgs_np * std + mean).clip(0, 1)

        figs = []
        for b in range(min(4, imgs_np.shape[0])):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            rgb = imgs_np[b].transpose(1, 2, 0)
            axes[0].imshow(rgb)
            axes[0].set_title("Image")
            axes[0].axis("off")
            axes[1].imshow(rgb)
            axes[1].imshow(gt_np[b].sum(0), alpha=0.6, cmap="hot")
            axes[1].set_title("GT heatmaps")
            axes[1].axis("off")
            axes[2].imshow(rgb)
            axes[2].imshow(pred_np[b].sum(0), alpha=0.6, cmap="hot")
            axes[2].set_title("Pred heatmaps")
            axes[2].axis("off")
            fig.tight_layout()
            figs.append(wandb.Image(fig))
            plt.close(fig)
        return figs
    except Exception:
        return []
