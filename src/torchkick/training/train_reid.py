"""
DINOv2 + LoRA + ArcFace ReID trainer.

Two training stages:
  supervised  — ArcFace metric learning on labelled player crops (SoccerNet + CVAT).
  ssl         — BYOL tracklet self-supervised learning; same track_id crops = positives.

Example:
    >>> from torchkick.training.train_reid import train_reid
    >>> train_reid(
    ...     data_dir="data/reid/",
    ...     stage="supervised",
    ...     epochs=30,
    ...     save_dir="weights/reid/",
    ... )

CLI:
    $ torchkick train reid --stage supervised --data-dir data/reid/ --epochs 30
    $ torchkick train reid --stage ssl       --data-dir data/reid/ --epochs 20
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


# ---------------------------------------------------------------------------
# BYOL projection / prediction heads (used in SSL stage)
# ---------------------------------------------------------------------------


class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 2048, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PredictionHead(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Supervised stage
# ---------------------------------------------------------------------------


def _train_supervised(
    model: nn.Module,
    arc_head: nn.Module,
    loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    epochs: int,
    save_path: Path,
    run=None,
) -> str:
    best_val_loss = float("inf")
    best_ckpt = str(save_path / "reid_supervised_best.pth")
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        arc_head.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in loader:
            crops, labels = batch["crops"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                # Forward through backbone → [B, 1024]
                features = model(crops)
                # ArcFace logits
                logits = arc_head(features, labels)
                loss = ce_loss_fn(logits, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(arc_head.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(arc_head.parameters()), 1.0)
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        arc_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                crops, labels = batch["crops"].to(device), batch["labels"].to(device)
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    features = model(crops)
                    logits = arc_head(features, labels)
                    val_loss += ce_loss_fn(logits, labels).item()

        avg_train = train_loss / len(loader)
        avg_val = val_loss / len(val_loader)
        print(f"[supervised] Epoch {epoch}/{epochs} | train={avg_train:.4f} val={avg_val:.4f} | {time.time()-t0:.1f}s")

        if run:
            run.log({"supervised/train_loss": avg_train, "supervised/val_loss": avg_val, "epoch": epoch})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            state = {
                "backbone_state_dict": model.state_dict(),
                "arcface_state_dict": arc_head.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val,
            }
            torch.save(state, best_ckpt)
            print(f"  Saved → {best_ckpt}")

    return best_ckpt


# ---------------------------------------------------------------------------
# BYOL SSL stage
# ---------------------------------------------------------------------------


def _byol_loss(online_pred: torch.Tensor, target_proj: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss (BYOL-style)."""
    online_pred = F.normalize(online_pred, dim=-1)
    target_proj = F.normalize(target_proj.detach(), dim=-1)
    return 2.0 - 2.0 * (online_pred * target_proj).sum(dim=-1).mean()


def _update_ema(online: nn.Module, ema: nn.Module, alpha: float = 0.999) -> None:
    """Exponential moving average update of EMA (target) network."""
    for p_o, p_e in zip(online.parameters(), ema.parameters()):
        p_e.data.mul_(alpha).add_(p_o.data, alpha=1.0 - alpha)


def _train_ssl(
    online_backbone: nn.Module,
    ema_backbone: nn.Module,
    proj: _ProjectionHead,
    pred: _PredictionHead,
    ema_proj: _ProjectionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    epochs: int,
    save_path: Path,
    ema_alpha: float = 0.999,
    run=None,
) -> str:
    best_ckpt = str(save_path / "reid_ssl_best.pth")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        online_backbone.train()
        proj.train()
        pred.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in loader:
            # batch["view1"] and batch["view2"] are two augmented views of the same crop
            view1 = batch["view1"].to(device)
            view2 = batch["view2"].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                # Online path: view1 → predict view2's projection
                z1 = proj(online_backbone(view1))
                p1 = pred(z1)
                z2 = proj(online_backbone(view2))
                p2 = pred(z2)
                # Target path (EMA — no grad)
                with torch.no_grad():
                    tz1 = ema_proj(ema_backbone(view1))
                    tz2 = ema_proj(ema_backbone(view2))

                loss = (_byol_loss(p1, tz2) + _byol_loss(p2, tz1)) * 0.5

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(online_backbone.parameters()) + list(proj.parameters()) + list(pred.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(online_backbone.parameters()) + list(proj.parameters()) + list(pred.parameters()), 1.0
                )
                optimizer.step()

            # Update EMA target networks
            _update_ema(online_backbone, ema_backbone, ema_alpha)
            _update_ema(proj, ema_proj, ema_alpha)

            train_loss += loss.item()

        scheduler.step()

        avg_loss = train_loss / len(loader)
        print(f"[ssl] Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | {time.time()-t0:.1f}s")

        if run:
            run.log({"ssl/loss": avg_loss, "epoch": epoch})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"backbone_state_dict": online_backbone.state_dict(), "epoch": epoch}, best_ckpt)
            print(f"  Saved → {best_ckpt}")

    return best_ckpt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_reid(
    data_dir: str,
    stage: str = "supervised",
    num_classes: int = 3,
    lora_rank: int = 16,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    ema_alpha: float = 0.999,
    val_split: float = 0.1,
    compile_model: bool = True,
    use_fsdp: bool = False,
    save_dir: str = "weights/reid/",
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
) -> str:
    """
    Train the DINOv2 + LoRA + ArcFace ReID model.

    Args:
        data_dir: Directory containing ReID crops organised by class sub-folder
            (for ``supervised``) or by track-id sub-folder (for ``ssl``).
        stage: ``"supervised"`` (ArcFace on labelled data) or
               ``"ssl"`` (BYOL tracklet self-supervised learning).
        num_classes: Number of identity classes for ArcFace (default 3: home/away/ref).
        lora_rank: LoRA rank for DINOv2 attention adapters.
        epochs: Training epochs.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        ema_alpha: EMA decay for target network in SSL stage.
        val_split: Validation fraction (supervised stage only).
        compile_model: Apply ``torch.compile`` to the backbone.
        use_fsdp: Wrap backbone in FullyShardedDataParallel for multi-GPU.
        save_dir: Directory for checkpoints.
        device: Device string (auto-detected when None).
        wandb_project: W&B project name.

    Returns:
        Path to best checkpoint.
    """
    try:
        from torchkick.models.reid import DINOv2ReIDBackbone, ArcFaceHead
    except ImportError as e:
        raise ImportError("ReID dependencies missing. Install: pip install torchkick[reid]") from e

    from torchkick.training.data.reid_dataset import ReIDDataset

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    run = None
    if wandb_project:
        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                config={
                    "stage": stage,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lora_rank": lora_rank,
                },
            )
        except ImportError:
            pass

    # -----------------------------------------------------------------------
    # Build backbone
    # -----------------------------------------------------------------------
    backbone = DINOv2ReIDBackbone(lora_rank=lora_rank, freeze_base=True).to(dev)

    if compile_model and hasattr(torch, "compile"):
        backbone = torch.compile(backbone, mode="reduce-overhead")
        print("Backbone compiled.")

    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        backbone = FSDP(backbone)

    scaler = torch.amp.GradScaler() if dev.type == "cuda" else None

    # -----------------------------------------------------------------------
    # Supervised stage
    # -----------------------------------------------------------------------
    if stage == "supervised":
        arc_head = ArcFaceHead(in_features=1024, num_classes=num_classes).to(dev)

        full_ds = ReIDDataset(data_dir, source_type="label_dir", augment=True)
        n_val = max(1, int(len(full_ds) * val_split))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        print(f"ReID supervised: {n_train} train | {n_val} val | {num_classes} classes")

        optimizer = torch.optim.AdamW(
            list(backbone.parameters()) + list(arc_head.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        ckpt = _train_supervised(
            backbone,
            arc_head,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            scaler,
            dev,
            epochs,
            save_path,
            run,
        )

    # -----------------------------------------------------------------------
    # BYOL SSL stage
    # -----------------------------------------------------------------------
    elif stage == "ssl":
        import copy

        # EMA (target) copies — no grad
        ema_backbone = copy.deepcopy(backbone)
        for p in ema_backbone.parameters():
            p.requires_grad_(False)

        in_dim = 1024
        proj = _ProjectionHead(in_dim=in_dim).to(dev)
        pred = _PredictionHead().to(dev)
        ema_proj = copy.deepcopy(proj)
        for p in ema_proj.parameters():
            p.requires_grad_(False)

        dataset = ReIDDataset(data_dir, source_type="tracklet_dir", augment=True, byol_mode=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print(f"ReID SSL (BYOL): {len(dataset)} tracklet crops")

        optimizer = torch.optim.AdamW(
            list(backbone.parameters()) + list(proj.parameters()) + list(pred.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        ckpt = _train_ssl(
            backbone,
            ema_backbone,
            proj,
            pred,
            ema_proj,
            loader,
            optimizer,
            scheduler,
            scaler,
            dev,
            epochs,
            save_path,
            ema_alpha=ema_alpha,
            run=run,
        )

    else:
        raise ValueError(f"Unknown stage '{stage}'. Choose 'supervised' or 'ssl'.")

    if run:
        run.finish()

    print(f"\nTraining complete. Best checkpoint: {ckpt}")
    return ckpt


def train_reid_from_video(
    video_path: str,
    yolo_weights: str,
    save_dir: str = "weights/reid/",
    calibration_duration: float = 120.0,
    n_sample_frames: int = 60,
    min_crops_per_class: int = 20,
    epochs: int = 20,
    batch_size: int = 64,
    lora_rank: int = 16,
    device: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    keep_tmp: bool = False,
) -> str:
    """
    Build a labeled crop dataset from a video and train DINOv2+ArcFace ReID.

    Pipeline:
        1. Randomly sample ``n_sample_frames`` frames from the first
           ``calibration_duration`` seconds.
        2. Run YOLO detection on each frame and extract player crops.
        3. Embed all crops with ``SigLIPTeamEmbedder`` (zero-shot, no labels).
        4. Cluster into k=3 (home, away, referee) via k-means; remap smallest
           cluster → ref (label 2).
        5. Write crops to ``tmp_dir/{0,1,2}/`` (label-dir format).
        6. Fine-tune ``DINOv2ReIDBackbone`` + ``ArcFaceHead`` on the pseudo-labels.

    Args:
        video_path: Input video path.
        yolo_weights: Path to YOLO detection weights (e.g. best.pt).
        save_dir: Directory for the trained checkpoint.
        calibration_duration: Seconds of video to sample from (default 120).
        n_sample_frames: Number of frames to randomly sample for crop extraction.
        min_crops_per_class: Skip training if any cluster has fewer crops than
            this (indicates bad clustering).
        epochs: ArcFace training epochs.
        batch_size: Training batch size.
        lora_rank: LoRA rank for DINOv2 adapters.
        device: Device string (auto-detected when None).
        tmp_dir: Directory for temporary crop storage. Defaults to a system
            temp directory.
        keep_tmp: If True, do not delete the temporary crop directory after
            training (useful for debugging pseudo-labels).

    Returns:
        Path to best checkpoint.
    """
    import random
    import shutil
    import tempfile

    import cv2
    import numpy as np

    from ultralytics import YOLO

    dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # Step 1-2: Sample frames and extract crops
    # -----------------------------------------------------------------------
    print(f"Extracting crops from first {calibration_duration:.0f}s of {video_path} …")

    detector = YOLO(yolo_weights)
    detector.to(torch.device(dev_str))

    all_crops: list = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(calibration_duration * fps))
    sampled_indices = sorted(random.sample(range(max(1, total_frames)), min(n_sample_frames, total_frames)))

    for frame_idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        results = detector(frame_bgr, verbose=False, conf=0.25, classes=[0])
        if results[0].boxes is None or len(results[0].boxes) == 0:
            continue
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0 and (x2 - x1) >= 16 and (y2 - y1) >= 16:
                all_crops.append(cv2.resize(crop, (128, 256)))
    cap.release()

    print(f"  Extracted {len(all_crops)} crops.")
    if len(all_crops) < 30:
        raise RuntimeError(
            f"Only {len(all_crops)} crops found — too few to cluster. "
            "Try a longer calibration_duration or lower conf threshold."
        )

    # -----------------------------------------------------------------------
    # Step 3-4: SigLIP embed + k-means pseudo-labels
    # -----------------------------------------------------------------------
    from torchkick.models.reid import SigLIPTeamEmbedder
    from sklearn.cluster import KMeans

    embedder = SigLIPTeamEmbedder(device=dev_str)
    print("  Embedding crops with SigLIP …")
    embeddings = embedder.embed(all_crops)  # [N, 768]

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    counts = np.bincount(labels, minlength=3)
    ref_cluster = int(np.argmin(counts))
    if ref_cluster != 2:
        swap = {ref_cluster: 2, 2: ref_cluster}
        labels = np.array([swap.get(int(l), int(l)) for l in labels])
        counts[[2, ref_cluster]] = counts[[ref_cluster, 2]]

    print(f"  Pseudo-labels: team0={counts[0]}, team1={counts[1]}, ref={counts[2]}")

    for cls in range(3):
        if counts[cls] < min_crops_per_class:
            raise RuntimeError(
                f"Class {cls} has only {counts[cls]} crops (min={min_crops_per_class}). "
                "Clustering may have failed. Try more sample frames."
            )

    # -----------------------------------------------------------------------
    # Step 5: Write crops to label-dir format
    # -----------------------------------------------------------------------
    tmp_root = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="torchkick_reid_"))
    for cls in range(3):
        (tmp_root / str(cls)).mkdir(parents=True, exist_ok=True)

    for i, (crop, label) in enumerate(zip(all_crops, labels)):
        cv2.imwrite(str(tmp_root / str(int(label)) / f"crop_{i:06d}.jpg"), crop)

    print(f"  Written pseudo-labeled crops to {tmp_root}")

    # -----------------------------------------------------------------------
    # Step 6: Train DINOv2 + ArcFace on pseudo-labels
    # -----------------------------------------------------------------------
    ckpt = train_reid(
        data_dir=str(tmp_root),
        stage="supervised",
        num_classes=3,
        lora_rank=lora_rank,
        epochs=epochs,
        batch_size=batch_size,
        save_dir=save_dir,
        device=dev_str,
    )

    if not keep_tmp and tmp_dir is None:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return ckpt
