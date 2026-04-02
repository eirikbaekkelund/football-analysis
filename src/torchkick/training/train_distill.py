"""
Knowledge distillation: DINOv2 ViT-L/14 (teacher) → DINOv2 ViT-S/8 (student).

Distills the trained ReID teacher model into a smaller student for real-time
inference (384-dim embeddings at ~3ms per frame on RTX 3090).

Loss:
    - Cosine similarity loss between teacher and student embeddings
    - KL divergence on softened logits (temperature=4.0)

Example:
    >>> from torchkick.training.train_distill import train_distill
    >>> train_distill(
    ...     teacher_weights="weights/reid/reid_supervised_best.pth",
    ...     data_dir="data/reid/",
    ...     save_dir="weights/reid/",
    ... )

CLI:
    $ torchkick train distill --teacher-weights weights/reid/reid_supervised_best.pth \\
                               --data-dir data/reid/
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


class _StudentBackbone(nn.Module):
    """
    DINOv2-Small (ViT-S/8) with a projection head mapping 384 → 1024 for
    alignment with the teacher's embedding space, plus a 384-dim final output.

    Architecture:
        DINOv2-S/8 (384-dim CLS) → align_proj (384→1024) → out_proj (1024→384)

    The ``align_proj`` is only used during distillation for computing the KL
    alignment loss.  At inference, ``embed()`` returns the 384-dim output.
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            from transformers import Dinov2Model
        except ImportError as e:
            raise ImportError("transformers>=4.35.0 required. Install: pip install torchkick[reid]") from e

        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
        student_dim = self.backbone.config.hidden_size  # 384

        # Projects student 384-dim → teacher 1024-dim (for KL loss)
        self.align_proj = nn.Sequential(
            nn.Linear(student_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        )
        # Final 384-dim output head (used at inference)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(student_dim),
            nn.Linear(student_dim, 384),
        )

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            aligned  — [B, 1024] for KL alignment loss
            embedded — [B, 384]  for cosine loss + inference
        """
        out = self.backbone(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0]  # [B, 384]
        return self.align_proj(cls), self.out_proj(cls)


def _distill_loss(
    teacher_emb: torch.Tensor,
    student_aligned: torch.Tensor,
    student_emb: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Combined cosine + KL distillation loss.

    Args:
        teacher_emb:    [B, 1024] teacher embeddings (no grad).
        student_aligned:[B, 1024] student projection in teacher space.
        student_emb:    [B, 384]  student output embeddings.
        temperature:    Softmax temperature for KL term.

    Returns:
        Scalar loss.
    """
    # Cosine similarity loss (in teacher space)
    cos_loss = 1.0 - F.cosine_similarity(student_aligned, teacher_emb.detach(), dim=-1).mean()

    # KL divergence on softened logits (treat embeddings as unnormalised logits)
    t_soft = F.log_softmax(teacher_emb.detach() / temperature, dim=-1)
    s_soft = F.log_softmax(student_aligned / temperature, dim=-1)
    kl_loss = F.kl_div(s_soft, t_soft.exp(), reduction="batchmean") * (temperature**2)

    return cos_loss + 0.5 * kl_loss


def train_distill(
    teacher_weights: str,
    data_dir: str,
    lora_rank: int = 16,
    temperature: float = 4.0,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    compile_model: bool = True,
    save_dir: str = "weights/reid/",
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
) -> str:
    """
    Distil the teacher DINOv2-Large ReID model into DINOv2-Small.

    Args:
        teacher_weights: Path to ``reid_supervised_best.pth`` from ``train_reid``.
        data_dir:        ReID crops directory (same as used for teacher training).
        lora_rank:       LoRA rank used when building the teacher backbone.
        temperature:     Softmax temperature for KL divergence term.
        epochs:          Training epochs.
        batch_size:      Per-device batch size.
        learning_rate:   Peak learning rate for AdamW.
        weight_decay:    AdamW weight decay.
        val_split:       Validation fraction.
        compile_model:   Apply ``torch.compile`` to both models.
        save_dir:        Checkpoint directory.
        device:          Device string (auto-detected when None).
        wandb_project:   W&B project name.

    Returns:
        Path to best student checkpoint.
    """
    try:
        from torchkick.models.reid import DINOv2ReIDBackbone
    except ImportError as e:
        raise ImportError("Install torchkick[reid] first.") from e

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
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "temperature": temperature,
                },
            )
        except ImportError:
            pass

    # -----------------------------------------------------------------------
    # Teacher (frozen)
    # -----------------------------------------------------------------------
    print("Loading teacher model …")
    teacher = DINOv2ReIDBackbone(lora_rank=lora_rank, freeze_base=True).to(dev)
    ckpt = torch.load(teacher_weights, map_location=dev)
    teacher.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Student
    # -----------------------------------------------------------------------
    print("Building student model (DINOv2-Small) …")
    student = _StudentBackbone().to(dev)

    if compile_model and hasattr(torch, "compile"):
        teacher = torch.compile(teacher, mode="reduce-overhead")
        student = torch.compile(student, mode="reduce-overhead")
        print("Both models compiled.")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    full_ds = ReIDDataset(data_dir, source_type="label_dir", augment=True)
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Dataset: {n_train} train | {n_val} val")

    # -----------------------------------------------------------------------
    # Optimiser
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if dev.type == "cuda" else None

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")
    best_ckpt = str(save_path / "reid_student_best.pth")

    for epoch in range(1, epochs + 1):
        student.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            crops = batch["crops"].to(dev)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                with torch.no_grad():
                    teacher_emb = teacher(crops)  # [B, 1024]
                student_aligned, student_emb = student(crops)
                loss = _distill_loss(teacher_emb, student_aligned, student_emb, temperature)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                crops = batch["crops"].to(dev)
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    teacher_emb = teacher(crops)
                    student_aligned, student_emb = student(crops)
                    val_loss += _distill_loss(teacher_emb, student_aligned, student_emb, temperature).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{epochs} | train={avg_train:.4f} val={avg_val:.4f} | {time.time()-t0:.1f}s")

        if run:
            run.log({"distill/train_loss": avg_train, "distill/val_loss": avg_val, "epoch": epoch})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_model = student if not hasattr(student, "_orig_mod") else student._orig_mod
            torch.save(
                {
                    "student_state_dict": save_model.state_dict(),
                    "epoch": epoch,
                    "val_loss": avg_val,
                    "embed_dim": 384,
                },
                best_ckpt,
            )
            print(f"  Saved → {best_ckpt}")

    if run:
        run.finish()

    print(f"\nDistillation complete. Best: {best_ckpt}")
    return best_ckpt
