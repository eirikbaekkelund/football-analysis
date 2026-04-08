"""
DINOv2 + LoRA + ArcFace player re-identification model.

Architecture:
    DINOv2 ViT-L/14 (frozen backbone)
        → LoRA adapter layers (rank 16-32, injected into attention)
        → Multi-scale feature aggregation (CLS token + mean patch tokens)
        → ArcFace head (s=64, m=0.5) — 3-class: team0, team1, other (gk, ref)
        → SpectralEmbedding — for unsupervised team clustering

At training time: use ViT-L/14 (1024-dim) with LoRA + ArcFace.
At inference time: use distilled ViT-S/8 (384-dim) via DINOv2ReIDEmbedder.

Augmentation constraint: WEAK color jitter during training — jersey color
is the primary discriminative signal for team classification.

Example:
    >>> from torchkick.models.reid import DINOv2ReIDEmbedder
    >>>
    >>> embedder = DINOv2ReIDEmbedder("weights/reid/student.pt", device="cuda")
    >>> crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in player_boxes]
    >>> embeddings = embedder.embed(crops)      # [N, 384] float32
    >>> teams = embedder.classify_team(crops)   # [N] int, 0=home 1=away 2=ref
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 backbone with PEFT LoRA adapters
# ─────────────────────────────────────────────────────────────────────────────


class DINOv2ReIDBackbone(nn.Module):
    """
    DINOv2 ViT-L/14 backbone with PEFT LoRA on attention Q/V projections.

    Uses HuggingFace PEFT to correctly inject trainable LoRA deltas into every
    attention query and value projection.  The base weights are frozen by PEFT;
    only the rank-r A/B matrices are updated during training.

    Output: CLS token + mean(patch tokens) → [B, hidden_size].

    Args:
        model_name: HuggingFace model ID (default: facebook/dinov2-large).
        lora_rank: LoRA rank (16 = compact, 32 = expressive).
        freeze_base: Freeze all non-LoRA parameters (handled by PEFT internally).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        lora_rank: int = 16,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()

        try:
            from transformers import Dinov2Model
            from peft import get_peft_model, LoraConfig
        except ImportError as e:
            raise ImportError(
                "transformers>=4.35.0 and peft>=0.13.0 required. " "Install: pip install torchkick[reid]"
            ) from e

        base = Dinov2Model.from_pretrained(model_name)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        self.backbone = get_peft_model(base, lora_config)
        # freeze_base is handled by PEFT (base weights are frozen by get_peft_model)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W] normalized input images.

        Returns:
            [B, hidden_size] — CLS token + mean patch tokens.
        """
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=False)
        cls_token = outputs.last_hidden_state[:, 0, :]
        patch_mean = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        return cls_token + patch_mean


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace head (3-class: home, away, referee)
# ─────────────────────────────────────────────────────────────────────────────


class ArcFaceHead(nn.Module):
    """
    ArcFace classification head for metric learning.

    Adds an angular margin to the cosine similarity before softmax,
    encouraging tight intra-class clusters and large inter-class margins.

    Args:
        in_features: Input embedding dimension (1024 for ViT-L).
        num_classes: Number of classes (3: home, away, ref).
        scale: Feature scale factor s (default 64).
        margin: Angular margin m in radians (default 0.5 ≈ 28.6°).
    """

    def __init__(
        self,
        in_features: int = 1024,
        num_classes: int = 3,
        scale: float = 64.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, D] L2-normalized embeddings.
            labels: [B] class indices. If None, returns raw cosine logits.

        Returns:
            [B, num_classes] logits.
        """
        normed_feat = F.normalize(features, p=2, dim=1)
        normed_weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(normed_feat, normed_weight)  # [B, C]

        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.scale


# ─────────────────────────────────────────────────────────────────────────────
# Spectral embedding for unsupervised team clustering
# ─────────────────────────────────────────────────────────────────────────────


class SpectralEmbedding(nn.Module):
    """
    Projects embeddings to a lower-dimensional spectral space.

    Used for unsupervised team clustering (no pre-labeled team colors).
    Spectral embeddings separate teams better than raw cosine similarity
    when appearance distributions overlap.

    Args:
        in_dim: Input embedding dimension.
        out_dim: Output spectral dimension (128 recommended).
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Inference-time embedder
# ─────────────────────────────────────────────────────────────────────────────

# Standard crop size for ReID (width, height)
_CROP_SIZE = (128, 256)

# ImageNet normalization (same as DINOv2 preprocessing)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_crops(crops: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """Preprocess BGR crop list to a batch tensor."""
    tensors = []
    for crop in crops:
        if crop is None or crop.size == 0:
            crop = np.zeros((_CROP_SIZE[1], _CROP_SIZE[0], 3), dtype=np.uint8)
        resized = cv2.resize(crop, _CROP_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - _MEAN) / _STD
        tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1)))
    return torch.stack(tensors).to(device)


class DINOv2ReIDEmbedder:
    """
    Inference-time player re-identification embedder.

    Uses the distilled ViT-S/8 model for speed (384-dim, ~8ms/batch on A100).
    Falls back to ViT-L/14 if no distilled weights are provided.

    Team classification:
        - If model has ArcFace head: use `classify_team()` (supervised).
        - Without labeled data: use `cluster_teams()` (unsupervised spectral).

    Args:
        weights_path: Path to distilled student checkpoint.
            If None, uses base DINOv2-small without fine-tuning.
        model_name: HuggingFace model ID for the student.
        device: Torch device string.
        batch_size: Crops per forward pass.
        use_fp16: Use FP16 on GPU.
        num_classes: Number of team classes (3: home, away, ref).

    Example:
        >>> embedder = DINOv2ReIDEmbedder("weights/reid/student.pt")
        >>> embeddings = embedder.embed(player_crops)  # [N, 384]
        >>> teams = embedder.classify_team(player_crops)  # [N]
    """

    TEAM_NAMES = {0: "home", 1: "away", 2: "referee"}

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        model_name: str = "facebook/dinov2-small",
        device: str = "cuda",
        batch_size: int = 32,
        use_fp16: bool = True,
        num_classes: int = 3,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and "cuda" in device
        self.num_classes = num_classes

        self._backbone: Optional[nn.Module] = None
        self._classifier: Optional[ArcFaceHead] = None
        self._spectral: Optional[SpectralEmbedding] = None
        self._embed_dim: int = 384  # DINOv2-small

        self._load_model(model_name, weights_path)

    def _load_model(self, model_name: str, weights_path: Optional[Union[str, Path]]) -> None:
        try:
            from transformers import Dinov2Model
        except ImportError:
            raise ImportError("transformers>=4.35.0 required. Install: pip install torchkick[reid]")

        backbone = Dinov2Model.from_pretrained(model_name)
        self._embed_dim = backbone.config.hidden_size

        # Simple wrapper that returns CLS + patch mean
        class _Backbone(nn.Module):
            def __init__(self, base) -> None:
                super().__init__()
                self.base = base
                self.classifier = ArcFaceHead(base.config.hidden_size, num_classes=3)
                self.spectral = SpectralEmbedding(base.config.hidden_size, out_dim=128)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.base(pixel_values=x)
                cls = out.last_hidden_state[:, 0, :]
                patch_mean = out.last_hidden_state[:, 1:, :].mean(1)
                return cls + patch_mean

        self._model_wrapper = _Backbone(backbone)

        if weights_path is not None and Path(str(weights_path)).exists():
            checkpoint = torch.load(str(weights_path), map_location=self.device, weights_only=True)
            state = checkpoint.get("model_state_dict", checkpoint)
            self._model_wrapper.load_state_dict(state, strict=False)

        self._backbone = self._model_wrapper.base
        self._classifier = self._model_wrapper.classifier
        self._spectral = self._model_wrapper.spectral

        self._model_wrapper.to(self.device)
        self._model_wrapper.eval()

        if self.use_fp16:
            self._model_wrapper = self._model_wrapper.half()

    @torch.inference_mode()
    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract ReID embeddings from BGR player crops.

        Args:
            crops: List of BGR player crop images.

        Returns:
            np.ndarray [N, embed_dim] float32.
        """
        if not crops:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i : i + self.batch_size]
            tensors = _preprocess_crops(batch, self.device)
            if self.use_fp16:
                tensors = tensors.half()
            features = self._model_wrapper(tensors)
            all_embeddings.append(features.float().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    @torch.inference_mode()
    def classify_team(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Classify each crop into home (0), away (1), or referee (2).

        Uses the ArcFace classification head (requires fine-tuned weights).

        Args:
            crops: List of BGR player crop images.

        Returns:
            np.ndarray [N] int with values in {0, 1, 2}.
        """
        if not crops:
            return np.zeros(0, dtype=np.int64)

        all_labels = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i : i + self.batch_size]
            tensors = _preprocess_crops(batch, self.device)
            if self.use_fp16:
                tensors = tensors.half()
            features = self._model_wrapper(tensors)
            if self._classifier is not None:
                logits = self._classifier(features.float(), labels=None)
                labels = logits.argmax(dim=1)
            else:
                labels = torch.zeros(len(batch), dtype=torch.long)
            all_labels.append(labels.cpu().numpy())

        return np.concatenate(all_labels, axis=0)

    @torch.inference_mode()
    def classify_from_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the classifier head on pre-computed embeddings.

        Skips the backbone forward pass — use when embeddings are already
        stored in ``TrackObservation.reid_embedding``.

        Args:
            embeddings: [N, embed_dim] float32 array.

        Returns:
            labels: [N] int array with values in {0, 1, 2}.
            probs:  [N, num_classes] float32 softmax probabilities.
        """
        if embeddings is None or len(embeddings) == 0:
            empty = np.zeros((0, self.num_classes), dtype=np.float32)
            return np.zeros(0, dtype=np.int64), empty

        if self._classifier is None:
            labels = np.zeros(len(embeddings), dtype=np.int64)
            probs = np.zeros((len(embeddings), self.num_classes), dtype=np.float32)
            probs[:, 0] = 1.0
            return labels, probs

        all_labels = []
        all_probs = []
        t = torch.from_numpy(embeddings).to(self.device)
        if self.use_fp16:
            t = t.half()

        for i in range(0, len(t), self.batch_size):
            batch = t[i : i + self.batch_size]
            logits = self._classifier(batch.float(), labels=None)
            p = torch.softmax(logits, dim=1)
            all_labels.append(p.argmax(dim=1).cpu().numpy())
            all_probs.append(p.cpu().numpy())

        return (
            np.concatenate(all_labels).astype(np.int64),
            np.concatenate(all_probs).astype(np.float32),
        )

    def cluster_teams(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Unsupervised team assignment via spectral clustering.

        Uses sklearn SpectralClustering on the spectral embedding space.
        Appropriate for first-run scenarios without pre-labeled team data.

        Args:
            embeddings: [N, D] float32 embedding array.

        Returns:
            np.ndarray [N] int cluster assignments (not calibrated to team IDs).
        """
        if len(embeddings) < 3:
            return np.zeros(len(embeddings), dtype=np.int64)

        try:
            from sklearn.cluster import SpectralClustering

            # Project to spectral space first
            if self._spectral is not None:
                with torch.inference_mode():
                    t = torch.from_numpy(embeddings).to(self.device)
                    if self.use_fp16:
                        t = t.half()
                    projected = self._spectral(t.float()).cpu().numpy()
            else:
                projected = embeddings

            clustering = SpectralClustering(
                n_clusters=self.num_classes,
                affinity="nearest_neighbors",
                random_state=42,
            )
            return clustering.fit_predict(projected).astype(np.int64)
        except Exception:
            # Fallback: k-means
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=self.num_classes, random_state=42, n_init=10)
            return km.fit_predict(embeddings).astype(np.int64)


class SigLIPTeamEmbedder:
    """
    Zero-shot team clustering via SigLIP image embeddings + KMeans.

    Requires no labeled training data. Embeds player crops with
    ``google/siglip-base-patch16-224`` (available via ``transformers>=4.38``,
    already in the reid dependency group) and clusters into 3 groups
    (home, away, referee) via KMeans.

    The cluster assigned to the smallest group (typically 1-3 referees)
    is remapped to label 2, matching the ``{0: home, 1: away, 2: ref}``
    convention used by ``DINOv2ReIDEmbedder``.

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device string.
        batch_size: Crops per forward pass.

    Example:
        >>> embedder = SigLIPTeamEmbedder(device="cuda")
        >>> embs = embedder.embed(crops)          # [N, 768]
        >>> labels = embedder.cluster_teams(embs) # [N] in {0,1,2}
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        try:
            from transformers import AutoImageProcessor, SiglipVisionModel

            self._processor = AutoImageProcessor.from_pretrained(model_name)
            self._model = SiglipVisionModel.from_pretrained(model_name).to(self.device).eval()
        except ImportError as e:
            raise ImportError(
                f"SigLIPTeamEmbedder dependency missing: {e}. "
                "Install with: pip install torchkick[reid] sentencepiece"
            ) from e

    @torch.inference_mode()
    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Embed player crops with SigLIP.

        Args:
            crops: List of BGR player crop images.

        Returns:
            np.ndarray [N, 768] L2-normalized float32 embeddings.
        """
        if not crops:
            return np.zeros((0, 768), dtype=np.float32)

        from PIL import Image

        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(crops), self.batch_size):
            batch_crops = crops[i : i + self.batch_size]
            pil_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in batch_crops]
            inputs = self._processor(images=pil_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self._model(**inputs)
            # SigLIP vision model returns BaseModelOutputWithPooling; use pooler_output
            image_features = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]
            image_features = F.normalize(image_features.float(), dim=-1)
            all_embeddings.append(image_features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def cluster_teams(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings into home (0), away (1), referee (2) labels.

        Args:
            embeddings: [N, D] L2-normalized float32 embeddings.

        Returns:
            np.ndarray [N] int labels in {0, 1, 2}.
        """
        if len(embeddings) < 3:
            return np.zeros(len(embeddings), dtype=np.int64)

        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        raw_labels = km.fit_predict(embeddings)

        # Remap smallest cluster → label 2 (referee convention)
        counts = np.bincount(raw_labels, minlength=3)
        ref_cluster = int(np.argmin(counts))
        # Swap ref_cluster with cluster 2 if needed
        if ref_cluster != 2:
            swap_map = {ref_cluster: 2, 2: ref_cluster}
            raw_labels = np.array([swap_map.get(l, l) for l in raw_labels], dtype=np.int64)

        return raw_labels.astype(np.int64)


__all__ = [
    "DINOv2ReIDBackbone",
    "ArcFaceHead",
    "SpectralEmbedding",
    "DINOv2ReIDEmbedder",
    "SigLIPTeamEmbedder",
]
