"""
Per-game self-supervised team embedder.

Uses a frozen DINOv2 backbone (reused from the pitch detector) to extract
CLS-token embeddings from player jersey crops, trains a small MLP projection
head with NT-Xent (SimCLR) loss using temporal consistency as the supervision
signal (same track_id = positive pair), then applies UMAP + HDBSCAN to
discover team clusters without any labels.

Example:
    >>> embedder = GameTeamEmbedder(backbone, device)
    >>> embedder.fit(crops_by_track)           # ~30-60s on GPU
    >>> labels = embedder.assign_teams(crops_by_track)
    >>> # labels: Dict[track_id, int]  0=team0, 1=team1, 2=ref, -1=noise
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ImageNet stats (DINOv2 pretraining)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Jersey region: top fraction of the bbox crop (avoids head & legs)
_JERSEY_FRAC = 0.55
_EMBED_SIZE = 112  # resize target for backbone input


class _ProjectionHead(nn.Module):
    """Small MLP: backbone_dim → hidden → L2-normalised projection."""

    def __init__(self, in_dim: int = 384, hidden_dim: int = 256, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def _nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent loss for N positive pairs (z_i[k], z_j[k])."""
    N = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
    sim = torch.mm(z, z.T) / temperature  # [2N, 2N]
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, targets)


def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Extract jersey region and preprocess to ImageNet-normalised float32 HWC."""
    h = crop_bgr.shape[0]
    jersey = crop_bgr[: max(10, int(h * _JERSEY_FRAC))]
    rgb = cv2.cvtColor(jersey, cv2.COLOR_BGR2RGB)
    rsz = cv2.resize(rgb, (_EMBED_SIZE, _EMBED_SIZE), interpolation=cv2.INTER_LINEAR)
    img = rsz.astype(np.float32) / 255.0
    return (img - _MEAN) / _STD  # HWC float32


class GameTeamEmbedder:
    """
    Per-game self-supervised team embedder.

    Workflow:
        1. fit(crops_by_track)     — pre-embeds crops via DINOv2, trains
                                     projection head with NT-Xent loss
        2. assign_teams(crops_by_track) — embeds + mean-pools per track,
                                          UMAP → HDBSCAN → team labels

    Args:
        backbone:       DINOv2 backbone (torch.nn.Module) with forward_features().
        device:         Torch device.
        embed_dim:      DINOv2 CLS token dimension (384 for ViT-S, 768 for ViT-B).
        proj_dim:       Projection head output dimension.
        temperature:    NT-Xent temperature.
        n_epochs:       Training epochs for projection head.
        batch_size:     Training batch size (number of positive pairs per step).
        lr:             AdamW learning rate.
        umap_components: UMAP output dimensionality before HDBSCAN.
        backbone_batch: Batch size for backbone inference.
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        embed_dim: int = 384,
        proj_dim: int = 64,
        temperature: float = 0.07,
        n_epochs: int = 40,
        batch_size: int = 128,
        lr: float = 1e-3,
        umap_components: int = 12,
        backbone_batch: int = 32,
    ) -> None:
        self.backbone = backbone
        self.device = device
        self.temperature = temperature
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.umap_components = umap_components
        self.backbone_batch = backbone_batch

        self.head = _ProjectionHead(embed_dim, 256, proj_dim).to(device)
        self._optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=0.01)
        self._cached_embs: Optional[Dict[int, np.ndarray]] = None  # {track_id: [N, D]}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _embed_crops(self, crops: List[np.ndarray]) -> np.ndarray:
        """Return DINOv2 CLS tokens [N, embed_dim] for a list of BGR crops."""
        self.backbone.eval()
        tensors = []
        for c in crops:
            hwc = _preprocess_crop(c)
            tensors.append(torch.from_numpy(hwc.transpose(2, 0, 1)))  # CHW
        imgs = torch.stack(tensors).to(self.device)

        all_cls = []
        for i in range(0, len(imgs), self.backbone_batch):
            batch = imgs[i : i + self.backbone_batch]
            feats = self.backbone.forward_features(batch)
            all_cls.append(feats["x_norm_clstoken"].cpu().numpy())
        return np.concatenate(all_cls, axis=0)  # [N, D]

    def _embed_all_tracks(self, crops_by_track: Dict[int, List[np.ndarray]]) -> Dict[int, np.ndarray]:
        """Pre-embed all crops. Returns {track_id: [N_crops, D]}."""
        result = {}
        for tid, crops in crops_by_track.items():
            if crops:
                result[tid] = self._embed_crops(crops)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, crops_by_track: Dict[int, List[np.ndarray]]) -> None:
        """
        Train the projection head using temporal consistency.

        Positive pairs: two random crops from the same track_id.
        The backbone is kept frozen — only the projection head is updated.

        Args:
            crops_by_track: Dict mapping track_id to list of BGR crop arrays.
        """
        valid = {tid: crops for tid, crops in crops_by_track.items() if len(crops) >= 2}
        if len(valid) < 6:
            print("[TeamEmbedder] Too few tracks for contrastive training — skipping head training")
            return

        print(f"[TeamEmbedder] Pre-embedding crops from {len(valid)} tracks …", flush=True)
        track_embs = self._embed_all_tracks(valid)
        self._cached_embs = track_embs

        self.head.train()
        print(f"[TeamEmbedder] Training projection head ({self.n_epochs} epochs) …", flush=True)

        for epoch in range(self.n_epochs):
            tids = list(track_embs.keys())
            np.random.shuffle(tids)

            # Build one positive pair per track
            pairs_i, pairs_j = [], []
            for tid in tids:
                embs = track_embs[tid]
                idx = np.random.choice(len(embs), size=2, replace=len(embs) < 2)
                pairs_i.append(embs[idx[0]])
                pairs_j.append(embs[idx[1]])

            indices = np.random.permutation(len(pairs_i))
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                zi = torch.tensor(np.stack([pairs_i[k] for k in batch]), dtype=torch.float32).to(self.device)
                zj = torch.tensor(np.stack([pairs_j[k] for k in batch]), dtype=torch.float32).to(self.device)

                loss = _nt_xent_loss(self.head(zi), self.head(zj), self.temperature)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                print(
                    f"  [TeamEmbedder] Epoch {epoch+1:3d}/{self.n_epochs}  " f"loss={epoch_loss/max(n_batches,1):.4f}",
                    flush=True,
                )

        self.head.eval()

    def assign_teams(self, crops_by_track: Dict[int, List[np.ndarray]]) -> Dict[int, int]:
        """
        Embed all tracks, cluster, and return team labels.

        Labels: 0=team0, 1=team1, 2=ref, -1=noise/unknown.
        The two largest clusters become teams; any third cluster becomes ref.
        Noise points (-1 from HDBSCAN) are assigned -1.

        Args:
            crops_by_track: Dict mapping track_id to list of BGR crop arrays.

        Returns:
            Dict[track_id, int] with team labels.
        """
        try:
            import umap as umap_lib
            import hdbscan as hdbscan_lib
        except ImportError:
            raise ImportError(
                "umap-learn and hdbscan are required for GameTeamEmbedder. "
                "Install with: pip install umap-learn hdbscan"
            )

        valid = {tid: crops for tid, crops in crops_by_track.items() if crops}
        if not valid:
            return {}

        if self._cached_embs is not None and set(valid.keys()) == set(self._cached_embs.keys()):
            print(f"[TeamEmbedder] Reusing cached embeddings for {len(valid)} tracks …", flush=True)
            track_embs = self._cached_embs
        else:
            print(f"[TeamEmbedder] Embedding {len(valid)} tracks for clustering …", flush=True)
            track_embs = self._embed_all_tracks(valid)

        # Project + mean-pool per track
        track_ids = list(track_embs.keys())
        pooled = []
        with torch.no_grad():
            for tid in track_ids:
                emb_t = torch.tensor(track_embs[tid], dtype=torch.float32).to(self.device)
                proj = self.head(emb_t).cpu().numpy()
                pooled.append(proj.mean(axis=0))

        X = np.stack(pooled)  # [N_tracks, proj_dim]

        # UMAP
        n_neighbors = min(15, len(track_ids) - 1)
        print(
            f"[TeamEmbedder] UMAP {X.shape[1]}→{self.umap_components} " f"(n_tracks={len(track_ids)}) …",
            flush=True,
        )
        reducer = umap_lib.UMAP(
            n_components=self.umap_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42,
            verbose=False,
        )
        reduced = reducer.fit_transform(X)

        # HDBSCAN
        min_cluster = max(3, len(track_ids) // 10)
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster, min_samples=3)
        raw_labels = clusterer.fit_predict(reduced)

        # Map cluster indices → team labels (0, 1, 2, -1)
        unique_clusters = sorted(
            [(cl, int((raw_labels == cl).sum())) for cl in set(raw_labels) if cl >= 0],
            key=lambda x: -x[1],
        )
        label_map: Dict[int, int] = {}
        for rank, (cl, _) in enumerate(unique_clusters):
            if rank == 0:
                label_map[cl] = 0
            elif rank == 1:
                label_map[cl] = 1
            else:
                label_map[cl] = 2  # smaller clusters → ref

        result: Dict[int, int] = {}
        for tid, raw in zip(track_ids, raw_labels):
            result[tid] = label_map.get(int(raw), -1)

        counts = {k: sum(1 for v in result.values() if v == k) for k in (0, 1, 2, -1)}
        print(
            f"[TeamEmbedder] Clusters: team0={counts[0]} team1={counts[1]} " f"ref={counts[2]} noise={counts[-1]}",
            flush=True,
        )
        return result
