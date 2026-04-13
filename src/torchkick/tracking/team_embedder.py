"""
Per-game team embedder.

Uses SigLIPTeamEmbedder to extract pose/occlusion/lighting-invariant
768-dim embeddings from player jersey crops, mean-pools per track,
reduces to 10D with UMAP, then applies K-Means k=3 to discover the
two teams and referee group — no fine-tuning or labels required.

Example:
    >>> from torchkick.models.reid import SigLIPTeamEmbedder
    >>> siglip = SigLIPTeamEmbedder(device="cuda")
    >>> embedder = GameTeamEmbedder(siglip)
    >>> labels = embedder.assign_teams(crops_by_track)
    >>> # labels: Dict[track_id, int]  0=team0, 1=team1, 2=ref
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


# Jersey region: top fraction of the bbox crop (avoids head & legs)
_JERSEY_FRAC = 0.55


def _jersey_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Return top _JERSEY_FRAC of a BGR crop (jersey region only)."""
    h = crop_bgr.shape[0]
    return crop_bgr[: max(10, int(h * _JERSEY_FRAC))]


class GameTeamEmbedder:
    """
    Per-game team embedder using SigLIP + UMAP(10D) + KMeans.

    Workflow:
        1. assign_teams(crops_by_track) — embeds jersey crops via SigLIP,
           mean-pools per track, UMAP 768→10, KMeans k=3 → team labels

    Args:
        siglip_embedder:  ``SigLIPTeamEmbedder`` instance with ``embed()``
                          method returning ``[N, 768]`` L2-normalised embeddings.
        umap_components:  UMAP output dimensionality (default 10).
        n_clusters:       Number of KMeans clusters (default 3: team0, team1, ref).

    Example:
        >>> embedder = GameTeamEmbedder(siglip)
        >>> labels = embedder.assign_teams(crops_by_track)
        >>> # labels: Dict[track_id, int]  0=team0, 1=team1, 2=ref
    """

    def __init__(
        self,
        siglip_embedder,
        umap_components: int = 10,
        n_clusters: int = 3,
        max_ref_tracks: int = 5,
    ) -> None:
        self.siglip = siglip_embedder
        self.umap_components = umap_components
        self.n_clusters = n_clusters
        self.max_ref_tracks = max_ref_tracks

    def fit(self, crops_by_track: Dict[int, List[np.ndarray]]) -> None:
        """No-op — SigLIP is pretrained, no fine-tuning required."""
        pass

    def assign_teams(self, crops_by_track: Dict[int, List[np.ndarray]]) -> Dict[int, int]:
        """
        Embed all tracks, reduce, cluster, and return team labels.

        Labels: 0=team0, 1=team1, 2=ref.
        The two largest KMeans clusters become teams; the smallest becomes ref.

        Args:
            crops_by_track: Dict mapping track_id to list of BGR crop arrays.

        Returns:
            Dict[track_id, int] with team labels.
        """
        try:
            import umap as umap_lib
        except ImportError:
            raise ImportError("umap-learn is required for GameTeamEmbedder. " "Install with: pip install umap-learn")
        from sklearn.cluster import KMeans

        valid = {tid: crops for tid, crops in crops_by_track.items() if crops}
        if not valid:
            return {}

        # Batch embed all jersey crops from all tracks in one call (aggressive batching)
        track_ids = list(valid.keys())
        track_id_to_crops = {}
        all_jersey_crops = []
        crop_to_track_map = []

        for tid in track_ids:
            jersey_crops = [_jersey_crop(c) for c in valid[tid]]
            track_id_to_crops[tid] = len(jersey_crops)
            for crop in jersey_crops:
                all_jersey_crops.append(crop)
                crop_to_track_map.append(tid)

        n_total_crops = len(all_jersey_crops)
        print(f"[TeamEmbedder] Embedding {n_total_crops} crops from {len(track_ids)} tracks via SigLIP …", flush=True)

        # Single GPU call: embed all crops together
        all_embeddings = self.siglip.embed(all_jersey_crops)  # [N_total_crops, 768]

        # Mean-pool per track
        pooled = []
        crop_idx = 0
        for tid in track_ids:
            n_crops = track_id_to_crops[tid]
            track_embeddings = all_embeddings[crop_idx : crop_idx + n_crops]
            pooled.append(track_embeddings.mean(axis=0))
            crop_idx += n_crops

        X = np.stack(pooled)  # [N_tracks, 768]

        # UMAP 768 → 3
        n_neighbors = min(15, len(track_ids) - 1)
        print(
            f"[TeamEmbedder] UMAP 768→{self.umap_components} (n_tracks={len(track_ids)}) …",
            flush=True,
        )
        reducer = umap_lib.UMAP(
            n_components=self.umap_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42,
            verbose=False,
        )
        reduced = reducer.fit_transform(X)  # [N_tracks, 3]

        # KMeans k=3
        km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        raw_labels = km.fit_predict(reduced)

        # Map cluster indices → team labels: largest→0, second→1, smallest→2 (ref)
        # The smallest cluster is only designated ref if it has ≤ max_ref_tracks tracks;
        # an oversized "ref" cluster means K-Means mislabelled players as refs.
        unique_clusters = sorted(
            [(cl, int((raw_labels == cl).sum())) for cl in range(self.n_clusters)],
            key=lambda x: -x[1],
        )
        label_map: Dict[int, int] = {}
        for rank, (cl, count) in enumerate(unique_clusters):
            if rank < 2:
                label_map[cl] = rank  # team0, team1
            else:
                # Only treat this cluster as ref if it's small enough
                label_map[cl] = 2 if count <= self.max_ref_tracks else 0

        result: Dict[int, int] = {tid: label_map[int(raw)] for tid, raw in zip(track_ids, raw_labels)}

        counts = {k: sum(1 for v in result.values() if v == k) for k in (0, 1, 2)}
        print(
            f"[TeamEmbedder] Clusters: team0={counts[0]} team1={counts[1]} ref={counts[2]}",
            flush=True,
        )
        return result
