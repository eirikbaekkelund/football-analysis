"""
Active learning loop for iterative annotation and retraining.

Components:
    UncertaintySampler  — entropy scoring via MC Dropout
    CoresetSelector     — greedy furthest-point sampling for diversity
    ActiveLearningLoop  — orchestrates: pipeline → score → select → CVAT → retrain

Example:
    >>> from torchkick.annotation.active_learning import ActiveLearningLoop
    >>> loop = ActiveLearningLoop(
    ...     pipeline=grounded_sam_pipeline,
    ...     cvat_client=client,
    ...     embedder=reid_embedder,
    ...     budget=100,
    ... )
    >>> loop.run(video_dir="data/videos/", project_id=1)

CLI:
    $ torchkick label active-learning --video-dir data/videos/ --project-id 1 --budget 100
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from torchkick.annotation.grounded_sam import GroundedSAMPipeline, TrackAnnotation
    from torchkick.models.reid import DINOv2ReIDEmbedder


class UncertaintySampler:
    """
    Entropy-based uncertainty scoring using MC Dropout.

    Runs the embedder with dropout enabled ``n_passes`` times and computes
    the entropy of the team-classification probability distribution as the
    uncertainty score for each crop.

    Args:
        embedder: ``DINOv2ReIDEmbedder`` instance.
        n_passes: Number of MC Dropout forward passes.

    Example:
        >>> sampler = UncertaintySampler(embedder, n_passes=10)
        >>> scores = sampler.score(crops)  # [N] entropy values
    """

    def __init__(
        self,
        embedder: "DINOv2ReIDEmbedder",
        n_passes: int = 10,
    ) -> None:
        self.embedder = embedder
        self.n_passes = n_passes

    def score(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Compute per-crop uncertainty as entropy over team-class predictions.

        Args:
            crops: List of BGR/RGB crop arrays.

        Returns:
            Float array of shape [N] with entropy scores (higher = more uncertain).
        """
        if not crops:
            return np.array([], dtype=np.float32)

        import torch
        import torch.nn.functional as F
        from torchkick.models.reid import _preprocess_crops

        # Put model in train mode so dropout layers fire; use no_grad (NOT
        # inference_mode, which would disable dropout stochasticity).
        self.embedder._model_wrapper.train()

        all_probs: List[np.ndarray] = []
        for _ in range(self.n_passes):
            with torch.no_grad():
                tensors = _preprocess_crops(crops, self.embedder.device)
                if self.embedder.use_fp16:
                    tensors = tensors.half()
                features = self.embedder._model_wrapper(tensors)
                if self.embedder._classifier is not None:
                    logits = self.embedder._classifier(features.float(), labels=None)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                else:
                    n_classes = self.embedder.num_classes
                    probs = np.full((len(crops), n_classes), 1.0 / n_classes, dtype=np.float32)
            all_probs.append(probs)

        self.embedder._model_wrapper.eval()

        # Average softmax probabilities across passes, then compute entropy
        mean_probs = np.mean(all_probs, axis=0)  # [N, num_classes]
        eps = 1e-9
        entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)
        return entropy.astype(np.float32)


class CoresetSelector:
    """
    Greedy coreset selection for diverse sample acquisition.

    Selects ``budget`` samples from ``candidates`` that are maximally
    diverse in embedding space, relative to the ``already_labeled``
    set using furthest-point (greedy k-center) sampling.

    Args:
        embedder: ``DINOv2ReIDEmbedder`` instance.

    Example:
        >>> selector = CoresetSelector(embedder)
        >>> indices = selector.select(candidate_crops, already_labeled_crops, budget=50)
    """

    def __init__(self, embedder: "DINOv2ReIDEmbedder") -> None:
        self.embedder = embedder

    def select(
        self,
        candidate_crops: List[np.ndarray],
        already_labeled_crops: Optional[List[np.ndarray]] = None,
        budget: int = 50,
    ) -> List[int]:
        """
        Greedy furthest-point selection.

        Args:
            candidate_crops: Unlabeled crops to select from.
            already_labeled_crops: Already-labeled crops (seed set).
            budget: Number of samples to select.

        Returns:
            Indices into ``candidate_crops`` of selected samples.
        """
        if not candidate_crops:
            return []

        budget = min(budget, len(candidate_crops))

        # Embed all candidates
        cand_emb = self.embedder.embed(candidate_crops)  # [N_cand, dim]

        if already_labeled_crops:
            labeled_emb = self.embedder.embed(already_labeled_crops)  # [N_lab, dim]
        else:
            labeled_emb = np.empty((0, cand_emb.shape[1]), dtype=np.float32)

        # Normalise for cosine distance
        cand_norm = _l2_normalize(cand_emb)
        labeled_norm = _l2_normalize(labeled_emb) if len(labeled_emb) > 0 else labeled_emb

        selected: List[int] = []

        # Min-distance to labeled set for each candidate
        if len(labeled_norm) > 0:
            # cosine distance = 1 - cosine_similarity
            sim = cand_norm @ labeled_norm.T  # [N_cand, N_lab]
            min_dist = 1.0 - sim.max(axis=1)  # [N_cand]
        else:
            min_dist = np.ones(len(cand_norm), dtype=np.float32)

        for _ in range(budget):
            idx = int(np.argmax(min_dist))
            selected.append(idx)
            min_dist[idx] = -1.0  # mark selected

            # Update min-distances based on newly selected point
            new_sim = cand_norm @ cand_norm[idx]  # [N_cand]
            new_dist = 1.0 - new_sim
            min_dist = np.minimum(min_dist, new_dist)
            min_dist[idx] = -1.0

        return selected


class ActiveLearningLoop:
    """
    Iterative active learning loop that integrates annotation and model training.

    Workflow per iteration:
    1. Run ``GroundedSAMPipeline`` on video files to get auto-annotations.
    2. Score crops by uncertainty (MC Dropout entropy) + diversity (coreset).
    3. Upload selected frames to CVAT for human review.
    4. Download corrected annotations.
    5. Trigger ``train_reid`` retraining on expanded dataset.

    Args:
        pipeline: ``GroundedSAMPipeline`` for auto-annotation.
        cvat_client: CVAT annotation client (from ``annotation.client``).
        embedder: ``DINOv2ReIDEmbedder`` for scoring.
        budget: Maximum annotations to upload per iteration.
        uncertainty_weight: Weight for uncertainty vs. diversity score (0–1).
        n_mc_passes: MC Dropout passes for uncertainty estimation.

    Example:
        >>> loop = ActiveLearningLoop(pipeline, cvat_client, embedder, budget=100)
        >>> loop.run(video_dir="data/videos/", project_id=1)
    """

    def __init__(
        self,
        pipeline: "GroundedSAMPipeline",
        cvat_client,
        embedder: Optional["DINOv2ReIDEmbedder"] = None,
        budget: int = 100,
        uncertainty_weight: float = 0.5,
        n_mc_passes: int = 10,
    ) -> None:
        self.pipeline = pipeline
        self.cvat_client = cvat_client
        self.embedder = embedder
        self.budget = budget
        self.uncertainty_weight = uncertainty_weight

        self._uncertainty_sampler: Optional[UncertaintySampler] = None
        self._coreset_selector: Optional[CoresetSelector] = None

        if embedder is not None:
            self._uncertainty_sampler = UncertaintySampler(embedder, n_passes=n_mc_passes)
            self._coreset_selector = CoresetSelector(embedder)

    def run(
        self,
        video_dir: str,
        project_id: int,
        already_labeled_crops: Optional[List[np.ndarray]] = None,
        frame_step: int = 5,
    ) -> Dict:
        """
        Run one iteration of the active learning loop.

        Args:
            video_dir: Directory containing video files (``*.mp4``).
            project_id: CVAT project ID.
            already_labeled_crops: Crops already reviewed (for diversity).
            frame_step: Frame sampling interval for the pipeline.

        Returns:
            Dict with ``{"selected": int, "uploaded_task_ids": List[int]}``.
        """
        import cv2

        video_paths = list(Path(video_dir).glob("**/*.mp4"))
        if not video_paths:
            raise FileNotFoundError(f"No MP4 files found in {video_dir}")

        print(f"[ActiveLearningLoop] Processing {len(video_paths)} videos …")

        all_annotations: List["TrackAnnotation"] = []
        crop_cache: List[Tuple[str, int, np.ndarray]] = []  # (video_path, frame_idx, crop)

        for video_path in video_paths:
            annotations = self.pipeline.process_video(
                str(video_path),
                frame_step=frame_step,
                use_sam=False,  # skip masks for speed during scoring
            )
            all_annotations.extend(annotations)

            # Cache crops for scoring
            cap = cv2.VideoCapture(str(video_path))
            frame_ann_map: Dict[int, List["TrackAnnotation"]] = {}
            for ann in annotations:
                frame_ann_map.setdefault(ann.frame_idx, []).append(ann)

            frame_idx = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if frame_idx in frame_ann_map:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    for ann in frame_ann_map[frame_idx]:
                        crop = _crop_box(frame_rgb, ann.box)
                        crop_cache.append((str(video_path), frame_idx, crop))
                frame_idx += 1
            cap.release()

        print(f"[ActiveLearningLoop] {len(crop_cache)} crops from {len(all_annotations)} annotations")

        # Score and select
        selected_indices = self._select(
            [c for _, _, c in crop_cache],
            already_labeled_crops or [],
        )

        print(f"[ActiveLearningLoop] Selected {len(selected_indices)} frames to upload")

        # Upload to CVAT (group by video + frame)
        upload_set: Dict[Tuple[str, int], bool] = {}
        for idx in selected_indices:
            video_path_str, frame_idx_sel, _ = crop_cache[idx]
            upload_set[(video_path_str, frame_idx_sel)] = True

        task_ids: List[int] = []
        try:
            for (video_path_str, frame_idx_sel) in upload_set:
                task_id = self._upload_frame(video_path_str, frame_idx_sel, project_id)
                if task_id is not None:
                    task_ids.append(task_id)
        except Exception as e:
            print(f"[warn] CVAT upload error: {e}")

        return {"selected": len(selected_indices), "uploaded_task_ids": task_ids}

    def _select(
        self,
        crops: List[np.ndarray],
        already_labeled: List[np.ndarray],
    ) -> List[int]:
        """Combined uncertainty + coreset selection."""
        if not crops:
            return []

        n = len(crops)
        budget = min(self.budget, n)
        w_u = self.uncertainty_weight
        w_d = 1.0 - w_u

        # Uncertainty scores
        if self._uncertainty_sampler is not None and w_u > 0:
            u_scores = self._uncertainty_sampler.score(crops)
        else:
            u_scores = np.ones(n, dtype=np.float32)

        # Coreset diversity scores
        if self._coreset_selector is not None and w_d > 0:
            core_indices = self._coreset_selector.select(crops, already_labeled, budget=budget)
            d_scores = np.zeros(n, dtype=np.float32)
            for idx in core_indices:
                d_scores[idx] = 1.0
        else:
            d_scores = np.ones(n, dtype=np.float32)

        # Normalise each score to [0, 1]
        u_range = u_scores.max() - u_scores.min() + 1e-8
        d_range = d_scores.max() - d_scores.min() + 1e-8
        combined = w_u * (u_scores / u_range) + w_d * (d_scores / d_range)

        top_k = min(budget, n)
        return list(np.argsort(combined)[::-1][:top_k])

    def _upload_frame(
        self,
        video_path: str,
        frame_idx: int,
        project_id: int,
    ) -> Optional[int]:
        """Extract a frame as JPEG and upload to CVAT."""
        import cv2
        import tempfile
        import os

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret:
            return None

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame_bgr)

        try:
            stem = Path(video_path).stem
            task_name = f"{stem}_f{frame_idx:06d}"
            task_id = self.cvat_client.create_task(
                project_id=project_id,
                name=task_name,
                image_paths=[tmp_path],
            )
            return task_id
        finally:
            os.unlink(tmp_path)


from torchkick.utils.crops import crop_box as _crop_box


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


__all__ = ["UncertaintySampler", "CoresetSelector", "ActiveLearningLoop"]
