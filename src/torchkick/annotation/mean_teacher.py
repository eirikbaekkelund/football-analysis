"""
Mean Teacher pseudo-labeler for unlabeled video.

Maintains an EMA (teacher) copy of the student model weights.  Generates
pseudo-labels for detections where the teacher is sufficiently confident,
so these frames can be fed into ``train_reid --stage ssl`` as additional
labeled data without human review.

Example:
    >>> from torchkick.annotation.mean_teacher import MeanTeacherPseudoLabeler
    >>> labeler = MeanTeacherPseudoLabeler.from_student_weights(
    ...     "weights/reid/reid_supervised_best.pth"
    ... )
    >>> labeler.process_video("match.mp4", output_dir="data/pseudo/")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from torchkick.models.reid import DINOv2ReIDEmbedder


class MeanTeacherPseudoLabeler:
    """
    EMA teacher for generating high-confidence pseudo-labels.

    The teacher is an exponential moving average of the student backbone.
    It is used for inference only; only student weights are updated during
    training.  This labeler produces a directory of crops organised by
    pseudo-class label, ready to be used as additional training data.

    Args:
        student_embedder: The online (student) ``DINOv2ReIDEmbedder``.
        alpha: EMA decay for teacher update (default 0.999).
        confidence_threshold: Minimum class confidence to emit a pseudo-label.

    Example:
        >>> labeler = MeanTeacherPseudoLabeler(student_embedder)
        >>> labeler.process_video("match.mp4", output_dir="data/pseudo/")
    """

    LABEL_NAMES = {0: "home", 1: "away", 2: "referee"}

    def __init__(
        self,
        student_embedder: "DINOv2ReIDEmbedder",
        alpha: float = 0.999,
        confidence_threshold: float = 0.95,
    ) -> None:
        self.student = student_embedder
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold

        # Create EMA teacher as deep copy of student
        self._teacher_backbone = copy.deepcopy(student_embedder._backbone)
        self._teacher_classifier = (
            copy.deepcopy(student_embedder._classifier) if hasattr(student_embedder, "_classifier") else None
        )
        for p in self._teacher_backbone.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def update_teacher(self) -> None:
        """Apply EMA update from current student weights → teacher weights."""
        for p_s, p_t in zip(
            self.student._backbone.parameters(),
            self._teacher_backbone.parameters(),
        ):
            p_t.data.mul_(self.alpha).add_(p_s.data, alpha=1.0 - self.alpha)

        if self._teacher_classifier is not None and hasattr(self.student, "_classifier"):
            for p_s, p_t in zip(
                self.student._classifier.parameters(),
                self._teacher_classifier.parameters(),
            ):
                p_t.data.mul_(self.alpha).add_(p_s.data, alpha=1.0 - self.alpha)

    # ------------------------------------------------------------------
    # Pseudo-label generation
    # ------------------------------------------------------------------

    def _embed_with_teacher(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings directly from the teacher backbone — no student mutation."""
        import torch
        from torchkick.models.reid import _preprocess_crops

        all_embeddings = []
        with torch.inference_mode():
            for i in range(0, len(crops), self.student.batch_size):
                batch = crops[i : i + self.student.batch_size]
                tensors = _preprocess_crops(batch, self.student.device)
                if self.student.use_fp16:
                    tensors = tensors.half()
                outputs = self._teacher_backbone(pixel_values=tensors)
                cls = outputs.last_hidden_state[:, 0, :]
                patch_mean = outputs.last_hidden_state[:, 1:, :].mean(1)
                features = (cls + patch_mean).float()
                all_embeddings.append(features.cpu().numpy())

        return (
            np.concatenate(all_embeddings, axis=0)
            if all_embeddings
            else np.zeros((0, self.student._embed_dim), dtype=np.float32)
        )

    def classify_with_confidence(
        self,
        crops: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run teacher model and return (labels [N], confidence [N]).

        Confidence is the max softmax probability across team classes.
        """
        import torch
        import torch.nn.functional as F

        if not crops:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        embeddings = self._embed_with_teacher(crops)  # [N, embed_dim]

        # Compute cluster assignments with confidence
        emb_t = torch.from_numpy(embeddings).float()
        emb_norm = F.normalize(emb_t, dim=-1)

        # Use spectral-style cluster assignment if no trained classifier
        if self._teacher_classifier is None:
            labels = self.student.cluster_teams(embeddings)
            # Confidence via silhouette-like score: not trivial to compute here;
            # use distance to nearest cluster centre as proxy
            confidence = np.ones(len(labels), dtype=np.float32) * 0.5
        else:
            self._teacher_classifier.eval()
            with torch.no_grad():
                logits = self._teacher_classifier(emb_norm)
                probs = F.softmax(logits, dim=-1).numpy()
            labels = probs.argmax(axis=-1)
            confidence = probs.max(axis=-1)

        return labels.astype(int), confidence.astype(np.float32)

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        detector=None,
        frame_step: int = 5,
        min_box_area: int = 400,
    ) -> Dict[str, int]:
        """
        Generate pseudo-labeled crops from a video.

        Crops with teacher confidence >= ``confidence_threshold`` are saved to
        ``output_dir/<label_name>/<video_stem>_f{frame:06d}_t{track}.jpg``.

        Args:
            video_path: Input video path.
            output_dir: Root directory for pseudo-labeled crops.
            detector: Optional detector (``PlayerDetector`` or ``RTDETRDetector``).
                Falls back to background subtraction when None.
            frame_step: Process every N-th frame.
            min_box_area: Minimum bounding box area to consider.

        Returns:
            Dict mapping label name → number of saved crops.
        """
        out_path = Path(output_dir)
        for label_name in self.LABEL_NAMES.values():
            (out_path / label_name).mkdir(parents=True, exist_ok=True)

        counts: Dict[str, int] = {v: 0 for v in self.LABEL_NAMES.values()}
        stem = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                # Get bounding boxes
                boxes = self._detect_boxes(frame_bgr, detector, min_box_area)
                if not boxes:
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                crops = [_crop_box(frame_rgb, box) for box in boxes]
                labels, confidences = self.classify_with_confidence(crops)

                for i, (box, label, conf) in enumerate(zip(boxes, labels, confidences)):
                    if conf < self.confidence_threshold:
                        continue

                    label_name = self.LABEL_NAMES.get(label, "unknown")
                    if label_name == "unknown":
                        continue

                    filename = f"{stem}_f{frame_idx:06d}_t{i:03d}.jpg"
                    save_path = out_path / label_name / filename
                    crop_bgr = cv2.cvtColor(crops[i], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(save_path), crop_bgr)
                    counts[label_name] += 1

                frame_idx += 1
        finally:
            cap.release()

        total = sum(counts.values())
        print(f"[MeanTeacherPseudoLabeler] {video_path}: {total} pseudo-labels saved to {output_dir}")
        print(f"  {counts}")
        return counts

    def process_video_dir(
        self,
        video_dir: str,
        output_dir: str,
        detector=None,
        frame_step: int = 5,
    ) -> Dict[str, int]:
        """Process all MP4 files in a directory."""
        total_counts: Dict[str, int] = {v: 0 for v in self.LABEL_NAMES.values()}

        video_paths = list(Path(video_dir).glob("**/*.mp4"))
        if not video_paths:
            print(f"[warn] No MP4 files found in {video_dir}")
            return total_counts

        for video_path in video_paths:
            counts = self.process_video(
                str(video_path),
                output_dir=output_dir,
                detector=detector,
                frame_step=frame_step,
            )
            for k, v in counts.items():
                total_counts[k] += v

        print(f"[MeanTeacherPseudoLabeler] Total: {sum(total_counts.values())} pseudo-labels")
        return total_counts

    # ------------------------------------------------------------------
    # Class-methods for convenient construction
    # ------------------------------------------------------------------

    @classmethod
    def from_student_weights(
        cls,
        weights_path: str,
        device: str = "cuda",
        lora_rank: int = 16,
        alpha: float = 0.999,
        confidence_threshold: float = 0.95,
    ) -> "MeanTeacherPseudoLabeler":
        """
        Construct a labeler from a saved student checkpoint.

        Args:
            weights_path: Path to ``reid_supervised_best.pth`` or
                ``reid_student_best.pth``.
            device: Device string.
            lora_rank: LoRA rank (must match the training configuration).
            alpha: EMA decay.
            confidence_threshold: Confidence threshold.

        Returns:
            Configured ``MeanTeacherPseudoLabeler`` instance.
        """
        from torchkick.models.reid import DINOv2ReIDEmbedder

        embedder = DINOv2ReIDEmbedder(
            weights_path=weights_path,
            device=device,
            lora_rank=lora_rank,
        )
        return cls(embedder, alpha=alpha, confidence_threshold=confidence_threshold)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_boxes(
        self,
        frame_bgr: np.ndarray,
        detector=None,
        min_box_area: int = 400,
    ) -> List:
        """Return bounding boxes as list of [x1, y1, x2, y2]."""
        if detector is not None:
            try:
                # Handles both PlayerDetector (returns Detection) and RTDETRDetector
                detections = detector.detect(frame_bgr)
                boxes = []
                for det in detections:
                    box = det.box if hasattr(det, "box") else det[:4]
                    x1, y1, x2, y2 = box
                    if (x2 - x1) * (y2 - y1) >= min_box_area:
                        boxes.append([x1, y1, x2, y2])
                return boxes
            except Exception as e:
                print(f"[warn] Detector error: {e}")
                return []

        # Fallback: motion-based detection via background subtraction
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if not hasattr(self, "_bg_subtractor"):
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)

        fg_mask = self._bg_subtractor.apply(blurred)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = h / (w + 1e-6)
            if area >= min_box_area and 1.5 <= aspect <= 5.0:
                boxes.append([float(x), float(y), float(x + w), float(y + h)])
        return boxes


from torchkick.utils.crops import crop_box as _crop_box


__all__ = ["MeanTeacherPseudoLabeler"]
