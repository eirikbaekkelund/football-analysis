"""
Grounded-SAM automated labeling pipeline.

GroundingDINO → bounding boxes from text prompt → SAM2 → instance masks →
DINOv2ReIDEmbedder → team assignment.

The pipeline outputs ``List[TrackAnnotation]`` objects that are compatible
with the existing ``prelabel.upload_player_tracks()`` interface.

Example:
    >>> from torchkick.annotation.grounded_sam import GroundedSAMPipeline
    >>> pipeline = GroundedSAMPipeline()
    >>> annotations = pipeline.process_video("match.mp4", frame_step=5)

CLI:
    $ torchkick label grounded-sam --video match.mp4 --project-id 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from torchkick.models.reid import DINOv2ReIDEmbedder


@dataclass
class TrackAnnotation:
    """Single detection/track annotation for a frame."""

    frame_idx: int
    track_id: int
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None  # H×W uint8
    team: int = -1  # 0=home, 1=away, 2=ref, -1=unknown
    confidence: float = 1.0


class GroundedSAMPipeline:
    """
    End-to-end automated labeling pipeline using GroundingDINO + SAM2.

    Runs GroundingDINO to obtain bounding boxes from a text prompt, then
    refines them to instance masks with SAM2.  Optionally uses a
    ``DINOv2ReIDEmbedder`` to assign team labels to each detection.

    Args:
        grounding_dino_model: GroundingDINO model name or local path.
        sam2_checkpoint: Path to SAM2 checkpoint (``sam2_hiera_large.pt``).
        sam2_config: SAM2 config name (e.g. ``"sam2_hiera_l.yaml"``).
        reid_embedder: Optional ``DINOv2ReIDEmbedder`` for team assignment.
        text_prompt: Text prompt for GroundingDINO object detection.
        box_threshold: Detection confidence threshold for GroundingDINO.
        text_threshold: Text similarity threshold.
        device: Device string.

    Example:
        >>> pipeline = GroundedSAMPipeline(reid_embedder=embedder)
        >>> anns = pipeline.process_video("match.mp4", frame_step=5)
    """

    DEFAULT_PROMPT = "soccer player . ball . referee"

    def __init__(
        self,
        grounding_dino_model: str = "IDEA-Research/grounding-dino-base",
        sam2_checkpoint: Optional[str] = None,
        sam2_config: str = "sam2_hiera_l.yaml",
        reid_embedder: Optional["DINOv2ReIDEmbedder"] = None,
        text_prompt: str = DEFAULT_PROMPT,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ) -> None:
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.reid_embedder = reid_embedder
        self.device = device

        # Lazy-load GroundingDINO
        self._gdino = None
        self._gdino_processor = None
        self._gdino_model_name = grounding_dino_model

        # Lazy-load SAM2
        self._sam2 = None
        self._sam2_checkpoint = sam2_checkpoint
        self._sam2_config = sam2_config

        # ByteTracker for frame-to-frame ID consistency
        from torchkick.tracking.bytetrack import ByteTracker

        self._tracker = ByteTracker(track_thresh=0.35, track_buffer=10, match_thresh=0.8)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_gdino(self) -> None:
        if self._gdino is not None:
            return
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            import torch

            self._gdino_processor = AutoProcessor.from_pretrained(self._gdino_model_name)
            self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(self._gdino_model_name).to(self.device)
            self._gdino.eval()
        except ImportError as e:
            raise ImportError(
                "GroundingDINO requires transformers>=4.35.0. " "Install: pip install torchkick[labeling]"
            ) from e

    def _load_sam2(self) -> None:
        if self._sam2 is not None:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch

            ckpt = self._sam2_checkpoint
            if ckpt is None:
                raise ValueError(
                    "sam2_checkpoint must be provided. "
                    "Download from https://github.com/facebookresearch/segment-anything-2"
                )
            sam2_model = build_sam2(self._sam2_config, ckpt, device=self.device)
            self._sam2 = SAM2ImagePredictor(sam2_model)
        except ImportError as e:
            raise ImportError("SAM2 requires sam2>=1.0. Install: pip install torchkick[labeling]") from e

    # ------------------------------------------------------------------
    # Core frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
        use_sam: bool = True,
    ) -> List[TrackAnnotation]:
        """
        Detect and (optionally) segment all players/ball/referee in one frame.

        Args:
            frame_bgr: BGR frame as uint8 numpy array.
            frame_idx: Frame index (used for output annotations).
            use_sam: If True, refine boxes to masks with SAM2.

        Returns:
            List of ``TrackAnnotation`` for this frame.
        """
        self._load_gdino()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # GroundingDINO detection
        boxes, scores = self._gdino_detect(frame_rgb)
        if len(boxes) == 0:
            return []

        # Track across frames — pass boxes with confidence scores
        if boxes:
            dets = np.hstack(
                [
                    np.array(boxes, dtype=np.float32),
                    np.array(scores, dtype=np.float32).reshape(-1, 1),
                ]
            )
        else:
            dets = np.empty((0, 5), dtype=np.float32)
        active_tracks = self._tracker.update(dets)

        # Optionally refine boxes to masks
        masks: List[Optional[np.ndarray]] = [None] * len(active_tracks)
        if use_sam and self._sam2_checkpoint is not None:
            self._load_sam2()
            track_boxes = [box for _, box in active_tracks]
            masks = self._sam2_segment(frame_rgb, track_boxes)

        # Extract crops and classify teams
        annotations = []
        crops = []
        for track_id, box in active_tracks:
            crop = _crop_box(frame_rgb, box)
            crops.append(crop)

        team_labels = np.full(len(crops), -1, dtype=int)
        if self.reid_embedder is not None and crops:
            try:
                team_labels = self.reid_embedder.classify_team(crops)
            except Exception:
                pass

        for i, (track_id, box) in enumerate(active_tracks):
            annotations.append(
                TrackAnnotation(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    box=tuple(float(v) for v in box),
                    mask=masks[i],
                    team=int(team_labels[i]),
                    confidence=1.0,
                )
            )

        return annotations

    def process_video(
        self,
        video_path: str,
        frame_step: int = 5,
        max_frames: Optional[int] = None,
        use_sam: bool = True,
    ) -> List[TrackAnnotation]:
        """
        Run the pipeline over a video file.

        Args:
            video_path: Path to input video.
            frame_step: Process every N-th frame.
            max_frames: Stop after this many frames (None = full video).
            use_sam: Refine boxes to instance masks.

        Returns:
            Flat list of all ``TrackAnnotation`` across all processed frames.
        """
        self._tracker.reset()
        all_annotations: List[TrackAnnotation] = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frame_idx = 0
        processed = 0

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    anns = self.process_frame(frame_bgr, frame_idx=frame_idx, use_sam=use_sam)
                    all_annotations.extend(anns)
                    processed += 1

                    if processed % 50 == 0:
                        print(f"[GroundedSAMPipeline] Processed {processed} frames ({frame_idx} total)")

                    if max_frames is not None and processed >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()

        print(f"[GroundedSAMPipeline] Done: {len(all_annotations)} annotations from {processed} frames")
        return all_annotations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gdino_detect(
        self,
        frame_rgb: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Run GroundingDINO and return (boxes [N,4], scores [N])."""
        import torch
        from PIL import Image

        image = Image.fromarray(frame_rgb)
        inputs = self._gdino_processor(
            images=image,
            text=self.text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._gdino(**inputs)

        results = self._gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )

        if not results or len(results[0]["boxes"]) == 0:
            return [], []

        boxes_t = results[0]["boxes"].cpu().numpy()  # [N, 4] xyxy
        scores_t = results[0]["scores"].cpu().numpy()  # [N]
        return list(boxes_t), list(scores_t)

    def _sam2_segment(
        self,
        frame_rgb: np.ndarray,
        boxes: List[np.ndarray],
    ) -> List[Optional[np.ndarray]]:
        """Return per-box binary masks using SAM2."""
        import torch
        import numpy as np

        if not boxes:
            return []

        self._sam2.set_image(frame_rgb)
        masks_out: List[Optional[np.ndarray]] = []

        boxes_arr = np.array(boxes, dtype=np.float32)  # [N, 4]
        with torch.no_grad():
            pred_masks, _, _ = self._sam2.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_arr,
                multimask_output=False,
            )

        # pred_masks: [N, 1, H, W]
        for i in range(len(boxes)):
            if i < pred_masks.shape[0]:
                m = pred_masks[i, 0].astype(np.uint8)
                masks_out.append(m)
            else:
                masks_out.append(None)

        return masks_out


from torchkick.utils.crops import crop_box as _crop_box


__all__ = ["TrackAnnotation", "GroundedSAMPipeline"]
