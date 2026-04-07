"""
Body pose estimation for soccer players using ViTPose-B.

Detects 17 COCO keypoints per player crop and returns them in full-frame
pixel coordinates, ready for 3D lifting via PoseLift3D.

Usage:
    >>> from torchkick.models.pose import BodyPoseDetector
    >>> detector = BodyPoseDetector(device="cuda")
    >>> results = detector.detect_batch(frame_bgr, boxes)  # boxes: [N, 4] xyxy
    >>> for r in results:
    ...     print(r.keypoints.shape)   # [17, 2] full-frame pixel coords
    ...     print(r.scores.shape)      # [17]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch


@dataclass
class PoseResult:
    """Single-player pose estimation result."""

    keypoints: np.ndarray  # [17, 2] float32 — full-frame pixel (x, y)
    scores: np.ndarray  # [17]    float32 — per-keypoint confidence
    bbox: Tuple[float, float, float, float]  # original [x1, y1, x2, y2]


class BodyPoseDetector:
    """
    ViTPose-B body pose detector.

    Outputs 17 COCO keypoints per player in full-frame pixel coordinates.
    Loads pretrained weights from HuggingFace (usyd-community/vitpose-base-simple).

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device string ("cuda" or "cpu").
        conf_threshold: Minimum keypoint confidence for visualization/lifting.
        max_batch_size: Maximum crops per forward pass (reduce if OOM).
    """

    # COCO 17-keypoint skeleton connections (for visualization)
    COCO_SKELETON: List[Tuple[int, int]] = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # head
        (5, 6),  # shoulders
        (5, 7),
        (7, 9),  # left arm
        (6, 8),
        (8, 10),  # right arm
        (5, 11),
        (6, 12),  # torso sides
        (11, 12),  # hips
        (11, 13),
        (13, 15),  # left leg
        (12, 14),
        (14, 16),  # right leg
    ]

    # COCO joint index → name
    JOINT_NAMES: List[str] = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    ANKLE_INDICES: Tuple[int, int] = (15, 16)  # left_ankle, right_ankle

    # ViTPose standard input size: width=192, height=256 (3:4 aspect ratio)
    _INPUT_W: int = 192
    _INPUT_H: int = 256

    def __init__(
        self,
        model_name: str = "usyd-community/vitpose-base-simple",
        device: str = "cuda",
        conf_threshold: float = 0.3,
        max_batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.conf_threshold = conf_threshold
        self.max_batch_size = max_batch_size
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import ViTPoseForPoseEstimation, ViTPoseImageProcessor
        except ImportError:
            raise ImportError(
                "transformers>=4.46.0 required for BodyPoseDetector. " "Install: pip install torchkick[reid]"
            )
        self._processor = ViTPoseImageProcessor.from_pretrained(self.model_name)
        self._model = ViTPoseForPoseEstimation.from_pretrained(self.model_name)
        self._model = self._model.to(self.device).eval()

    def _pad_box_to_aspect(
        self,
        frame_h: int,
        frame_w: int,
        box: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Expand box to 3:4 (w:h) aspect ratio symmetrically, clamped to frame.

        Returns (padded_box [4], meta) where meta stores the padded box
        coordinates needed to remap keypoints back to full-frame space.
        """
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        target_aspect = self._INPUT_W / self._INPUT_H  # 192/256 = 0.75
        current_aspect = bw / max(bh, 1)

        if current_aspect < target_aspect:
            # Too tall — expand width
            new_w = bh * target_aspect
            cx = (x1 + x2) / 2
            x1 = cx - new_w / 2
            x2 = cx + new_w / 2
        else:
            # Too wide — expand height
            new_h = bw / target_aspect
            cy = (y1 + y2) / 2
            y1 = cy - new_h / 2
            y2 = cy + new_h / 2

        # Clamp to frame
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(frame_w), x2)
        y2 = min(float(frame_h), y2)

        padded_box = np.array([x1, y1, x2, y2], dtype=np.float32)
        meta = {
            "x1": x1,
            "y1": y1,
            "crop_w": x2 - x1,
            "crop_h": y2 - y1,
        }
        return padded_box, meta

    @torch.inference_mode()
    def detect_batch(
        self,
        frame_bgr: np.ndarray,
        boxes: np.ndarray,
    ) -> List[PoseResult]:
        """
        Detect 17 COCO keypoints for each box in a single batched forward pass.

        Args:
            frame_bgr: Full frame in BGR uint8.
            boxes: [N, 4] float32 bounding boxes in [x1, y1, x2, y2] pixel coords.

        Returns:
            List[PoseResult] of length N. Empty list if boxes is empty.
        """
        if len(boxes) == 0:
            return []

        self._ensure_loaded()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame_bgr.shape[:2]

        results: List[PoseResult] = []
        boxes = np.asarray(boxes, dtype=np.float32)

        # Process in chunks to avoid OOM
        for chunk_start in range(0, len(boxes), self.max_batch_size):
            chunk_boxes = boxes[chunk_start : chunk_start + self.max_batch_size]
            chunk_results = self._process_chunk(frame_rgb, frame_h, frame_w, chunk_boxes)
            results.extend(chunk_results)

        return results

    def _process_chunk(
        self,
        frame_rgb: np.ndarray,
        frame_h: int,
        frame_w: int,
        boxes: np.ndarray,
    ) -> List[PoseResult]:
        from PIL import Image as PILImage

        crops_pil: List[PILImage.Image] = []
        metas: List[dict] = []
        original_boxes: List[np.ndarray] = []

        for box in boxes:
            padded_box, meta = self._pad_box_to_aspect(frame_h, frame_w, box)
            x1, y1, x2, y2 = padded_box.astype(int)
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((self._INPUT_H, self._INPUT_W, 3), dtype=np.uint8)
            crops_pil.append(PILImage.fromarray(crop))
            metas.append(meta)
            original_boxes.append(box)

        # Batch preprocessing via ViTPoseImageProcessor
        inputs = self._processor(images=crops_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)

        # Post-process: heatmaps → keypoints in crop space
        # boxes_for_postprocess: list of [x1, y1, x2, y2] in crop space (full crop)
        boxes_crop = [[[0, 0, meta["crop_w"], meta["crop_h"]]] for meta in metas]
        # ViTPoseImageProcessor.post_process_pose_estimation returns a list of dicts
        # Each dict has "keypoints" [1, 17, 2] and "scores" [1, 17]
        pose_list = self._processor.post_process_pose_estimation(
            outputs,
            boxes=boxes_crop,
        )

        results: List[PoseResult] = []
        for i, pose_per_image in enumerate(pose_list):
            meta = metas[i]
            orig_box = original_boxes[i]

            if len(pose_per_image) == 0:
                # No detection — return zeros
                results.append(
                    PoseResult(
                        keypoints=np.zeros((17, 2), dtype=np.float32),
                        scores=np.zeros(17, dtype=np.float32),
                        bbox=tuple(orig_box.tolist()),
                    )
                )
                continue

            person = pose_per_image[0]
            kps = np.array(person["keypoints"], dtype=np.float32)  # [17, 2]
            scores = np.array(person["scores"], dtype=np.float32)  # [17]

            # Remap from crop space to full-frame pixel space
            scale_x = meta["crop_w"] / self._INPUT_W
            scale_y = meta["crop_h"] / self._INPUT_H
            kps[:, 0] = kps[:, 0] * scale_x + meta["x1"]
            kps[:, 1] = kps[:, 1] * scale_y + meta["y1"]

            results.append(
                PoseResult(
                    keypoints=kps,
                    scores=scores,
                    bbox=tuple(orig_box.tolist()),
                )
            )

        return results
