"""
Ball-specific detector using YOLOv11-nano, with optional tiled (SAHI-style)
inference for small-ball detection in wide-angle broadcast frames.

The ball is small, fast, and easily missed by general player detectors.
A dedicated lightweight model (YOLOv11-nano) running in parallel with the
player detector at ~1ms/frame gives much higher ball recall.

``BallInferenceSlicer`` wraps ``BallDetector`` with tiled inference:
frame → overlapping 320×320 tiles → detect on each → offset coords → NMS merge.
This catches balls as small as 8px that single-pass inference misses.

Example:
    >>> from torchkick.models.ball import BallDetector, BallInferenceSlicer
    >>>
    >>> detector = BallDetector("weights/ball/yolo11n_ball.pt", device="cuda")
    >>> slicer = BallInferenceSlicer(detector)
    >>> balls = slicer.detect(frame_bgr)
    >>> for x1, y1, x2, y2, conf in balls:
    ...     print(f"Ball at ({x1:.0f},{y1:.0f}) conf={conf:.2f}")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


# Ball detection result: (x1, y1, x2, y2, confidence)
BallDetection = Tuple[float, float, float, float, float]


class BallDetector:
    """
    YOLOv11-nano ball detector.

    Runs in parallel with player detection. Filters results to the
    ball class only and returns raw bounding boxes with confidence scores.

    Args:
        weights_path: Path to YOLOv11n weights fine-tuned on ball data.
        device: Torch device string.
        conf_threshold: Minimum confidence to keep detection.
        ball_class_id: Class index for the ball in the model's head (default 0).

    Example:
        >>> detector = BallDetector("weights/yolo11n_ball.pt")
        >>> balls = detector.detect(frame_bgr)
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "cuda",
        conf_threshold: float = 0.3,
        ball_class_id: int = 0,
    ) -> None:
        self.weights_path = str(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.ball_class_id = ball_class_id
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.weights_path)
            self._model.to(self.device)
        except ImportError:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")

    def detect(self, frame: np.ndarray) -> List[BallDetection]:
        """
        Detect ball in a single BGR frame.

        Args:
            frame: BGR image array (H, W, 3).

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples for each ball detection.
            Usually 0 or 1 entries.
        """
        if self._model is None:
            return []

        results = self._model(
            frame,
            conf=self.conf_threshold,
            classes=[self.ball_class_id],
            verbose=False,
        )

        detections: List[BallDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                detections.append((float(x1), float(y1), float(x2), float(y2), conf))

        return detections

    def detect_best(self, frame: np.ndarray) -> Optional[BallDetection]:
        """
        Return the single highest-confidence ball detection or None.

        Args:
            frame: BGR image array.

        Returns:
            (x1, y1, x2, y2, confidence) for the best detection, or None.
        """
        dets = self.detect(frame)
        if not dets:
            return None
        return max(dets, key=lambda d: d[4])


class BallInferenceSlicer:
    """
    Tiled (SAHI-style) inference wrapper for small-ball detection.

    Divides a frame into overlapping tiles, runs ``BallDetector`` on each tile,
    offsets detection coordinates back to full-frame space, merges with a
    single full-frame pass, and applies NMS.  Catches balls as small as 8px
    that single-pass YOLOv11-nano misses in wide-angle broadcast frames.

    No external dependencies — pure NumPy/OpenCV.

    Args:
        detector: A ``BallDetector`` instance to wrap.
        slice_height: Tile height in pixels (default 320).
        slice_width: Tile width in pixels (default 320).
        overlap_ratio: Fractional overlap between adjacent tiles (default 0.2).
        iou_threshold: NMS IoU suppression threshold (default 0.5).

    Example:
        >>> slicer = BallInferenceSlicer(BallDetector("weights/yolo11n_ball.pt"))
        >>> best = slicer.detect_best(frame_bgr)
    """

    def __init__(
        self,
        detector: "BallDetector",
        slice_height: int = 320,
        slice_width: int = 320,
        overlap_ratio: float = 0.2,
        iou_threshold: float = 0.5,
    ) -> None:
        self.detector = detector
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio
        self.iou_threshold = iou_threshold

    def _tile_offsets(self, frame_h: int, frame_w: int):
        """Yield (y_start, x_start) for each tile covering the frame."""
        stride_y = max(1, int(self.slice_height * (1 - self.overlap_ratio)))
        stride_x = max(1, int(self.slice_width * (1 - self.overlap_ratio)))
        y = 0
        while y < frame_h:
            x = 0
            while x < frame_w:
                yield y, x
                x += stride_x
                if x + self.slice_width >= frame_w and x < frame_w - 1:
                    x = frame_w - self.slice_width
            y += stride_y
            if y + self.slice_height >= frame_h and y < frame_h - 1:
                y = frame_h - self.slice_height

    @staticmethod
    def _iou(a: BallDetection, b: BallDetection) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _nms(self, detections: "List[BallDetection]") -> "List[BallDetection]":
        if not detections:
            return []
        detections = sorted(detections, key=lambda d: d[4], reverse=True)
        kept: "List[BallDetection]" = []
        suppressed = [False] * len(detections)
        for i, det in enumerate(detections):
            if suppressed[i]:
                continue
            kept.append(det)
            for j in range(i + 1, len(detections)):
                if not suppressed[j] and self._iou(det, detections[j]) > self.iou_threshold:
                    suppressed[j] = True
        return kept

    def detect(self, frame: "np.ndarray") -> "List[BallDetection]":
        """
        Detect ball using tiled inference + full-frame pass merged with NMS.

        Args:
            frame: BGR image array (H, W, 3).

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples after NMS.
        """
        frame_h, frame_w = frame.shape[:2]
        all_dets: "List[BallDetection]" = []

        # Full-frame pass (catches medium-sized balls near tile boundaries)
        all_dets.extend(self.detector.detect(frame))

        # Tiled passes
        for y0, x0 in self._tile_offsets(frame_h, frame_w):
            y1 = min(y0 + self.slice_height, frame_h)
            x1 = min(x0 + self.slice_width, frame_w)
            tile = frame[y0:y1, x0:x1]
            tile_dets = self.detector.detect(tile)
            for tx1, ty1, tx2, ty2, conf in tile_dets:
                all_dets.append((tx1 + x0, ty1 + y0, tx2 + x0, ty2 + y0, conf))

        return self._nms(all_dets)

    def detect_best(self, frame: "np.ndarray") -> "Optional[BallDetection]":
        """Return highest-confidence detection after tiled NMS, or None."""
        dets = self.detect(frame)
        if not dets:
            return None
        return max(dets, key=lambda d: d[4])


__all__ = ["BallDetector", "BallDetection", "BallInferenceSlicer"]
