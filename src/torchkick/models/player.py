"""
Player detection model wrapper.

Example:
    >>> from torchkick.models.player import PlayerDetector
    >>> detector = PlayerDetector("yolov11_player_tracker.pt")
    >>> detections = detector.detect(frame)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    """
    Container for a single detection.

    Attributes:
        bbox: [x1, y1, x2, y2] bounding box.
        confidence: Detection confidence score.
        class_id: Class label ID.
        class_name: Human-readable class name.
        track_id: Optional tracking ID (from tracker).
        embedding: Optional appearance embedding.
    """

    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str = "player"
    track_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


class PlayerDetector:
    """
    YOLO-based player detection.

    Wraps Ultralytics YOLO for player/ball/referee detection.

    Args:
        weights_path: Path to YOLO weights (.pt or .onnx).
        device: Torch device string.
        conf_threshold: Confidence threshold.
        iou_threshold: NMS IoU threshold.
        classes: List of class IDs to detect.

    Example:
        >>> detector = PlayerDetector("yolov11_player_tracker.pt")
        >>> detections = detector.detect(frame)
        >>> for det in detections:
        ...     print(det.bbox, det.confidence)
    """

    CLASS_NAMES = {
        0: "player",
        1: "goalkeeper",
        2: "referee",
        3: "ball",
    }

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
    ) -> None:
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or [0, 1, 2, 3]

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(weights_path))
            self.model.to(device)
        except ImportError:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")

    def detect(self, frame: np.ndarray, verbose: bool = False) -> List[Detection]:
        """
        Detect players in a frame.

        Args:
            frame: BGR image.
            verbose: Print detection info.

        Returns:
            List of Detection objects.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=verbose,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                detections.append(
                    Detection(
                        bbox=tuple(bbox),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self.CLASS_NAMES.get(cls_id, "unknown"),
                    )
                )
        return detections

    def detect_with_tracking(
        self,
        frame: np.ndarray,
        tracker: str = "botsort",
        persist: bool = True,
    ) -> List[Detection]:
        """
        Detect and track players.

        Args:
            frame: BGR image.
            tracker: Tracker type ("bytetrack", "botsort").
            persist: Persist tracks across frames.

        Returns:
            List of Detection objects with track_id populated.
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            tracker=f"{tracker}.yaml",
            persist=persist,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                track_id = int(boxes.id[i].cpu()) if boxes.id is not None else None
                detections.append(
                    Detection(
                        bbox=tuple(bbox),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self.CLASS_NAMES.get(cls_id, "unknown"),
                        track_id=track_id,
                    )
                )
        return detections


__all__ = ["Detection", "PlayerDetector"]
