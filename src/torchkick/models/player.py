"""
Player detection and tracking models.

This module provides model wrappers for player detection, including
YOLO-based detection and ReID-based appearance embedding.

Example:
    >>> from torchkick.models.player import PlayerDetector
    >>> 
    >>> detector = PlayerDetector("yolov11_player_tracker.pt")
    >>> detections = detector.detect(frame)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

# Type aliases
TensorLike = Union[np.ndarray, torch.Tensor]
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

        # Load YOLO model
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(weights_path))
            self.model.to(device)
        except ImportError:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")

    def detect(
        self,
        frame: np.ndarray,
        verbose: bool = False,
    ) -> List[Detection]:
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
                cls_name = self.CLASS_NAMES.get(cls_id, "unknown")

                det = Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                )
                detections.append(det)

        return detections

    def detect_with_tracking(
        self,
        frame: np.ndarray,
        tracker: str = "bytetrack",
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
                cls_name = self.CLASS_NAMES.get(cls_id, "unknown")

                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu())

                det = Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    track_id=track_id,
                )
                detections.append(det)

        return detections


@dataclass
class ReIDConfig:
    """Configuration for ReID model."""

    embedding_dim: int = 512
    num_classes: int = 1000
    backbone: str = "osnet"
    input_size: Tuple[int, int] = (128, 256)  # width, height
    use_team_head: bool = True
    use_role_head: bool = True
    dropout: float = 0.3
    pretrained: bool = True


class AppearanceEmbedder:
    """
    Extract appearance embeddings for player re-identification.

    Supports both handcrafted features (color histograms) and
    deep ReID embeddings.

    Args:
        use_deep_features: Use deep ReID model.
        reid_model_path: Path to ReID model weights.
        device: Torch device string.

    Example:
        >>> embedder = AppearanceEmbedder(use_deep_features=False)
        >>> features = embedder.extract(player_crop)
        >>> print(features.shape)  # (768,) for handcrafted
    """

    def __init__(
        self,
        use_deep_features: bool = False,
        reid_model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ) -> None:
        self.use_deep_features = use_deep_features
        self.device = device
        self.reid_model = None

        if use_deep_features and reid_model_path:
            self._load_reid_model(reid_model_path)

    def _load_reid_model(self, path: Union[str, Path]) -> None:
        """Load deep ReID model."""
        # Placeholder - actual model loading would go here
        pass

    def extract_color_histogram(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bins: Tuple[int, int, int] = (8, 12, 8),
    ) -> np.ndarray:
        """
        Extract HSV color histogram.

        Args:
            image: BGR player crop.
            mask: Optional region mask.
            bins: Histogram bins per channel.

        Returns:
            Normalized histogram vector.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, list(bins), [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def get_jersey_mask(
        self,
        image: np.ndarray,
        upper_ratio: float = 0.5,
    ) -> np.ndarray:
        """
        Create mask for jersey region.

        Args:
            image: Player crop.
            upper_ratio: Fraction of height for upper body.

        Returns:
            Binary mask.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        upper_h = int(h * upper_ratio)
        margin_w = int(w * 0.2)
        mask[int(h * 0.1) : upper_h, margin_w : w - margin_w] = 255

        # Remove green background
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))

        return mask

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract full appearance feature vector.

        Args:
            image: BGR player crop.

        Returns:
            Feature vector.
        """
        jersey_mask = self.get_jersey_mask(image)
        hist = self.extract_color_histogram(image, jersey_mask)

        if self.use_deep_features and self.reid_model is not None:
            # TODO: Add deep feature extraction
            pass

        return hist

    def compute_similarity(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
    ) -> float:
        """
        Compute similarity between feature vectors.

        Args:
            feat1: First feature vector.
            feat2: Second feature vector.

        Returns:
            Similarity score (0-1).
        """
        # Histogram intersection
        return np.minimum(feat1, feat2).sum()


__all__ = [
    "Detection",
    "PlayerDetector",
    "ReIDConfig",
    "AppearanceEmbedder",
]
