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


class RTDETRDetector:
    """
    RT-DETR-X player/ball/referee detector.

    Wraps HuggingFace RT-DETR (rtdetr_r101vd) for high-accuracy transformer-based
    detection. Uses torch.compile for ~2x speedup after first frame.

    Args:
        weights_path: Path to fine-tuned checkpoint (.pth). If None, uses base HF weights.
        model_name: HuggingFace model identifier.
        device: Torch device string.
        conf_threshold: Minimum confidence to keep a detection.
        player_class_id: Class index for players in the fine-tuned head (default 0).

    Example:
        >>> detector = RTDETRDetector("weights/rtdetr_finetuned.pth")
        >>> detections = detector.detect(frame_bgr)
        >>> for det in detections:
        ...     print(det.bbox, det.confidence)
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        model_name: str = "PekingU/rtdetr_r101vd",
        device: str = "cuda",
        conf_threshold: float = 0.3,
        player_class_id: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.player_class_id = player_class_id

        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

            self._processor = RTDetrImageProcessor.from_pretrained(model_name)
            self._model = RTDetrForObjectDetection.from_pretrained(model_name)

            if weights_path is not None:
                checkpoint = torch.load(str(weights_path), map_location=self.device, weights_only=True)
                state = checkpoint.get("model_state_dict", checkpoint)
                self._model.load_state_dict(state)

            self._model.to(self.device)
            self._model.eval()
            self._model = torch.compile(self._model, mode="reduce-overhead")
        except ImportError:
            raise ImportError("transformers>=4.35.0 required. Install with: pip install torchkick[reid]")

    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect players (and ball/referee) in a single BGR frame.

        Args:
            frame: BGR image array.

        Returns:
            List of Detection objects with bbox, confidence, class_id.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)

        h, w = frame.shape[:2]
        target_size = torch.tensor([[h, w]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_size,
            threshold=self.conf_threshold,
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy(),
        ):
            detections.append(
                Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(score),
                    class_id=int(label),
                    class_name="player" if int(label) == self.player_class_id else "other",
                )
            )

        return detections

    @torch.inference_mode()
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect in a batch of BGR frames.

        Args:
            frames: List of BGR image arrays.

        Returns:
            List of detection lists, one per frame.
        """
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        inputs = self._processor(images=frames_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)

        target_sizes = torch.tensor([[f.shape[0], f.shape[1]] for f in frames], device=self.device)

        all_results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.conf_threshold,
        )

        batch_results = []
        for i, frame in enumerate(frames):
            results = all_results[i]

            detections = []
            for score, label, box in zip(
                results["scores"].cpu().numpy(),
                results["labels"].cpu().numpy(),
                results["boxes"].cpu().numpy(),
            ):
                detections.append(
                    Detection(
                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        confidence=float(score),
                        class_id=int(label),
                        class_name="player" if int(label) == self.player_class_id else "other",
                    )
                )
            batch_results.append(detections)

        return batch_results


class RFDETRDetector:
    """
    RF-DETR player/referee detector (DINOv2 backbone).

    AP50 73.6 vs RT-DETR-R101's ~60 at the same ~5ms latency. Drop-in
    replacement for ``RTDETRDetector`` with the same ``detect`` / ``detect_batch``
    interface.

    Args:
        weights_path: Path to fine-tuned checkpoint. If None, downloads pretrained weights.
        model_size: "m" (RFDETRBase, default) or "l" (RFDETRLarge).
        device: Torch device string.
        conf_threshold: Minimum confidence to keep a detection.
        player_class_id: Class index for players in the fine-tuned head (default 0).

    Example:
        >>> detector = RFDETRDetector()  # downloads pretrained weights
        >>> detections = detector.detect(frame_bgr)
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        model_size: str = "m",
        device: str = "cuda",
        conf_threshold: float = 0.3,
        player_class_id: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.player_class_id = player_class_id

        try:
            from rfdetr import RFDETRBase, RFDETRLarge

            model_cls = RFDETRLarge if model_size == "l" else RFDETRBase
            kwargs = {"pretrain_weights": str(weights_path)} if weights_path is not None else {}
            self._model = model_cls(**kwargs)
            self._model = torch.compile(self._model, mode="reduce-overhead")
        except ImportError:
            raise ImportError("rf-detr required. Install with: pip install torchkick[rfdetr]")

    def _to_detections(self, sv_detections) -> List[Detection]:
        """Convert supervision.Detections to Detection objects."""
        results = []
        if sv_detections is None or len(sv_detections) == 0:
            return results
        xyxy = sv_detections.xyxy
        conf = sv_detections.confidence if sv_detections.confidence is not None else [1.0] * len(xyxy)
        cls_ids = sv_detections.class_id if sv_detections.class_id is not None else [0] * len(xyxy)
        for box, score, label in zip(xyxy, conf, cls_ids):
            results.append(
                Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(score),
                    class_id=int(label),
                    class_name="player" if int(label) == self.player_class_id else "other",
                )
            )
        return results

    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect players (and ball/referee) in a single BGR frame.

        Args:
            frame: BGR image array.

        Returns:
            List of Detection objects with bbox, confidence, class_id.
        """
        from PIL import Image

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        sv_detections = self._model.predict(img_pil, threshold=self.conf_threshold)
        return self._to_detections(sv_detections)

    @torch.inference_mode()
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect in a batch of BGR frames (sequential — rfdetr has no native batch API).

        Args:
            frames: List of BGR image arrays.

        Returns:
            List of detection lists, one per frame.
        """
        return [self.detect(f) for f in frames]


__all__ = [
    "Detection",
    "PlayerDetector",
    "RTDETRDetector",
    "RFDETRDetector",
]
