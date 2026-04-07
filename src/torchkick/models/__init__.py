"""
Neural network model wrappers for football analysis.

Submodules:
    pitch: Pitch keypoint detection (ViTPose / YOLO-pose).
    player: Player detection (YOLO, FCNN, RT-DETR, RF-DETR).
    ball: Ball detection (YOLOv11-nano, with optional tiled SAHI-style slicer).
    reid: ReID embedding (DINOv2 + PEFT LoRA).

Example:
    >>> from torchkick.models import (
    ...     ViTPoseKeypointDetector,
    ...     YOLOPoseKeypointDetector,
    ...     PlayerDetector,
    ...     RTDETRDetector,
    ...     RFDETRDetector,
    ... )
    >>> pitch_det = YOLOPoseKeypointDetector("weights/yolo_pitch_pose.pt")
    >>> kps, conf = pitch_det.detect(frame)
"""

from __future__ import annotations

# Ball detector (YOLOv11-nano)
from torchkick.models.ball import BallDetector, BallDetection, BallInferenceSlicer

# Pitch keypoint detectors
from torchkick.models.pitch import (
    ViTPoseKeypointDetector,
    YOLOPoseKeypointDetector,
)

# Player models
from torchkick.models.player import (
    Detection,
    PlayerDetector,
    RTDETRDetector,
    RFDETRDetector,
)

# Body pose estimation
from torchkick.models.pose import BodyPoseDetector, PoseResult

__all__ = [
    # Ball
    "BallDetector",
    "BallDetection",
    "BallInferenceSlicer",
    # Pitch
    "ViTPoseKeypointDetector",
    "YOLOPoseKeypointDetector",
    # Player
    "Detection",
    "PlayerDetector",
    "RTDETRDetector",
    "RFDETRDetector",
    # Body pose
    "BodyPoseDetector",
    "PoseResult",
]
