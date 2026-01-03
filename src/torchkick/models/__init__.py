"""
Neural network model wrappers for football analysis.

This module provides wrappers for pitch line detection, player detection,
and appearance embedding models.

Submodules:
    pitch: Pitch line keypoint detection and camera calibration.
    player: Player detection and appearance embedding.

Example:
    >>> from torchkick.models import (
    ...     PitchLineDetector,
    ...     PitchCalibrator,
    ...     PlayerDetector,
    ...     AppearanceEmbedder,
    ... )
    >>> 
    >>> # Pitch calibration
    >>> detector = PitchLineDetector(weights_kp, weights_lines)
    >>> calibrator = PitchCalibrator(detector)
    >>> P = calibrator.process_frame(frame)
    >>> 
    >>> # Player detection
    >>> player_det = PlayerDetector("yolo_weights.pt")
    >>> detections = player_det.detect(frame)
"""

from __future__ import annotations

# Pitch models
from torchkick.models.pitch import (
    LINE_COORDINATES_3D,
    projection_from_cam_params,
    project_lines_to_image,
    PitchLineDetector,
    PitchCalibrator,
)

# Player models
from torchkick.models.player import (
    Detection,
    PlayerDetector,
    ReIDConfig,
    AppearanceEmbedder,
)

__all__ = [
    # Pitch
    "LINE_COORDINATES_3D",
    "projection_from_cam_params",
    "project_lines_to_image",
    "PitchLineDetector",
    "PitchCalibrator",
    # Player
    "Detection",
    "PlayerDetector",
    "ReIDConfig",
    "AppearanceEmbedder",
]
