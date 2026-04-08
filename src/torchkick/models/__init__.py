"""
Neural network model wrappers for football analysis.

Submodules:
    pitch: Pitch keypoint detection (DINOv2 heatmap).
    player: Player detection (YOLO wrapper).
    reid: ReID embedding (DINOv2 LoRA, SigLIP zero-shot).

Example:
    >>> from torchkick.models import HeatmapPitchDetector, SigLIPTeamEmbedder
    >>> pitch_det = HeatmapPitchDetector("weights/pitch/best.pt")
    >>> kps, conf = pitch_det.detect(frame)
"""

from __future__ import annotations

# Pitch keypoint detector
from torchkick.models.pitch import (
    DINOv2PitchModel,
    HeatmapPitchDetector,
)

# Player detection wrapper
from torchkick.models.player import (
    Detection,
    PlayerDetector,
)

# ReID and team embedding
from torchkick.models.reid import (
    DINOv2ReIDEmbedder,
    SigLIPTeamEmbedder,
)

__all__ = [
    # Pitch
    "DINOv2PitchModel",
    "HeatmapPitchDetector",
    # Player
    "Detection",
    "PlayerDetector",
    # ReID
    "DINOv2ReIDEmbedder",
    "SigLIPTeamEmbedder",
]
