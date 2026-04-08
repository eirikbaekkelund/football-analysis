"""
Tracking module for football video analysis.

Submodules:
    models: Core data models (TrackObservation, TrackData, PlayerSlot)
    trajectory: Trajectory storage and IMM Kalman smoothing
    identity: Team classification and player ID assignment
    homography: Pitch projection via keypoint homography
    team_embedder: Per-game SigLIP team assignment
    pitch_viz: 2D pitch visualization

Example:
    >>> from torchkick.tracking import (
    ...     IdentityAssigner,
    ...     HomographyEstimator,
    ...     GameTeamEmbedder,
    ... )
"""

from __future__ import annotations

# Data models
from torchkick.tracking.models import (
    MAX_PLAYER_SPEED_MS,
    TYPICAL_PLAYER_SPEED_MS,
    HALF_LENGTH,
    HALF_WIDTH,
    TrackObservation,
    TrackData,
    PlayerSlot,
)

# Trajectory management
from torchkick.tracking.trajectory import (
    TrajectoryStore,
    IMMKalmanSmoother,
    TrajectorySmoother,  # backward-compat alias for IMMKalmanSmoother
)

# Identity and team assignment
from torchkick.tracking.identity import (
    IdentityAssigner,
    PitchSlotManager,
)

# Homography and projection
from torchkick.tracking.homography import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    PitchPoint,
    PITCH_LINE_COORDINATES,
    HomographyEstimator,
    KeypointTracker,
)

# Per-game self-supervised team embedder
from torchkick.tracking.team_embedder import GameTeamEmbedder

# Visualization
from torchkick.tracking.pitch_viz import PitchVisualizer

__all__ = [
    # Constants
    "MAX_PLAYER_SPEED_MS",
    "TYPICAL_PLAYER_SPEED_MS",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    # Data models
    "TrackObservation",
    "TrackData",
    "PlayerSlot",
    "PitchPoint",
    # Trajectory
    "TrajectoryStore",
    "IMMKalmanSmoother",
    "TrajectorySmoother",
    # Identity
    "IdentityAssigner",
    "PitchSlotManager",
    # Homography
    "PITCH_LINE_COORDINATES",
    "HomographyEstimator",
    "KeypointTracker",
    # Team embedding
    "GameTeamEmbedder",
    # Visualization
    "PitchVisualizer",
]
