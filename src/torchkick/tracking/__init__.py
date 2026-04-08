"""
Tracking module for football video analysis.

This module provides comprehensive player and ball tracking capabilities,
including BoT-SORT multi-object tracking, Kalman filtering, team classification
via ReID embeddings, identity assignment, and pitch projection.

Submodules:
    models: Core data models (TrackObservation, TrackData, PlayerSlot, BallState)
    trajectory: Trajectory storage and smoothing
    iou_tracker: MaskIoUTracker (SAM3 mask tracking only)
    identity: Team classification and player ID assignment
    ball: Ball tracking with Kalman filtering
    homography: Pitch projection via keypoint homography
    pitch_viz: 2D pitch visualization

Example:
    >>> from torchkick.tracking import (
    ...     SoccerTracker,
    ...     IdentityAssigner,
    ...     HomographyEstimator,
    ...     BallTracker,
    ... )
    >>>
    >>> tracker = SoccerTracker()
    >>> homography = HomographyEstimator()
    >>> ball_tracker = BallTracker()
"""

from __future__ import annotations

# Data models
from torchkick.tracking.models import (
    MAX_PLAYER_SPEED_MS,
    TYPICAL_PLAYER_SPEED_MS,
    PENALTY_AREA_X,
    HALF_LENGTH,
    HALF_WIDTH,
    TrackObservation,
    TrackData,
    PlayerSlot,
    BallState,
)

# Trajectory management
from torchkick.tracking.trajectory import (
    TrajectoryStore,
    IMMKalmanSmoother,
    TrajectorySmoother,  # backward-compat alias for IMMKalmanSmoother
)

# SAM3 mask-based tracking only (SimpleIoUTracker removed — use SoccerTracker)
from torchkick.tracking.iou_tracker import MaskIoUTracker

# BoT-SORT multi-object tracker (primary MOT); ByteTracker is a compat alias
from torchkick.tracking.mot_tracker import SoccerTracker, ByteTracker

# Identity and team assignment
from torchkick.tracking.identity import (
    IdentityAssigner,
    PitchSlotManager,
)

# Ball tracking
from torchkick.tracking.ball import (
    BallKalmanTrack,
    BallTracker,
)

# Homography and projection
from torchkick.tracking.homography import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    PitchPoint,
    PITCH_LINE_COORDINATES,
    CameraPoseKalmanFilter,
    HomographyKalmanFilter,  # backward-compat alias
    HomographyEstimator,
    KeypointTracker,
    GeometricConstraintSolver,
)

# 3D pose lifting
from torchkick.tracking.lifting import (
    CameraModel,
    build_camera_model,
    PoseLift3D,
)

# Per-game self-supervised team embedder
from torchkick.tracking.team_embedder import GameTeamEmbedder

# Visualization
from torchkick.tracking.pitch_viz import (
    COLOR_TEAM_1,
    COLOR_TEAM_2,
    COLOR_REFEREE,
    PitchVisualizer,
)

__all__ = [
    # Constants
    "MAX_PLAYER_SPEED_MS",
    "TYPICAL_PLAYER_SPEED_MS",
    "PENALTY_AREA_X",
    "HALF_LENGTH",
    "HALF_WIDTH",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "COLOR_TEAM_1",
    "COLOR_TEAM_2",
    "COLOR_REFEREE",
    # Data models
    "TrackObservation",
    "TrackData",
    "PlayerSlot",
    "BallState",
    "PitchPoint",
    # Trajectory
    "TrajectoryStore",
    "IMMKalmanSmoother",
    "TrajectorySmoother",
    # Tracking
    "SoccerTracker",
    "ByteTracker",
    "MaskIoUTracker",
    "IdentityAssigner",
    "PitchSlotManager",
    "BallKalmanTrack",
    "BallTracker",
    # Homography
    "PITCH_LINE_COORDINATES",
    "CameraPoseKalmanFilter",
    "HomographyKalmanFilter",
    "HomographyEstimator",
    "KeypointTracker",
    "GeometricConstraintSolver",
    # 3D lifting
    "CameraModel",
    "build_camera_model",
    "PoseLift3D",
    # Team embedding
    "GameTeamEmbedder",
    # Visualization
    "PitchVisualizer",
]
