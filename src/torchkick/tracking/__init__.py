"""
Tracking module for football video analysis.

This module provides comprehensive player and ball tracking capabilities,
including IoU-based detection linking, Kalman filtering, team classification,
identity assignment, and pitch projection.

Submodules:
    models: Core data models (TrackObservation, TrackData, PlayerSlot, BallState)
    trajectory: Trajectory storage and smoothing
    iou_tracker: Simple IoU-based detection linking
    identity: Team classification and player ID assignment
    ball: Ball tracking with Kalman filtering
    homography: Pitch projection via keypoint homography
    pitch_viz: 2D pitch visualization
    color_features: Jersey color extraction

Example:
    >>> from torchkick.tracking import (
    ...     SimpleIoUTracker,
    ...     IdentityAssigner,
    ...     HomographyEstimator,
    ...     BallTracker,
    ... )
    >>> 
    >>> # Initialize components
    >>> tracker = SimpleIoUTracker(iou_threshold=0.3)
    >>> identity = IdentityAssigner(n_teams=2)
    >>> homography = HomographyEstimator()
    >>> ball_tracker = BallTracker()
    >>> 
    >>> # Process frame
    >>> track_ids = tracker.update(detections)
    >>> identity.assign_teams(features, track_ids)
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
    TrajectorySmoother,
    FallbackProjector,
)

# Detection linking
from torchkick.tracking.iou_tracker import SimpleIoUTracker

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
    HomographyEstimator,
)

# Visualization
from torchkick.tracking.pitch_viz import (
    COLOR_TEAM_1,
    COLOR_TEAM_2,
    COLOR_REFEREE,
    PitchVisualizer,
)

# Color features
from torchkick.tracking.color_features import (
    get_jersey_color_feature,
    get_dominant_color_feature,
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
    "TrajectorySmoother",
    "FallbackProjector",
    # Tracking
    "SimpleIoUTracker",
    "IdentityAssigner",
    "PitchSlotManager",
    "BallKalmanTrack",
    "BallTracker",
    # Homography
    "PITCH_LINE_COORDINATES",
    "HomographyEstimator",
    # Visualization
    "PitchVisualizer",
    # Features
    "get_jersey_color_feature",
    "get_dominant_color_feature",
]
