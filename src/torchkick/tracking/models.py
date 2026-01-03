"""
Data models for tracking system.

This module defines the core data structures used throughout the
tracking pipeline, including track observations, trajectory data,
and player slot assignments.

Example:
    >>> from torchkick.tracking import TrackObservation, TrackData
    >>> 
    >>> obs = TrackObservation(
    ...     frame_idx=0,
    ...     box=np.array([100, 200, 150, 300]),
    ...     pitch_pos=(10.5, -5.2),
    ... )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# Physical constants for football pitch
MAX_PLAYER_SPEED_MS = 12.0  # m/s (generous for sprints)
TYPICAL_PLAYER_SPEED_MS = 6.0  # m/s
PENALTY_AREA_X = 52.5 - 16.5  # 36m from center = inside penalty area
HALF_LENGTH = 52.5  # m
HALF_WIDTH = 34.0  # m


class TrackObservation(BaseModel):
    """
    Single observation of a tracked object in one frame.

    Stores bounding box, optional pitch position, and color features
    for team classification.

    Attributes:
        frame_idx: Frame number in the video.
        box: Bounding box as [x1, y1, x2, y2] in pixels.
        pitch_pos: Optional (x, y) position on pitch in meters.
        color_feature: Optional color feature vector for team classification.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_idx: int
    box: np.ndarray
    pitch_pos: Optional[Tuple[float, float]] = None
    color_feature: Optional[np.ndarray] = None


class TrackData(BaseModel):
    """
    Complete tracking data for a single detected object.

    Accumulates observations over time and stores derived properties
    like smoothed positions, team assignment, and player role.

    Attributes:
        track_id: Unique identifier for this track.
        observations: List of frame-by-frame observations.
        smoothed_positions: Optional smoothed trajectory array.
        smoothed_frames: Frame indices for smoothed positions.
        role: Detected role (player, goalie, referee, linesman).
        team: Team index (0 or 1) or -1 for referee.
        player_id: Jersey number if detected.
        frame_team_votes: Per-frame team classification votes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    track_id: int
    observations: List[TrackObservation] = Field(default_factory=list)
    smoothed_positions: Optional[np.ndarray] = None
    smoothed_frames: Optional[np.ndarray] = None
    role: Optional[str] = None
    team: Optional[int] = None
    player_id: Optional[int] = None
    frame_team_votes: Optional[List[int]] = None

    def duration_frames(self) -> int:
        """Number of frames this track has been observed."""
        return len(self.observations)

    def duration_seconds(self, fps: float) -> float:
        """Duration of track in seconds."""
        return self.duration_frames() / fps

    def mean_pitch_position(self) -> Optional[Tuple[float, float]]:
        """
        Average position on pitch across all observations.

        Returns:
            (x, y) in meters, or None if no pitch positions available.
        """
        positions = [o.pitch_pos for o in self.observations if o.pitch_pos is not None]
        if not positions:
            return None
        arr = np.array(positions)
        return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))

    def pitch_position_stats(self) -> Dict[str, float]:
        """
        Compute statistics for role inference.

        Returns:
            Dict with mean, std, min, max for x and y positions.
        """
        positions = [o.pitch_pos for o in self.observations if o.pitch_pos is not None]
        if not positions:
            return {}
        arr = np.array(positions)
        return {
            "mean_x": float(arr[:, 0].mean()),
            "mean_y": float(arr[:, 1].mean()),
            "std_x": float(arr[:, 0].std()),
            "std_y": float(arr[:, 1].std()),
            "min_x": float(arr[:, 0].min()),
            "max_x": float(arr[:, 0].max()),
            "min_y": float(arr[:, 1].min()),
            "max_y": float(arr[:, 1].max()),
            "n_samples": len(positions),
        }


class PlayerSlot(BaseModel):
    """
    Fixed slot for a player in the 11v11 formation.

    Used for consistent identity assignment by mapping fragmented
    tracks to persistent player slots.

    Attributes:
        slot_id: Slot index (0-10 for field players).
        team: Team index (0, 1, or -1 for referee).
        position: Current (x, y) position on pitch.
        last_observed_frame: Last frame this slot was updated.
        assigned_track_ids: All track IDs assigned to this slot.
        color_feature: Average color for this slot.
        is_goalie: Whether this slot is the goalkeeper.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    slot_id: int
    team: int
    position: Tuple[float, float] = (0.0, 0.0)
    last_observed_frame: int = -1
    assigned_track_ids: List[int] = Field(default_factory=list)
    color_feature: Optional[np.ndarray] = None
    is_goalie: bool = False


class BallState(BaseModel):
    """
    Current state of the tracked ball.

    Attributes:
        position: (x, y) position in image pixels.
        velocity: (vx, vy) velocity in pixels/frame.
        confidence: Detection confidence score.
        is_visible: Whether ball is currently detected.
        in_play: Whether ball is in play (not out of bounds).
        track_id: Ball track identifier.
    """

    position: Tuple[float, float]
    velocity: Tuple[float, float]
    confidence: float
    is_visible: bool
    in_play: bool
    track_id: int


__all__ = [
    "MAX_PLAYER_SPEED_MS",
    "TYPICAL_PLAYER_SPEED_MS",
    "PENALTY_AREA_X",
    "HALF_LENGTH",
    "HALF_WIDTH",
    "TrackObservation",
    "TrackData",
    "PlayerSlot",
    "BallState",
]
