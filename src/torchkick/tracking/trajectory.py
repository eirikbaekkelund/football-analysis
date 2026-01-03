"""
Trajectory storage and analysis.

This module provides the TrajectoryStore for accumulating track
observations and the TrajectorySmootherVelocityConstrained for
applying physics-based smoothing to trajectories.

Example:
    >>> from torchkick.tracking import TrajectoryStore, TrajectorySmoother
    >>> 
    >>> store = TrajectoryStore(fps=30.0)
    >>> store.add_observation(track_id=1, frame_idx=0, box=box, pitch_pos=(0, 0))
    >>> 
    >>> smoother = TrajectorySmoother(fps=30.0)
    >>> smoother.smooth_all(store)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from torchkick.tracking.models import (
    MAX_PLAYER_SPEED_MS,
    TrackData,
    TrackObservation,
)


class TrajectoryStore:
    """
    Storage for all track data with analysis utilities.

    Accumulates observations for multiple tracks and provides
    methods for querying and filtering tracks.

    Args:
        fps: Video frame rate.

    Example:
        >>> store = TrajectoryStore(fps=30.0)
        >>> store.add_observation(1, 0, box, pitch_pos=(0, 0))
        >>> track = store.get_track(1)
    """

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps
        self.tracks: Dict[int, TrackData] = {}
        self.total_frames = 0
        self.frame_homographies: Dict[int, np.ndarray] = {}

    def add_observation(
        self,
        track_id: int,
        frame_idx: int,
        box: np.ndarray,
        pitch_pos: Optional[Tuple[float, float]] = None,
        color_feature: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a single observation for a track.

        Args:
            track_id: Unique track identifier.
            frame_idx: Frame number.
            box: Bounding box [x1, y1, x2, y2].
            pitch_pos: Optional pitch position in meters.
            color_feature: Optional color feature vector.
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackData(track_id=track_id)

        obs = TrackObservation(
            frame_idx=frame_idx,
            box=box,
            pitch_pos=pitch_pos,
            color_feature=color_feature,
        )
        self.tracks[track_id].observations.append(obs)
        self.total_frames = max(self.total_frames, frame_idx + 1)

    def get_track(self, track_id: int) -> Optional[TrackData]:
        """Get track by ID, or None if not found."""
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> List[TrackData]:
        """Get all tracks."""
        return list(self.tracks.values())

    def get_long_tracks(self, min_frames: int = 30) -> List[TrackData]:
        """
        Get tracks with at least min_frames observations.

        Args:
            min_frames: Minimum number of frames required.

        Returns:
            List of qualifying TrackData objects.
        """
        return [t for t in self.tracks.values() if t.duration_frames() >= min_frames]


class TrajectorySmoother:
    """
    Smooth 2D trajectories with physics-based velocity constraints.

    Applies:
    1. Outlier rejection (impossible speed jumps)
    2. Gaussian smoothing for noise reduction
    3. Velocity clamping to physical limits

    Args:
        fps: Video frame rate.
        max_speed_ms: Maximum player speed in m/s.
        smooth_sigma: Gaussian smoothing sigma in frames.
        outlier_threshold_ms: Speed threshold for outlier rejection.

    Example:
        >>> smoother = TrajectorySmoother(fps=30.0)
        >>> smoother.smooth_all(store, min_frames=10)
    """

    def __init__(
        self,
        fps: float = 30.0,
        max_speed_ms: float = MAX_PLAYER_SPEED_MS,
        smooth_sigma: float = 2.0,
        outlier_threshold_ms: float = 15.0,
    ) -> None:
        self.fps = fps
        self.max_speed_ms = max_speed_ms
        self.max_speed_per_frame = max_speed_ms / fps
        self.smooth_sigma = smooth_sigma
        self.outlier_threshold = outlier_threshold_ms / fps

    def _remove_outliers(
        self,
        positions: np.ndarray,
        frames: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove positions representing impossible speed jumps."""
        if len(positions) < 3:
            return positions, frames

        valid_mask = np.ones(len(positions), dtype=bool)

        for i in range(1, len(positions) - 1):
            dt_prev = max(1, frames[i] - frames[i - 1])
            dt_next = max(1, frames[i + 1] - frames[i])

            dx1 = positions[i, 0] - positions[i - 1, 0]
            dy1 = positions[i, 1] - positions[i - 1, 1]
            v1 = np.sqrt(dx1**2 + dy1**2) / dt_prev

            dx2 = positions[i + 1, 0] - positions[i, 0]
            dy2 = positions[i + 1, 1] - positions[i, 1]
            v2 = np.sqrt(dx2**2 + dy2**2) / dt_next

            # If both velocities extreme, likely an outlier
            if v1 > self.outlier_threshold and v2 > self.outlier_threshold:
                valid_mask[i] = False

        return positions[valid_mask], frames[valid_mask]

    def _interpolate_gaps(
        self,
        positions: np.ndarray,
        frames: np.ndarray,
        total_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fill trajectory gaps with linear interpolation."""
        if len(positions) < 2:
            return positions, frames

        min_frame, max_frame = frames[0], frames[-1]
        dense_frames = np.arange(min_frame, max_frame + 1)

        x_interp = np.interp(dense_frames, frames, positions[:, 0])
        y_interp = np.interp(dense_frames, frames, positions[:, 1])

        return np.stack([x_interp, y_interp], axis=1), dense_frames

    def smooth_track(self, track: TrackData, interpolate: bool = True) -> bool:
        """
        Smooth a single track's trajectory.

        Args:
            track: TrackData to smooth in-place.
            interpolate: Whether to interpolate gaps.

        Returns:
            True if successful, False if insufficient data.
        """
        # Extract pitch positions
        positions = []
        frames = []

        for obs in track.observations:
            if obs.pitch_pos is not None:
                positions.append(obs.pitch_pos)
                frames.append(obs.frame_idx)

        if len(positions) < 5:
            return False

        positions = np.array(positions)
        frames = np.array(frames)

        # Step 1: Remove outliers
        positions, frames = self._remove_outliers(positions, frames)
        if len(positions) < 5:
            return False

        # Step 2: Interpolate gaps
        if interpolate and track.observations:
            positions, frames = self._interpolate_gaps(positions, frames, track.observations[-1].frame_idx)

        # Step 3: Gaussian smoothing
        x_smooth = gaussian_filter1d(positions[:, 0], sigma=self.smooth_sigma)
        y_smooth = gaussian_filter1d(positions[:, 1], sigma=self.smooth_sigma)

        # Step 4: Forward velocity clamping
        for i in range(1, len(x_smooth)):
            dt = max(1, frames[i] - frames[i - 1])
            max_dist = self.max_speed_per_frame * dt

            dx = x_smooth[i] - x_smooth[i - 1]
            dy = y_smooth[i] - y_smooth[i - 1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > max_dist:
                scale = max_dist / dist
                x_smooth[i] = x_smooth[i - 1] + dx * scale
                y_smooth[i] = y_smooth[i - 1] + dy * scale

        # Step 5: Backward velocity clamping
        for i in range(len(x_smooth) - 2, -1, -1):
            dt = max(1, frames[i + 1] - frames[i])
            max_dist = self.max_speed_per_frame * dt

            dx = x_smooth[i] - x_smooth[i + 1]
            dy = y_smooth[i] - y_smooth[i + 1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > max_dist:
                scale = max_dist / dist
                x_smooth[i] = x_smooth[i + 1] + dx * scale
                y_smooth[i] = y_smooth[i + 1] + dy * scale

        # Step 6: Final smoothing pass
        x_smooth = gaussian_filter1d(x_smooth, sigma=self.smooth_sigma / 2)
        y_smooth = gaussian_filter1d(y_smooth, sigma=self.smooth_sigma / 2)

        # Store results
        track.smoothed_positions = np.stack([x_smooth, y_smooth], axis=1)
        track.smoothed_frames = frames

        return True

    def smooth_all(self, store: TrajectoryStore, min_frames: int = 10) -> int:
        """
        Smooth all qualifying tracks in the store.

        Args:
            store: TrajectoryStore to process.
            min_frames: Minimum frames required for smoothing.

        Returns:
            Number of successfully smoothed tracks.
        """
        count = 0
        for track in store.tracks.values():
            if track.duration_frames() >= min_frames:
                if self.smooth_track(track):
                    count += 1
        return count


class FallbackProjector:
    """
    Project bounding boxes to pitch when homography fails.

    Uses calibration from frames with valid homography to estimate
    positions from bounding box properties alone.

    Args:
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.

    Example:
        >>> projector = FallbackProjector(1920, 1080)
        >>> projector.add_reference(box, (10.0, -5.0))
        >>> position = projector.project(new_box)
    """

    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Calibration parameters
        self.ref_scale_x: Optional[float] = None
        self.ref_scale_y: Optional[float] = None
        self.ref_offset_x: Optional[float] = None
        self.ref_offset_y: Optional[float] = None

        # Reference points for calibration
        self.reference_points: List[Tuple[np.ndarray, Tuple[float, float]]] = []

    def add_reference(
        self,
        box: np.ndarray,
        pitch_pos: Tuple[float, float],
    ) -> None:
        """
        Add a bbox -> pitch_pos correspondence for calibration.

        Args:
            box: Bounding box [x1, y1, x2, y2].
            pitch_pos: Known pitch position in meters.
        """
        self.reference_points.append((box.copy(), pitch_pos))

        if len(self.reference_points) >= 50:
            self._calibrate()

    def _calibrate(self) -> None:
        """Compute calibration from reference points."""
        if len(self.reference_points) < 20:
            return

        bboxes = []
        pitches = []
        for box, pitch in self.reference_points[-200:]:
            bboxes.append(box)
            pitches.append(pitch)

        bboxes = np.array(bboxes)
        pitches = np.array(pitches)

        # X: center_x -> pitch_x
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cx_norm = cx / self.image_width

        A_x = np.column_stack([cx_norm, np.ones_like(cx_norm)])
        result = np.linalg.lstsq(A_x, pitches[:, 0], rcond=None)
        if len(result[0]) >= 2:
            self.ref_scale_x = result[0][0]
            self.ref_offset_x = result[0][1]

        # Y: bottom_y -> pitch_y
        bottom = bboxes[:, 3] / self.image_height

        A_y = np.column_stack([bottom, np.ones_like(bottom)])
        result = np.linalg.lstsq(A_y, pitches[:, 1], rcond=None)
        if len(result[0]) >= 2:
            self.ref_scale_y = result[0][0]
            self.ref_offset_y = result[0][1]

    def project(self, box: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Project bounding box to pitch coordinates.

        Args:
            box: Bounding box [x1, y1, x2, y2].

        Returns:
            (x, y) pitch position, or None if not calibrated.
        """
        if self.ref_scale_x is None or self.ref_scale_y is None:
            return None

        cx = (box[0] + box[2]) / 2
        bottom = box[3]

        cx_norm = cx / self.image_width
        bottom_norm = bottom / self.image_height

        pitch_x = self.ref_scale_x * cx_norm + self.ref_offset_x
        pitch_y = self.ref_scale_y * bottom_norm + self.ref_offset_y

        pitch_x = np.clip(pitch_x, -self.pitch_length / 2, self.pitch_length / 2)
        pitch_y = np.clip(pitch_y, -self.pitch_width / 2, self.pitch_width / 2)

        return (float(pitch_x), float(pitch_y))

    def is_calibrated(self) -> bool:
        """Check if projector has been calibrated."""
        return self.ref_scale_x is not None and self.ref_scale_y is not None


__all__ = [
    "TrajectoryStore",
    "TrajectorySmoother",
    "FallbackProjector",
]
