"""
Ball tracking with Kalman filtering.

This module provides specialized tracking for the ball using
Kalman filtering for smooth trajectory prediction and handling
occlusions.

Example:
    >>> from torchkick.tracking import BallTracker
    >>> 
    >>> tracker = BallTracker(max_age=30)
    >>> state = tracker.update(detections, player_boxes)
    >>> if state and state.is_visible:
    ...     print(f"Ball at {state.position}")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from torchkick.tracking.models import BallState


class BallKalmanTrack:
    """
    Kalman filter-based ball track.

    Uses a constant velocity motion model with position and
    velocity state.

    Args:
        position: Initial (x, y) position.
        track_id: Unique track identifier.
    """

    def __init__(self, position: Tuple[float, float], track_id: int) -> None:
        self.id = track_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.confidence = 1.0

        # Initialize Kalman filter
        self.kf = cv2.KalmanFilter(4, 2)

        # State: [x, y, vx, vy]
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )

        # Measurement: [x, y]
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            np.float32,
        )

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.kf.statePost = np.array(
            [[position[0]], [position[1]], [0], [0]],
            np.float32,
        )

    def predict(self) -> Tuple[float, float]:
        """Predict next position and return it."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_position()

    def update(self, position: Tuple[float, float], confidence: float) -> None:
        """Update with new measurement."""
        measurement = np.array([[position[0]], [position[1]]], np.float32)
        self.kf.correct(measurement)
        self.hits += 1
        self.time_since_update = 0
        self.confidence = 0.8 * self.confidence + 0.2 * confidence

    def get_position(self) -> Tuple[float, float]:
        """Get current position estimate."""
        return (
            float(self.kf.statePost[0].item()),
            float(self.kf.statePost[1].item()),
        )

    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return (
            float(self.kf.statePost[2].item()),
            float(self.kf.statePost[3].item()),
        )


class BallTracker:
    """
    Multi-hypothesis ball tracker with occlusion handling.

    Maintains multiple track hypotheses and uses Kalman filtering
    for smooth position estimation. Handles occlusions by players.

    Args:
        max_age: Frames to keep undetected tracks alive.
        min_hits: Minimum detections before track is confirmed.
        distance_threshold: Maximum distance for matching.
        velocity_threshold: Maximum velocity change.
        occlusion_handler: Enable physics-based occlusion prediction.

    Example:
        >>> tracker = BallTracker(max_age=30)
        >>> for frame_idx, (ball_dets, player_boxes) in enumerate(video):
        ...     state = tracker.update(ball_dets, player_boxes)
        ...     if state:
        ...         draw_ball(frame, state.position)
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        distance_threshold: float = 50.0,
        velocity_threshold: float = 100.0,
        occlusion_handler: bool = True,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.velocity_threshold = velocity_threshold
        self.occlusion_handler = occlusion_handler

        self.tracks: List[BallKalmanTrack] = []
        self.next_id = 1
        self.frame_id = 0
        self.primary_track_id: Optional[int] = None

        self.trajectory_buffer: List[Tuple[float, float]] = []
        self.trajectory_max_len = 60

        self.occlusion_frames = 0
        self.last_known_velocity = (0.0, 0.0)

    def _euclidean_distance(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _validate_detection(
        self,
        detection: Tuple[float, float, float, float, float],
        player_boxes: List[List[float]],
    ) -> bool:
        """
        Validate ball detection based on size, shape, and player overlap.

        Args:
            detection: (x1, y1, x2, y2, confidence).
            player_boxes: List of player bounding boxes.

        Returns:
            True if detection is valid.
        """
        x1, y1, x2, y2, conf = detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        # Size constraints
        if w > 50 or h > 50:
            return False

        # Aspect ratio check (ball should be roughly circular)
        aspect_ratio = w / max(h, 1)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False

        # Check player overlap
        for pbox in player_boxes:
            px1, py1, px2, py2 = pbox[:4]
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                player_h = py2 - py1
                # Allow if at top (header) or bottom (feet) of player
                if cy < py1 + player_h * 0.3 or cy > py1 + player_h * 0.9:
                    return True
                return False

        return True

    def _predict_occlusion_position(self) -> Optional[Tuple[float, float]]:
        """Predict ball position during occlusion using physics."""
        if not self.trajectory_buffer or len(self.trajectory_buffer) < 2:
            return None

        vx, vy = self.last_known_velocity
        last_pos = self.trajectory_buffer[-1]

        # Simple physics with gravity
        gravity_effect = 0.5 * self.occlusion_frames
        predicted_y = last_pos[1] + vy * self.occlusion_frames + gravity_effect
        predicted_x = last_pos[0] + vx * self.occlusion_frames

        return (predicted_x, predicted_y)

    def update(
        self,
        detections: List[Tuple[float, float, float, float, float]],
        player_boxes: Optional[List[List[float]]] = None,
    ) -> Optional[BallState]:
        """
        Update tracker with new detections.

        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples.
            player_boxes: Optional list of player bounding boxes.

        Returns:
            BallState if ball is tracked, None otherwise.
        """
        self.frame_id += 1
        player_boxes = player_boxes or []

        # Predict all tracks
        for track in self.tracks:
            track.predict()

        # Filter valid detections
        valid_detections = []
        for det in detections:
            if self._validate_detection(det, player_boxes):
                x1, y1, x2, y2, conf = det
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                valid_detections.append((cx, cy, conf))

        # Match to primary track first
        matched_tracks = set()
        matched_dets = set()

        if self.primary_track_id is not None:
            primary_track = next(
                (t for t in self.tracks if t.id == self.primary_track_id),
                None,
            )
            if primary_track and valid_detections:
                pred_pos = primary_track.get_position()
                pred_vel = primary_track.get_velocity()

                best_det_idx = None
                best_score = float("inf")

                for i, (cx, cy, conf) in enumerate(valid_detections):
                    dist = self._euclidean_distance(pred_pos, (cx, cy))

                    expected_x = pred_pos[0] + pred_vel[0]
                    expected_y = pred_pos[1] + pred_vel[1]
                    vel_dist = self._euclidean_distance((expected_x, expected_y), (cx, cy))

                    score = dist + 0.3 * vel_dist - 10 * conf

                    if dist < self.distance_threshold * 2 and score < best_score:
                        best_score = score
                        best_det_idx = i

                if best_det_idx is not None:
                    cx, cy, conf = valid_detections[best_det_idx]
                    primary_track.update((cx, cy), conf)
                    matched_tracks.add(primary_track.id)
                    matched_dets.add(best_det_idx)
                    self.occlusion_frames = 0

        # Match remaining tracks
        for track in self.tracks:
            if track.id in matched_tracks:
                continue

            pred_pos = track.get_position()
            best_det_idx = None
            best_dist = self.distance_threshold

            for i, (cx, cy, conf) in enumerate(valid_detections):
                if i in matched_dets:
                    continue
                dist = self._euclidean_distance(pred_pos, (cx, cy))
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = i

            if best_det_idx is not None:
                cx, cy, conf = valid_detections[best_det_idx]
                track.update((cx, cy), conf)
                matched_tracks.add(track.id)
                matched_dets.add(best_det_idx)

        # Create new tracks for unmatched detections
        for i, (cx, cy, conf) in enumerate(valid_detections):
            if i in matched_dets:
                continue
            new_track = BallKalmanTrack((cx, cy), self.next_id)
            self.next_id += 1
            self.tracks.append(new_track)

        # Prune old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Select primary track
        if self.tracks:
            confirmed = [t for t in self.tracks if t.hits >= self.min_hits]
            if confirmed:
                primary = max(confirmed, key=lambda t: t.hits)
                self.primary_track_id = primary.id

                pos = primary.get_position()
                vel = primary.get_velocity()

                # Update trajectory buffer
                self.trajectory_buffer.append(pos)
                if len(self.trajectory_buffer) > self.trajectory_max_len:
                    self.trajectory_buffer.pop(0)

                self.last_known_velocity = vel

                return BallState(
                    position=pos,
                    velocity=vel,
                    confidence=primary.confidence,
                    is_visible=primary.time_since_update == 0,
                    in_play=True,
                    track_id=primary.id,
                )

        # Handle occlusion
        if self.occlusion_handler and self.trajectory_buffer:
            self.occlusion_frames += 1
            predicted = self._predict_occlusion_position()
            if predicted:
                return BallState(
                    position=predicted,
                    velocity=self.last_known_velocity,
                    confidence=0.3,
                    is_visible=False,
                    in_play=True,
                    track_id=-1,
                )

        return None

    def reset(self) -> None:
        """Clear all tracks and reset state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_id = 0
        self.primary_track_id = None
        self.trajectory_buffer.clear()
        self.occlusion_frames = 0


__all__ = [
    "BallKalmanTrack",
    "BallTracker",
]
