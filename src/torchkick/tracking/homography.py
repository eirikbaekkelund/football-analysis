"""
Homography estimation for pitch projection.

This module provides tools for estimating homography transforms
from detected pitch line keypoints, enabling projection between
image pixels and pitch coordinates.

Example:
    >>> from torchkick.tracking import HomographyEstimator
    >>> 
    >>> estimator = HomographyEstimator()
    >>> success = estimator.estimate(keypoints, visibility, confidence, (1080, 1920))
    >>> if success:
    ...     pitch_pos = estimator.project_player_to_pitch(player_bbox)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Pitch dimensions in meters
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2  # 52.5m
HALF_WIDTH = PITCH_WIDTH / 2  # 34m
PENALTY_AREA_WIDTH = 40.32
PENALTY_AREA_DEPTH = 16.5
GOAL_AREA_WIDTH = 18.32
GOAL_AREA_DEPTH = 5.5
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DISTANCE = 11.0
GOAL_WIDTH = 7.32


class PitchPoint:
    """Lightweight pitch coordinate container."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y], dtype=np.float32)


def _get_circle_points(
    center_x: float,
    center_y: float,
    radius: float,
    n_points: int = 9,
) -> List[Tuple[int, PitchPoint]]:
    """Generate evenly spaced points around a circle."""
    points = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append((i, PitchPoint(x=x, y=y)))
    return points


# Line coordinates for homography estimation
PITCH_LINE_COORDINATES: Dict[str, List[Tuple[int, PitchPoint]]] = {
    "Side line top": [
        (0, PitchPoint(x=-HALF_LENGTH, y=HALF_WIDTH)),
        (1, PitchPoint(x=HALF_LENGTH, y=HALF_WIDTH)),
    ],
    "Side line bottom": [
        (0, PitchPoint(x=-HALF_LENGTH, y=-HALF_WIDTH)),
        (1, PitchPoint(x=HALF_LENGTH, y=-HALF_WIDTH)),
    ],
    "Side line left": [
        (0, PitchPoint(x=-HALF_LENGTH, y=-HALF_WIDTH)),
        (1, PitchPoint(x=-HALF_LENGTH, y=HALF_WIDTH)),
    ],
    "Side line right": [
        (0, PitchPoint(x=HALF_LENGTH, y=-HALF_WIDTH)),
        (1, PitchPoint(x=HALF_LENGTH, y=HALF_WIDTH)),
    ],
    "Middle line": [
        (0, PitchPoint(x=0, y=-HALF_WIDTH)),
        (1, PitchPoint(x=0, y=HALF_WIDTH)),
    ],
    "Big rect. left main": [
        (0, PitchPoint(x=-HALF_LENGTH + PENALTY_AREA_DEPTH, y=-PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(x=-HALF_LENGTH + PENALTY_AREA_DEPTH, y=PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. right main": [
        (0, PitchPoint(x=HALF_LENGTH - PENALTY_AREA_DEPTH, y=-PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(x=HALF_LENGTH - PENALTY_AREA_DEPTH, y=PENALTY_AREA_WIDTH / 2)),
    ],
}

# Add circles
PITCH_LINE_COORDINATES["Circle central"] = _get_circle_points(0, 0, CENTER_CIRCLE_RADIUS)


# 30-keypoint pitch layout (0-indexed).
# Converted to center-origin meters to match the existing HomographyEstimator coordinate system
# (x: -52.5 … +52.5 along length, y: -34 … +34 along width).
# Raw dimensions: 120m × 70m, origin at top-left corner.
_RF_L = 120.0  # pitch length m
_RF_W = 70.0  # pitch width m
_RF_PBW = 41.0  # penalty box width m
_RF_PBD = 20.15  # penalty box depth m
_RF_GBW = 18.32  # goal box width m
_RF_GBD = 5.50  # goal box depth m
_RF_CCR = 9.15  # centre circle radius m
_RF_PS = 11.0  # penalty spot distance m
_RF_HX = _RF_L / 2  # 60.0
_RF_HY = _RF_W / 2  # 35.0


def _rf(x_raw: float, y_raw: float) -> Tuple[float, float]:
    """Convert corner-origin coords to center-origin meters.

    Normalises from the 120×70m template to the codebase-standard
    105×68m coordinate system (HALF_LENGTH=52.5, HALF_WIDTH=34) so that
    pitch_viz renders correctly.
    """
    x_scaled = (x_raw / _RF_L) * PITCH_LENGTH - HALF_LENGTH
    y_scaled = (y_raw / _RF_W) * PITCH_WIDTH - HALF_WIDTH
    return (x_scaled, y_scaled)


VERTICES: List[Tuple[float, float]] = [
    _rf(0, 0),  # 0  top-left corner
    _rf(0, (_RF_W - _RF_PBW) / 2),  # 1  left penalty box top
    _rf(0, (_RF_W - _RF_GBW) / 2),  # 2  left goal box top
    _rf(0, (_RF_W + _RF_GBW) / 2),  # 3  left goal box bottom
    _rf(0, (_RF_W + _RF_PBW) / 2),  # 4  left penalty box bottom
    _rf(0, _RF_W),  # 5  bottom-left corner
    _rf(_RF_GBD, (_RF_W - _RF_GBW) / 2),  # 6  left goal box front top
    _rf(_RF_GBD, (_RF_W + _RF_GBW) / 2),  # 7  left goal box front bottom
    _rf(_RF_PBD, (_RF_W - _RF_PBW) / 2),  # 8  left penalty box front top
    _rf(_RF_PBD, (_RF_W - _RF_GBW) / 2),  # 9  left penalty box inner top
    _rf(_RF_PBD, (_RF_W + _RF_GBW) / 2),  # 10 left penalty box inner bottom
    _rf(_RF_PBD, (_RF_W + _RF_PBW) / 2),  # 11 left penalty box front bottom
    _rf(_RF_HX, 0),  # 12 halfway line top
    _rf(_RF_HX, _RF_HY - _RF_CCR),  # 13 centre circle top
    _rf(_RF_HX, _RF_HY + _RF_CCR),  # 14 centre circle bottom
    _rf(_RF_HX, _RF_W),  # 15 halfway line bottom
    _rf(_RF_L - _RF_PBD, (_RF_W - _RF_PBW) / 2),  # 16 right penalty box front top
    _rf(_RF_L - _RF_PBD, (_RF_W - _RF_GBW) / 2),  # 17 right penalty box inner top
    _rf(_RF_L - _RF_PBD, (_RF_W + _RF_GBW) / 2),  # 18 right penalty box inner bottom
    _rf(_RF_L - _RF_PBD, (_RF_W + _RF_PBW) / 2),  # 19 right penalty box front bottom
    _rf(_RF_L - _RF_GBD, (_RF_W - _RF_GBW) / 2),  # 20 right goal box front top
    _rf(_RF_L - _RF_GBD, (_RF_W + _RF_GBW) / 2),  # 21 right goal box front bottom
    _rf(_RF_L, 0),  # 22 top-right corner
    _rf(_RF_L, (_RF_W - _RF_PBW) / 2),  # 23 right penalty box top
    _rf(_RF_L, (_RF_W - _RF_GBW) / 2),  # 24 right goal box top
    _rf(_RF_L, (_RF_W + _RF_GBW) / 2),  # 25 right goal box bottom
    _rf(_RF_L, (_RF_W + _RF_PBW) / 2),  # 26 right penalty box bottom
    _rf(_RF_L, _RF_W),  # 27 bottom-right corner
    _rf(_RF_HX - _RF_CCR, _RF_HY),  # 28 centre circle left
    _rf(_RF_HX + _RF_CCR, _RF_HY),  # 29 centre circle right
]


class CameraPoseKalmanFilter:
    """
    8-DoF Kalman filter in camera pose space [rvec(3), log_scale, velocities(4)].

    Filters broadcast camera rotation (Rodrigues vector) and zoom (log-scale)
    rather than raw homography elements.  Advantages over the 18-DoF raw-H filter:

      1. Bounded state — rotation angles are physically limited (~±90° for broadcast)
      2. Geometrically correct interpolation — Rodrigues interpolation is correct for SO(3)
      3. Smaller state dimension → better-conditioned Kalman gain during long prediction gaps
      4. Scale stays positive (log-space prevents inverted-scale predictions during occlusion)

    The decomposition approximates K from image dimensions (assuming ~60° FOV) and
    decomposes H = K * [r1*s | r2*s | t] to extract rotation and scale.  The translation
    column is stored verbatim (it changes slowly for a fixed-position broadcast camera)
    and used for reconstruction.

    Args:
        process_noise_rot: Process noise for rotation components (rad per frame).
        process_noise_scale: Process noise for log-scale component.
        measurement_noise_rot: Measurement noise for rotation (rad).
        measurement_noise_scale: Measurement noise for log-scale.

    Example:
        >>> kf = CameraPoseKalmanFilter()
        >>> H_smooth = kf.update(H_raw, image_size=(1080, 1920))
        >>> H_pred = kf.predict()
    """

    STATE_DIM = 8  # [r1, r2, r3, log_s,  dr1, dr2, dr3, dlog_s]
    MEASURE_DIM = 4  # [r1, r2, r3, log_s]

    def __init__(
        self,
        process_noise_rot: float = 5e-4,
        process_noise_scale: float = 1e-4,
        measurement_noise_rot: float = 1e-2,
        measurement_noise_scale: float = 5e-3,
    ) -> None:
        self._kf = cv2.KalmanFilter(self.STATE_DIM, self.MEASURE_DIM)

        # Constant-velocity transition
        F = np.eye(self.STATE_DIM, dtype=np.float32)
        F[:4, 4:] = np.eye(4, dtype=np.float32)
        self._kf.transitionMatrix = F

        # Observe position components only
        H_m = np.zeros((self.MEASURE_DIM, self.STATE_DIM), dtype=np.float32)
        H_m[:4, :4] = np.eye(4, dtype=np.float32)
        self._kf.measurementMatrix = H_m

        Q = np.zeros(self.STATE_DIM, dtype=np.float32)
        Q[:3] = process_noise_rot
        Q[3] = process_noise_scale
        Q[4:7] = process_noise_rot * 2.0
        Q[7] = process_noise_scale * 2.0
        self._kf.processNoiseCov = np.diag(Q)

        R_m = np.zeros(self.MEASURE_DIM, dtype=np.float32)
        R_m[:3] = measurement_noise_rot
        R_m[3] = measurement_noise_scale
        self._kf.measurementNoiseCov = np.diag(R_m)
        self._kf.errorCovPost = np.eye(self.STATE_DIM, dtype=np.float32)

        self._initialized = False
        self._last_K: Optional[np.ndarray] = None
        self._last_translation: Optional[np.ndarray] = None

    @staticmethod
    def _approx_K(image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        f = float(max(h, w))
        return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)

    def _H_to_pose(
        self,
        H: np.ndarray,
        image_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Decompose H → (rvec[3], log_scale, K, translation[3])."""
        H_n = H.astype(np.float64) / (abs(float(H[2, 2])) + 1e-8)
        K = self._approx_K(image_size)
        M = np.linalg.inv(K) @ H_n  # ≈ [r1*s, r2*s, t]

        s1 = np.linalg.norm(M[:, 0])
        s2 = np.linalg.norm(M[:, 1])
        scale = (s1 + s2) / 2.0
        log_scale = np.log(max(scale, 1e-6))

        r1 = M[:, 0] / (s1 + 1e-8)
        r2 = M[:, 1] / (s2 + 1e-8)
        r3 = np.cross(r1, r2)
        U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
        if np.linalg.det(U @ Vt) < 0:
            U[:, -1] *= -1
        R_clean = U @ Vt

        rvec, _ = cv2.Rodrigues(R_clean.astype(np.float32))
        return rvec.flatten().astype(np.float64), log_scale, K, M[:, 2]

    def _pose_to_H(
        self,
        rvec: np.ndarray,
        log_scale: float,
        K: np.ndarray,
        translation: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Reconstruct H from (rvec, log_scale, K, translation)."""
        scale = np.exp(float(log_scale))
        R, _ = cv2.Rodrigues(rvec.astype(np.float32))
        M = np.column_stack([R[:, 0] * scale, R[:, 1] * scale, translation])
        H = K @ M
        denom = abs(float(H[2, 2]))
        if denom < 1e-8:
            return None
        return H / denom

    def predict(self) -> Optional[np.ndarray]:
        """Predict next homography from Kalman state."""
        if not self._initialized or self._last_K is None:
            return None
        predicted = self._kf.predict()
        rvec = predicted[:3, 0].astype(np.float64)
        log_s = float(predicted[3, 0])
        return self._pose_to_H(rvec, log_s, self._last_K, self._last_translation)

    def update(self, H_raw: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Update Kalman filter with a new homography measurement.

        Args:
            H_raw: 3x3 homography matrix from RANSAC.
            image_size: (height, width) of the source frame.

        Returns:
            Smoothed 3x3 homography, or H_raw on decomposition failure.
        """
        try:
            rvec, log_s, K, translation = self._H_to_pose(H_raw, image_size)
        except Exception:
            return H_raw

        measurement = np.array([rvec[0], rvec[1], rvec[2], log_s], dtype=np.float32).reshape(self.MEASURE_DIM, 1)

        if not self._initialized:
            self._kf.statePost = np.zeros((self.STATE_DIM, 1), dtype=np.float32)
            self._kf.statePost[:4, 0] = measurement[:, 0]
            self._initialized = True

        self._kf.predict()
        corrected = self._kf.correct(measurement)

        rvec_f = corrected[:3, 0].astype(np.float64)
        log_s_f = float(corrected[3, 0])

        self._last_K = K
        self._last_translation = translation

        H_smooth = self._pose_to_H(rvec_f, log_s_f, K, translation)
        return H_smooth if H_smooth is not None else H_raw

    def reset(self) -> None:
        """Reset Kalman filter state."""
        self._initialized = False
        self._kf.errorCovPost = np.eye(self.STATE_DIM, dtype=np.float32)
        self._last_K = None
        self._last_translation = None


# Backward-compatible alias
HomographyKalmanFilter = CameraPoseKalmanFilter


# Geometric consistency constraint pairs for the 30-keypoint pitch schema.
#
# Keypoint index reference:
#  0=TL-corner  1=L-pen-top   2=L-goal-top   3=L-goal-bot   4=L-pen-bot
#  5=BL-corner  6=L-goal-front-top  7=L-goal-front-bot  8=L-pen-spot
#  9=L-pen-front-top  10=L-pen-inner-top  11=L-pen-inner-bot  12=L-pen-front-bot
#  13=HW-top  14=CC-top  15=CC-bot  16=HW-bot
#  17=R-pen-front-top  18=R-pen-inner-top  19=R-pen-inner-bot  20=R-pen-front-bot
#  21=R-pen-spot  22=R-goal-front-top  23=R-goal-front-bot
#  24=TR-corner  25=R-pen-top  26=R-goal-top  27=R-goal-bot  28=R-pen-bot
#  29=BR-corner  30=CC-left  31=CC-right
#
# Each entry (a, b) asserts that keypoint[a].x < keypoint[b].x (a is left of b).
_HORIZ_PAIRS: List[Tuple[int, int]] = [
    # Corners & halfway line
    (0, 24),  # TL < TR
    (5, 29),  # BL < BR
    (0, 13),  # TL < HW-top
    (13, 24),  # HW-top < TR
    (5, 16),  # BL < HW-bot
    (16, 29),  # HW-bot < BR
    # Penalty box tops/bots
    (1, 25),  # L-pen-top < R-pen-top
    (4, 28),  # L-pen-bot < R-pen-bot
    (1, 13),  # L-pen-top < HW-top
    (13, 25),  # HW-top < R-pen-top
    (4, 16),  # L-pen-bot < HW-bot
    (16, 28),  # HW-bot < R-pen-bot
    # Goal tops/bots
    (2, 26),  # L-goal-top < R-goal-top
    (3, 27),  # L-goal-bot < R-goal-bot
    (2, 13),  # L-goal-top < HW-top
    (13, 26),  # HW-top < R-goal-top
    # Goal front
    (6, 22),  # L-goal-front-top < R-goal-front-top
    (7, 23),  # L-goal-front-bot < R-goal-front-bot
    # Penalty front/inner
    (9, 17),  # L-pen-front-top < R-pen-front-top
    (12, 20),  # L-pen-front-bot < R-pen-front-bot
    (10, 18),  # L-pen-inner-top < R-pen-inner-top
    (11, 19),  # L-pen-inner-bot < R-pen-inner-bot
    # Penalty spots & centre circle
    (8, 21),  # L-pen-spot < R-pen-spot
    (30, 31),  # CC-left < CC-right
    (8, 13),  # L-pen-spot < HW (spot is in left half)
    (13, 21),  # HW < R-pen-spot
]

# Each entry (a, b) asserts that keypoint[a].y < keypoint[b].y (a is above b in frame).
_VERT_PAIRS: List[Tuple[int, int]] = [
    # Corners
    (0, 5),  # TL above BL
    (24, 29),  # TR above BR
    # Halfway line
    (13, 16),  # HW-top above HW-bot
    # Centre circle
    (14, 15),  # CC-top above CC-bot
    # Penalty box tops/bots
    (1, 4),  # L-pen-top above L-pen-bot
    (25, 28),  # R-pen-top above R-pen-bot
    # Goal tops/bots
    (2, 3),  # L-goal-top above L-goal-bot
    (26, 27),  # R-goal-top above R-goal-bot
    # Goal front top/bot
    (6, 7),  # L-goal-front-top above L-goal-front-bot
    (22, 23),  # R-goal-front-top above R-goal-front-bot
    # Penalty front top/bot
    (9, 12),  # L-pen-front-top above L-pen-front-bot
    (17, 20),  # R-pen-front-top above R-pen-front-bot
    # Penalty inner top/bot
    (10, 11),  # L-pen-inner-top above L-pen-inner-bot
    (18, 19),  # R-pen-inner-top above R-pen-inner-bot
]


# Priority-ordered keypoint indices for homography estimation (30-keypoint pitch schema).
# Ordered by geometric spread value: corners first (maximum spread), then halfway,
# then penalty spots, then centre circle, then penalty box corners.
# Used to fill remaining slots after zone-guaranteed selection.
_KP_PRIORITY: List[int] = [
    0,
    5,
    24,
    29,  # corners (TL, BL, TR, BR) — max spread, most stable
    13,
    16,  # halfway top/bot — constrains L/R orientation
    8,
    21,  # penalty spots (L, R) — single-pixel landmarks
    14,
    15,
    30,
    31,  # centre circle (top, bot, left, right)
    1,
    4,
    25,
    28,  # penalty box corners (L-pen-top/bot, R-pen-top/bot)
    2,
    3,
    26,
    27,  # goal box corners (L-goal-top/bot, R-goal-top/bot)
    6,
    7,
    22,
    23,  # goal front (L/R top/bot)
    9,
    12,
    17,
    20,  # penalty front (L/R top/bot)
    10,
    11,
    18,
    19,  # penalty inner (L/R top/bot)
]

# Coverage zones — each list contains keypoint indices that represent a spatial region.
# Zone-guaranteed selection ensures at least one point from each zone before filling
# remaining slots via _KP_PRIORITY, preventing all points clustering on one side.
_ZONE_LEFT: List[int] = [0, 5, 1, 4, 2, 3, 8, 6, 7, 9, 10, 11, 12]
_ZONE_RIGHT: List[int] = [24, 29, 25, 28, 26, 27, 21, 22, 23, 17, 18, 19, 20]
_ZONE_CENTER: List[int] = [14, 15, 30, 31, 13, 16]  # centre circle + halfway tops/bots
_ZONE_TOP: List[int] = [0, 24, 13, 1, 25, 2, 26, 6, 22, 14, 30, 31]
_ZONE_BOTTOM: List[int] = [5, 29, 16, 4, 28, 3, 27, 7, 23, 15]
_COVERAGE_ZONES: List[List[int]] = [_ZONE_LEFT, _ZONE_RIGHT, _ZONE_CENTER, _ZONE_TOP, _ZONE_BOTTOM]


def _filter_geometric_consistency(
    positions: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """
    Penalise keypoints that violate expected geometric ordering.

    For each constraint pair (a, b) where both keypoints are active,
    checks that the positional ordering holds.  When violated, the
    lower-confidence keypoint in the pair is multiplied by 0.3 (soft
    suppression rather than hard zero, to avoid over-killing valid
    detections at foreshortened broadcast angles).

    Args:
        positions:  [N, 2] EMA-smoothed pixel positions.
        confidence: [N]    effective confidence scores.

    Returns:
        Filtered confidence array — violated keypoints are penalised by 0.3×
        rather than zeroed, to avoid over-suppression at broadcast angles where
        perspective foreshortening can legitimately violate strict ordering.
    """
    _PENALTY = 0.3
    conf = confidence.copy()
    n = len(conf)

    for a, b in _HORIZ_PAIRS:
        if a >= n or b >= n:
            continue
        if conf[a] <= 0.0 or conf[b] <= 0.0:
            continue
        if positions[a, 0] >= positions[b, 0]:  # violation: a should be left of b
            if conf[a] < conf[b]:
                conf[a] *= _PENALTY
            else:
                conf[b] *= _PENALTY

    for a, b in _VERT_PAIRS:
        if a >= n or b >= n:
            continue
        if conf[a] <= 0.0 or conf[b] <= 0.0:
            continue
        if positions[a, 1] >= positions[b, 1]:  # violation: a should be above b
            if conf[a] < conf[b]:
                conf[a] *= _PENALTY
            else:
                conf[b] *= _PENALTY

    return conf


class KeypointTracker:
    """
    Per-keypoint temporal tracker for pitch landmark stabilization.

    Maintains an EMA of each keypoint's pixel position, a track-age counter,
    and an EMA position variance.  Effective confidence is:

        eff_conf = raw_conf × age_factor × stability

    where ``age_factor`` rewards long-lived tracks and ``stability`` penalises
    keypoints whose position residual (distance from EMA) is large across
    frames.  This means geometrically stable landmarks (corner flags, penalty
    spots) automatically dominate RANSAC over noisy or intermittently visible
    ones.

    Large position jumps (> jump_threshold px) trigger a track reset so that
    hard camera cuts do not blend old and new positions.

    Args:
        num_keypoints:   Number of tracked keypoints (default 32).
        ema_alpha:       EMA weight for the new observation (0 = frozen, 1 = raw, no smoothing).
                         Default 1.0 — raw positions passed to RANSAC so homography tracks
                         camera movement without lag.
        var_alpha:       EMA weight for variance update (slower than position).
        max_gap_frames:  Frames without a detection before a track is considered lost.
        age_saturation:  Track age at which the age boost saturates (frames).
        jump_threshold:  Pixel distance that triggers a track reset (hard cut).
        conf_threshold:  Minimum raw confidence to accept a detection.
        stability_scale: Residual std (px) at which stability weight = 0.5.
                         Lower → tighter penalty for noisy keypoints.
    """

    def __init__(
        self,
        num_keypoints: int = 30,
        ema_alpha: float = 1.0,
        var_alpha: float = 0.1,
        max_gap_frames: int = 10,
        age_saturation: int = 30,
        jump_threshold: float = 80.0,
        conf_threshold: float = 0.1,
        stability_scale: float = 15.0,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.ema_alpha = ema_alpha
        self.var_alpha = var_alpha
        self.max_gap = max_gap_frames
        self.age_saturation = age_saturation
        self.jump_threshold = jump_threshold
        self.conf_threshold = conf_threshold
        self.stability_scale = stability_scale

        self._positions = np.zeros((num_keypoints, 2), dtype=np.float32)
        self._age = np.zeros(num_keypoints, dtype=np.int32)
        self._gap = np.full(num_keypoints, max_gap_frames + 1, dtype=np.int32)
        self._ema_var = np.zeros(num_keypoints, dtype=np.float32)  # EMA of squared residual (px²)

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update tracks with new detector output and return stabilised keypoints.

        Args:
            keypoints:  [N, 2] pixel coordinates from the detector.
            confidence: [N]    raw confidence scores in [0, 1].

        Returns:
            smoothed_kps:   [N, 2]  EMA-smoothed positions (only valid where effective_conf > 0).
            effective_conf: [N]     conf × age_factor × stability; 0 for undetected keypoints.
        """
        effective_conf = np.zeros(self.num_keypoints, dtype=np.float32)

        for k in range(self.num_keypoints):
            if confidence[k] <= self.conf_threshold:
                self._gap[k] += 1
                if self._gap[k] > self.max_gap:
                    self._age[k] = 0
                    self._ema_var[k] = 0.0
                continue

            cold_start = self._gap[k] > self.max_gap
            dist = float(np.linalg.norm(keypoints[k] - self._positions[k]))
            jump = not cold_start and dist > self.jump_threshold

            if cold_start or jump:
                self._positions[k] = keypoints[k].copy()
                self._age[k] = 1
                self._ema_var[k] = 0.0
            else:
                # Measure residual from current EMA before updating it
                self._ema_var[k] = (1.0 - self.var_alpha) * self._ema_var[k] + self.var_alpha * dist**2
                self._positions[k] = self.ema_alpha * keypoints[k] + (1.0 - self.ema_alpha) * self._positions[k]
                self._age[k] += 1

            self._gap[k] = 0

            age_factor = min(1.0, self._age[k] / self.age_saturation)
            std = float(np.sqrt(self._ema_var[k]))
            stability = 1.0 / (1.0 + std / self.stability_scale)
            effective_conf[k] = confidence[k] * (0.5 + 0.5 * age_factor) * stability

        # Geometric consistency: zero out keypoints that violate expected
        # relative positions (e.g. left-side kp should have smaller x than right-side).
        # Only applied when both keypoints in a pair are active.
        effective_conf = _filter_geometric_consistency(self._positions, effective_conf)

        return self._positions.copy(), effective_conf

    def reset(self) -> None:
        """Reset all tracks (e.g. after a known scene cut)."""
        self._age[:] = 0
        self._gap[:] = self.max_gap + 1
        self._ema_var[:] = 0.0


class HomographyEstimator:
    """
    Estimate homography from detected line keypoints.

    Uses RANSAC for robust estimation and temporal smoothing
    to reduce jitter in the projection.

    Args:
        min_correspondences: Minimum point pairs for estimation.
        min_inliers: Minimum inliers to accept homography.
        ransac_reproj_threshold: RANSAC reprojection threshold.
        confidence_threshold: Minimum line confidence to use.
        visibility_threshold: Minimum keypoint visibility.
        smoothing_alpha: Temporal smoothing factor.

    Example:
        >>> estimator = HomographyEstimator()
        >>> estimator.estimate(keypoints, visibility, confidence, (1080, 1920))
        >>> position = estimator.project_player_to_pitch([100, 200, 150, 350])
    """

    def __init__(
        self,
        min_correspondences: int = 4,
        min_inliers: int = 6,
        ransac_reproj_threshold: float = 3.0,
        confidence_threshold: float = 0.5,
        visibility_threshold: float = 0.5,
        max_correspondences: int = 12,
        smoothing_alpha: float = 0.15,
        use_kalman: bool = True,
        reproject_interval: int = 5,
    ) -> None:
        self.min_correspondences = min_correspondences
        self.min_inliers = min_inliers
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.confidence_threshold = confidence_threshold
        self.visibility_threshold = visibility_threshold
        self.max_correspondences = max_correspondences
        self.smoothing_alpha = smoothing_alpha
        self.reproject_interval = reproject_interval

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.H_smoothed: Optional[np.ndarray] = None
        self.inliers: Optional[np.ndarray] = None
        self.num_inliers: int = 0
        self.selected_indices: frozenset = frozenset()  # keypoint indices used in last successful estimate
        self.mean_reprojection_error: float = float("inf")  # RANSAC inlier reprojection error (pitch metres)

        # Temporal fallback — extended to 30 frames so Kalman prediction covers ~1 s at 30fps
        self.frames_since_valid: int = 0
        self.max_fallback_frames: int = 60
        self.last_valid_H: Optional[np.ndarray] = None

        # Optical flow tracking for robust left/right disambiguation
        self.prev_gray: Optional[np.ndarray] = None
        self.H_flow_pred: Optional[np.ndarray] = None

        # Camera-pose Kalman smoother (replaces EMA when use_kalman=True)
        self._kalman: Optional[CameraPoseKalmanFilter] = CameraPoseKalmanFilter() if use_kalman else None
        self._last_image_size: Optional[Tuple[int, int]] = None

        # Inlier correspondences from the most recent successful estimate() call.
        # Used by get_camera_model() to recover a full K[R|t] via solvePnP.
        self._matched_src: Optional[np.ndarray] = None  # [M, 2] image pixel coords
        self._matched_dst: Optional[np.ndarray] = None  # [M, 2] pitch 2D coords (metres)

        # Load line classes
        try:
            from torchkick.soccernet.calibration_data import LINE_CLASSES

            self.line_classes = LINE_CLASSES
        except ImportError:
            try:
                from soccernet.calibration_data import LINE_CLASSES

                self.line_classes = LINE_CLASSES
            except ImportError:
                self.line_classes = list(PITCH_LINE_COORDINATES.keys())

    def _project_with_H(
        self,
        points: np.ndarray,
        H: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Project points using specific homography."""
        if H is None:
            return None

        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, 2)

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.hstack([points, ones])

        projected = (H @ points_h.T).T

        w = projected[:, 2:3]
        result = projected[:, :2] / w
        if not np.all(np.isfinite(result)):
            return None
        return result

    def estimate(
        self,
        keypoints,
        visibility,
        confidence,
        image_size: Tuple[int, int],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Estimate homography from detected keypoints.

        Supports two input formats:
          - HRNet format: keypoints [num_classes, max_pts, 2] (3D tensor, normalized)
          - ViTPose format: keypoints [N, 2] pixel coords, confidence [N] (2D array)

        Args:
            keypoints: Keypoint coordinates (see above).
            visibility: Visibility scores matching keypoints shape.
            confidence: Per-class or per-keypoint confidence scores.
            image_size: (height, width) of source image.
            frame: Optional [H, W, 3] RGB frame used to calculate optical flow for stabilizing camera left/right panning.

        Returns:
            True if homography was successfully computed.
        """
        h, w = image_size
        self._last_image_size = image_size

        # --- OPTICAL FLOW PASS ---
        # Update flow prediction _before_ falling back into keypoint parsing.
        if frame is not None and self.last_valid_H is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
            if self.prev_gray is not None:
                p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200, qualityLevel=0.02, minDistance=30)
                if p0 is not None and len(p0) > 10:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None)
                    if p1 is not None and len(p1[st == 1]) > 10:
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]
                        H_cam, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3.0)
                        if H_cam is not None:
                            # H_cam maps pixels from N-1 to N.
                            # P_old = H_cam^-1 @ P_new
                            # P_pitch = H_old @ P_old = H_old @ H_cam^-1 @ P_new
                            try:
                                inv_H_cam = np.linalg.inv(H_cam)
                                self.H_flow_pred = self.last_valid_H @ inv_H_cam
                                self.H_flow_pred /= self.H_flow_pred[2, 2] + 1e-8
                            except np.linalg.LinAlgError:
                                pass
            self.prev_gray = gray

        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(visibility, torch.Tensor):
            visibility = visibility.cpu().numpy()
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()

        keypoints = np.asarray(keypoints)
        visibility = np.asarray(visibility)
        confidence = np.asarray(confidence)

        # Build correspondences
        src_points = []
        dst_points = []

        if keypoints.ndim == 2:
            # Flat format: [N, 2] pixel coords, confidence [N]
            # If N <= 30, use vertex lookup directly (index → pitch coordinate).
            # Fall back to SoccerNet LINE_CLASSES mapping only when N > 30.
            use_vertex_lookup = len(keypoints) <= len(VERTICES)

            if use_vertex_lookup:
                # Use all keypoints above confidence threshold — RANSAC handles
                # outlier rejection, so more points = better constraints.
                # No cap: letting RANSAC work on the full above-threshold set
                # is strictly better than pre-selecting a subset.
                n = len(keypoints)
                selected_indices: set = set()
                for i in range(min(n, len(VERTICES))):
                    if confidence[i] < self.confidence_threshold:
                        continue
                    img_x, img_y = keypoints[i]
                    if img_x < 0 or img_x > w or img_y < 0 or img_y > h:
                        continue
                    selected_indices.add(i)
                    px, py = VERTICES[i]
                    src_points.append([float(img_x), float(img_y)])
                    dst_points.append([px, py])
                self.selected_indices = frozenset(selected_indices)
            else:
                for i, (conf, (img_x, img_y)) in enumerate(zip(confidence, keypoints)):
                    if conf < self.confidence_threshold:
                        continue
                    if img_x < 0 or img_x > w or img_y < 0 or img_y > h:
                        continue
                    # SoccerNet LINE_CLASSES mapping
                    if i < len(self.line_classes):
                        class_name = self.line_classes[i]
                        if class_name in PITCH_LINE_COORDINATES:
                            pitch_pts = PITCH_LINE_COORDINATES[class_name]
                            if pitch_pts:
                                _, pitch_point = pitch_pts[0]
                                src_points.append([img_x, img_y])
                                dst_points.append(pitch_point.to_array())
        else:
            # HRNet format: [num_classes, max_pts, 2] normalized
            for class_idx, class_name in enumerate(self.line_classes):
                if class_idx >= len(confidence):
                    continue
                if confidence[class_idx] < self.confidence_threshold:
                    continue
                if class_name not in PITCH_LINE_COORDINATES:
                    continue

                pitch_coords = PITCH_LINE_COORDINATES[class_name]

                for kp_idx, pitch_point in pitch_coords:
                    if kp_idx >= keypoints.shape[1]:
                        continue
                    if class_idx >= visibility.shape[0]:
                        continue
                    if visibility[class_idx, kp_idx] < self.visibility_threshold:
                        continue

                    img_x = keypoints[class_idx, kp_idx, 0] * w
                    img_y = keypoints[class_idx, kp_idx, 1] * h

                    if img_x < 0 or img_x > w or img_y < 0 or img_y > h:
                        continue

                    src_points.append([img_x, img_y])
                    dst_points.append(pitch_point.to_array())

        if len(src_points) < self.min_correspondences:
            self.H = None
            self.H_inv = None
            return False

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Try both orientations
        H1, mask1 = cv2.findHomography(src_points, dst_points, cv2.RANSAC, self.ransac_reproj_threshold)
        inliers1 = np.sum(mask1.ravel() == 1) if H1 is not None else 0

        dst_mirrored = dst_points.copy()
        dst_mirrored[:, 0] *= -1
        H2, mask2 = cv2.findHomography(src_points, dst_mirrored, cv2.RANSAC, self.ransac_reproj_threshold)
        inliers2 = np.sum(mask2.ravel() == 1) if H2 is not None else 0

        if inliers2 > inliers1 and inliers2 >= self.min_inliers:
            self.H = H2
            self.inliers = mask2.ravel() == 1
            self.num_inliers = inliers2
            self._matched_src = src_points[self.inliers]
            self._matched_dst = dst_mirrored[self.inliers]
        else:
            self.H = H1
            self.inliers = mask1.ravel() == 1 if H1 is not None else np.zeros(len(src_points), dtype=bool)
            self.num_inliers = inliers1
            self._matched_src = src_points[self.inliers]
            self._matched_dst = dst_points[self.inliers]

        # Left/Right ambiguity correction via Optical Flow tracking
        if self.H is not None and self.H_flow_pred is not None:
            # Check where the image center projects on the pitch
            center_h = np.array([[w / 2.0, h / 2.0, 1.0]], dtype=float).T

            p_vit = self.H @ center_h
            p_vit = p_vit / (p_vit[2] + 1e-8)

            p_flow = self.H_flow_pred @ center_h
            p_flow = p_flow / (p_flow[2] + 1e-8)

            # If the x-coordinates have opposite signs and are far apart, it's likely a ViT left/right mistake
            if p_vit[0, 0] * p_flow[0, 0] < 0 and abs(p_vit[0, 0] - p_flow[0, 0]) > 20.0:
                # Reject the ViT's homography, force fallback to flow prediction
                self.H = None
                self.num_inliers = 0

        # Compute mean reprojection error on inliers (pitch metres)
        if self.H is not None and self._matched_src is not None and len(self._matched_src) > 0:
            proj = cv2.perspectiveTransform(self._matched_src.reshape(-1, 1, 2), self.H).reshape(-1, 2)
            self.mean_reprojection_error = float(np.mean(np.linalg.norm(proj - self._matched_dst, axis=1)))
        else:
            self.mean_reprojection_error = float("inf")

        if self.H is None or self.num_inliers < self.min_inliers:
            self.frames_since_valid += 1

            # OPTICAL FLOW OVERRIDE priority - if flow prediction is valid, use it for seamless tracking
            if self.H_flow_pred is not None and self.last_valid_H is not None:
                self.H_smoothed = self.H_flow_pred
                self.last_valid_H = self.H_smoothed.copy()
                try:
                    self.H_inv = np.linalg.inv(self.H_smoothed)
                except np.linalg.LinAlgError:
                    self.H_inv = None
                return True

            if self.last_valid_H is not None and self.frames_since_valid <= self.max_fallback_frames:
                # Use Kalman prediction if available
                if self._kalman is not None:
                    predicted = self._kalman.predict()
                    if predicted is not None:
                        self.H_smoothed = predicted
                        return True
                self.H_smoothed = self.last_valid_H
                return True
            return False

        self.frames_since_valid = 0

        # Smooth with Kalman or EMA
        if self._kalman is not None:
            self.H_smoothed = self._kalman.update(self.H, image_size)
        else:
            if self.H_smoothed is None:
                self.H_smoothed = self.H.copy()
            else:
                self.H_smoothed = (1 - self.smoothing_alpha) * self.H_smoothed + self.smoothing_alpha * self.H

        self.last_valid_H = self.H_smoothed.copy()

        try:
            self.H_inv = np.linalg.inv(self.H_smoothed)
        except np.linalg.LinAlgError:
            self.H_inv = None

        return True

    def project_to_pitch(
        self,
        points: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Project image points to pitch coordinates.

        Args:
            points: [N, 2] pixel coordinates.

        Returns:
            [N, 2] pitch coordinates in meters.
        """
        H = self.H_smoothed if self.H_smoothed is not None else self.H
        if H is None:
            return None

        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, 2)

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.hstack([points, ones])

        projected = (H @ points_h.T).T
        return projected[:, :2] / projected[:, 2:3]

    def project_to_image(
        self,
        pitch_points: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Project pitch coordinates to image pixels.

        Args:
            pitch_points: [N, 2] pitch coordinates.

        Returns:
            [N, 2] pixel coordinates.
        """
        if self.H_inv is None:
            return None

        pitch_points = np.asarray(pitch_points, dtype=np.float32)
        if pitch_points.ndim == 1:
            pitch_points = pitch_points.reshape(1, 2)

        ones = np.ones((pitch_points.shape[0], 1), dtype=np.float32)
        points_h = np.hstack([pitch_points, ones])

        projected = (self.H_inv @ points_h.T).T
        return projected[:, :2] / projected[:, 2:3]

    def get_player_foot_position(
        self,
        bbox: List[float],
    ) -> Tuple[float, float]:
        """
        Get foot position from bounding box.

        Args:
            bbox: [x1, y1, x2, y2] player box.

        Returns:
            (x, y) foot position in pixels.
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, y2)

    def project_player_to_pitch(
        self,
        bbox: List[float],
        feet_uv: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Project player position to pitch coordinates.

        When feet_uv is provided (from ViTPose ankle keypoints), uses that
        instead of the bbox-derived foot position for higher accuracy.

        Args:
            bbox: [x1, y1, x2, y2] player bounding box.
            feet_uv: Optional (x, y) pixel of ankle keypoint from ViTPose.

        Returns:
            (x, y) in meters, or None if projection fails.
        """
        if feet_uv is not None:
            foot_x, foot_y = feet_uv
        else:
            foot_x, foot_y = self.get_player_foot_position(bbox)
        projected = self.project_to_pitch(np.array([[foot_x, foot_y]]))

        if projected is None:
            return None

        x, y = float(projected[0, 0]), float(projected[0, 1])

        # Validate and clamp
        MAX_X = HALF_LENGTH + 5.0
        MAX_Y = HALF_WIDTH + 5.0

        if abs(x) > MAX_X * 2 or abs(y) > MAX_Y * 2:
            return None

        return (
            float(np.clip(x, -MAX_X, MAX_X)),
            float(np.clip(y, -MAX_Y, MAX_Y)),
        )


class GeometricConstraintSolver:
    """
    Augment RANSAC correspondence points using geometric primitives.

    Takes raw line segments detected on the pitch (e.g. from HRNet or ViTPose)
    and computes additional virtual correspondence points by:
      1. Line-line intersections (e.g. penalty box corner = side line ∩ end line)
      2. Circle arc sampling (center circle tangent points)

    Each additional point feeds directly into HomographyEstimator's
    src_points / dst_points arrays, increasing RANSAC robustness on
    partial pitch views.

    Reference: https://arxiv.org/html/2410.07401v1

    Example:
        >>> solver = GeometricConstraintSolver()
        >>> extra_src, extra_dst = solver.solve(line_segments, frame_size)
        >>> # Append to existing correspondences before cv2.findHomography
    """

    # Known pitch line intersection → pitch coordinates (meters, pitch-centred)
    _CORNER_INTERSECTIONS = [
        # (line_class_A, line_class_B, pitch_x_m, pitch_y_m)
        ("Side line top", "Goal line left", -52.5, 34.0),
        ("Side line top", "Goal line right", 52.5, 34.0),
        ("Side line bottom", "Goal line left", -52.5, -34.0),
        ("Side line bottom", "Goal line right", 52.5, -34.0),
        ("Side line top", "Middle line", 0.0, 34.0),
        ("Side line bottom", "Middle line", 0.0, -34.0),
        ("Side line top", "Large rect. left top", -36.0, 34.0),
        ("Side line top", "Large rect. right top", 36.0, 34.0),
        ("Side line bottom", "Large rect. left bottom", -36.0, -34.0),
        ("Side line bottom", "Large rect. right bottom", 36.0, -34.0),
    ]

    def __init__(self, confidence_threshold: float = 0.4) -> None:
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _line_intersection(
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute intersection of line p1-p2 and line p3-p4."""
        d1 = p2 - p1
        d2 = p4 - p3
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-8:
            return None  # parallel
        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
        return p1 + t * d1

    def solve(
        self,
        line_endpoints: Dict[str, List[np.ndarray]],
        frame_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute additional correspondence points from line intersections.

        Args:
            line_endpoints: Dict mapping line class name → list of (pt_a, pt_b)
                endpoint pairs in image pixel coordinates.
            frame_size: (height, width) of the frame.

        Returns:
            extra_src: [M, 2] image-space intersection points.
            extra_dst: [M, 2] corresponding pitch-space points (meters).
        """
        h, w = frame_size
        src_pts: List[np.ndarray] = []
        dst_pts: List[np.ndarray] = []

        for line_a, line_b, pitch_x, pitch_y in self._CORNER_INTERSECTIONS:
            segs_a = line_endpoints.get(line_a, [])
            segs_b = line_endpoints.get(line_b, [])
            if not segs_a or not segs_b:
                continue

            # Use the first segment of each line class
            p1, p2 = np.asarray(segs_a[0][0], dtype=np.float32), np.asarray(segs_a[0][1], dtype=np.float32)
            p3, p4 = np.asarray(segs_b[0][0], dtype=np.float32), np.asarray(segs_b[0][1], dtype=np.float32)

            pt = self._line_intersection(p1, p2, p3, p4)
            if pt is None:
                continue
            # Reject points outside the frame
            if pt[0] < 0 or pt[0] >= w or pt[1] < 0 or pt[1] >= h:
                continue

            src_pts.append(pt)
            dst_pts.append(np.array([pitch_x, pitch_y], dtype=np.float32))

        if src_pts:
            return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32)
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)


__all__ = [
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "HALF_LENGTH",
    "HALF_WIDTH",
    "PitchPoint",
    "PITCH_LINE_COORDINATES",
    "CameraPoseKalmanFilter",
    "HomographyKalmanFilter",  # backward-compat alias
    "HomographyEstimator",
    "GeometricConstraintSolver",
]
