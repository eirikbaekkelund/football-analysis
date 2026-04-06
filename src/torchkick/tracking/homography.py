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


# Roboflow 32-keypoint pitch layout (SoccerPitchConfiguration.vertices, 0-indexed).
# Converted to center-origin meters to match the existing HomographyEstimator coordinate system
# (x: -52.5 … +52.5 along length, y: -34 … +34 along width).
# Source: https://github.com/roboflow/sports/blob/main/sports/configs/soccer.py
# Roboflow raw dimensions: 12000cm × 7000cm, origin at top-left corner.
# Conversion: x_center = x_raw/100 - 60, y_center = y_raw/100 - 35
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
    """Convert Roboflow corner-origin coords to center-origin meters.

    Normalises from the Roboflow 120×70m template to the codebase-standard
    105×68m coordinate system (HALF_LENGTH=52.5, HALF_WIDTH=34) so that
    pitch_viz renders correctly.
    """
    x_scaled = (x_raw / _RF_L) * PITCH_LENGTH - HALF_LENGTH
    y_scaled = (y_raw / _RF_W) * PITCH_WIDTH - HALF_WIDTH
    return (x_scaled, y_scaled)


ROBOFLOW_VERTICES: List[Tuple[float, float]] = [
    _rf(0, 0),  # 0  top-left corner
    _rf(0, (_RF_W - _RF_PBW) / 2),  # 1  left penalty box top
    _rf(0, (_RF_W - _RF_GBW) / 2),  # 2  left goal box top
    _rf(0, (_RF_W + _RF_GBW) / 2),  # 3  left goal box bottom
    _rf(0, (_RF_W + _RF_PBW) / 2),  # 4  left penalty box bottom
    _rf(0, _RF_W),  # 5  bottom-left corner
    _rf(_RF_GBD, (_RF_W - _RF_GBW) / 2),  # 6  left goal box front top
    _rf(_RF_GBD, (_RF_W + _RF_GBW) / 2),  # 7  left goal box front bottom
    _rf(_RF_PS, _RF_HY),  # 8  left penalty spot
    _rf(_RF_PBD, (_RF_W - _RF_PBW) / 2),  # 9  left penalty box front top
    _rf(_RF_PBD, (_RF_W - _RF_GBW) / 2),  # 10 left penalty box inner top
    _rf(_RF_PBD, (_RF_W + _RF_GBW) / 2),  # 11 left penalty box inner bottom
    _rf(_RF_PBD, (_RF_W + _RF_PBW) / 2),  # 12 left penalty box front bottom
    _rf(_RF_HX, 0),  # 13 halfway line top
    _rf(_RF_HX, _RF_HY - _RF_CCR),  # 14 centre circle top
    _rf(_RF_HX, _RF_HY + _RF_CCR),  # 15 centre circle bottom
    _rf(_RF_HX, _RF_W),  # 16 halfway line bottom
    _rf(_RF_L - _RF_PBD, (_RF_W - _RF_PBW) / 2),  # 17 right penalty box front top
    _rf(_RF_L - _RF_PBD, (_RF_W - _RF_GBW) / 2),  # 18 right penalty box inner top
    _rf(_RF_L - _RF_PBD, (_RF_W + _RF_GBW) / 2),  # 19 right penalty box inner bottom
    _rf(_RF_L - _RF_PBD, (_RF_W + _RF_PBW) / 2),  # 20 right penalty box front bottom
    _rf(_RF_L - _RF_PS, _RF_HY),  # 21 right penalty spot
    _rf(_RF_L - _RF_GBD, (_RF_W - _RF_GBW) / 2),  # 22 right goal box front top
    _rf(_RF_L - _RF_GBD, (_RF_W + _RF_GBW) / 2),  # 23 right goal box front bottom
    _rf(_RF_L, 0),  # 24 top-right corner
    _rf(_RF_L, (_RF_W - _RF_PBW) / 2),  # 25 right penalty box top
    _rf(_RF_L, (_RF_W - _RF_GBW) / 2),  # 26 right goal box top
    _rf(_RF_L, (_RF_W + _RF_GBW) / 2),  # 27 right goal box bottom
    _rf(_RF_L, (_RF_W + _RF_PBW) / 2),  # 28 right penalty box bottom
    _rf(_RF_L, _RF_W),  # 29 bottom-right corner
    _rf(_RF_HX - _RF_CCR, _RF_HY),  # 30 centre circle left
    _rf(_RF_HX + _RF_CCR, _RF_HY),  # 31 centre circle right
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
        smoothing_alpha: float = 0.15,
        use_kalman: bool = True,
        reproject_interval: int = 5,
    ) -> None:
        self.min_correspondences = min_correspondences
        self.min_inliers = min_inliers
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.confidence_threshold = confidence_threshold
        self.visibility_threshold = visibility_threshold
        self.smoothing_alpha = smoothing_alpha
        self.reproject_interval = reproject_interval

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.H_smoothed: Optional[np.ndarray] = None
        self.inliers: Optional[np.ndarray] = None
        self.num_inliers: int = 0

        # Temporal fallback — extended to 30 frames so Kalman prediction covers ~1 s at 30fps
        self.frames_since_valid: int = 0
        self.max_fallback_frames: int = 30
        self.last_valid_H: Optional[np.ndarray] = None

        # Camera-pose Kalman smoother (replaces EMA when use_kalman=True)
        self._kalman: Optional[CameraPoseKalmanFilter] = CameraPoseKalmanFilter() if use_kalman else None
        self._last_image_size: Optional[Tuple[int, int]] = None

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

        Returns:
            True if homography was successfully computed.
        """
        h, w = image_size
        self._last_image_size = image_size

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
            # If N <= 32, use Roboflow vertex lookup directly (index → pitch coordinate).
            # Roboflow dataset has 32 keypoints; older versions may have fewer (e.g. 29).
            # Fall back to SoccerNet LINE_CLASSES mapping only when N > 32.
            use_roboflow = len(keypoints) <= len(ROBOFLOW_VERTICES)

            for i, (conf, (img_x, img_y)) in enumerate(zip(confidence, keypoints)):
                if conf < self.confidence_threshold:
                    continue
                if img_x < 0 or img_x > w or img_y < 0 or img_y > h:
                    continue

                if use_roboflow:
                    px, py = ROBOFLOW_VERTICES[i]
                    src_points.append([img_x, img_y])
                    dst_points.append([px, py])
                    continue

                # SoccerNet LINE_CLASSES mapping
                if i < len(self.line_classes):
                    class_name = self.line_classes[i]
                    if class_name in PITCH_LINE_COORDINATES:
                        pitch_pts = PITCH_LINE_COORDINATES[class_name]
                        if pitch_pts:
                            # Use first point of the line class as representative
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
        else:
            self.H = H1
            self.inliers = mask1.ravel() == 1 if H1 is not None else np.zeros(len(src_points), dtype=bool)
            self.num_inliers = inliers1

        if self.H is None or self.num_inliers < self.min_inliers:
            self.frames_since_valid += 1
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
