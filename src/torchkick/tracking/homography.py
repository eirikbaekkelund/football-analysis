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
    ) -> None:
        self.min_correspondences = min_correspondences
        self.min_inliers = min_inliers
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.confidence_threshold = confidence_threshold
        self.visibility_threshold = visibility_threshold
        self.smoothing_alpha = smoothing_alpha

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.H_smoothed: Optional[np.ndarray] = None
        self.inliers: Optional[np.ndarray] = None
        self.num_inliers: int = 0

        # Temporal fallback
        self.frames_since_valid: int = 0
        self.max_fallback_frames: int = 15
        self.last_valid_H: Optional[np.ndarray] = None

        # Load line classes
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
        if np.any(np.abs(w) < 1e-6):
            return None

        return projected[:, :2] / w

    def estimate(
        self,
        keypoints: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> bool:
        """
        Estimate homography from detected keypoints.

        Args:
            keypoints: [num_classes, max_points, 2] normalized coords.
            visibility: [num_classes, max_points] visibility scores.
            confidence: [num_classes] line confidence scores.
            image_size: (height, width) of source image.

        Returns:
            True if homography was successfully computed.
        """
        h, w = image_size

        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(visibility, torch.Tensor):
            visibility = visibility.cpu().numpy()
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()

        # Build correspondences
        src_points = []
        dst_points = []

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
                self.H_smoothed = self.last_valid_H
                return True
            return False

        self.frames_since_valid = 0

        # Temporal smoothing
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
    ) -> Optional[Tuple[float, float]]:
        """
        Project player position to pitch coordinates.

        Args:
            bbox: [x1, y1, x2, y2] player box.

        Returns:
            (x, y) in meters, or None if projection fails.
        """
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


__all__ = [
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "HALF_LENGTH",
    "HALF_WIDTH",
    "PitchPoint",
    "PITCH_LINE_COORDINATES",
    "HomographyEstimator",
]
