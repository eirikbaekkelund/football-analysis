import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from utils.timing import timed, DEBUG_TIMING


# pitch dimensions (in meters)
PITCH_LENGTH = 105.0  # touchline length
PITCH_WIDTH = 68.0  # goal line length
HALF_LENGTH = PITCH_LENGTH / 2  # 52.5m
HALF_WIDTH = PITCH_WIDTH / 2  # 34m
PENALTY_AREA_WIDTH = 40.32
PENALTY_AREA_DEPTH = 16.5
GOAL_AREA_WIDTH = 18.32
GOAL_AREA_DEPTH = 5.5
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DISTANCE = 11.0
GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44


class PitchPoint:
    """Lightweight pitch point - no Pydantic overhead."""

    __slots__ = ('x', 'y')

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


PITCH_LINE_COORDINATES: Dict[str, List[Tuple[int, PitchPoint]]] = {
    "Side line top": [
        (0, PitchPoint(x=-HALF_LENGTH, y=HALF_WIDTH)),  # top-left corner
        (1, PitchPoint(x=HALF_LENGTH, y=HALF_WIDTH)),  # top-right corner
    ],
    "Side line bottom": [
        (0, PitchPoint(x=-HALF_LENGTH, y=-HALF_WIDTH)),  # bottom-left corner
        (1, PitchPoint(x=HALF_LENGTH, y=-HALF_WIDTH)),  # bottom-right corner
    ],
    "Side line left": [
        (0, PitchPoint(x=-HALF_LENGTH, y=-HALF_WIDTH)),  # bottom-left corner
        (1, PitchPoint(x=-HALF_LENGTH, y=HALF_WIDTH)),  # top-left corner
    ],
    "Side line right": [
        (0, PitchPoint(x=HALF_LENGTH, y=-HALF_WIDTH)),  # bottom-right corner
        (1, PitchPoint(x=HALF_LENGTH, y=HALF_WIDTH)),  # top-right corner
    ],
    "Middle line": [
        (0, PitchPoint(x=0, y=-HALF_WIDTH)),  # bottom of halfway line
        (1, PitchPoint(x=0, y=HALF_WIDTH)),  # top of halfway line
    ],
    "Big rect. left main": [
        (0, PitchPoint(x=-HALF_LENGTH + PENALTY_AREA_DEPTH, y=-PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(x=-HALF_LENGTH + PENALTY_AREA_DEPTH, y=PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. left top": [
        (0, PitchPoint(x=-HALF_LENGTH, y=PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(x=-HALF_LENGTH + PENALTY_AREA_DEPTH, y=PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. left bottom": [
        (0, PitchPoint(x=-HALF_LENGTH, y=-PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(x=-HALF_LENGTH + PENALTY_AREA_DEPTH, y=-PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. right main": [
        (0, PitchPoint(x=HALF_LENGTH - PENALTY_AREA_DEPTH, y=-PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(x=HALF_LENGTH - PENALTY_AREA_DEPTH, y=PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. right top": [
        (0, PitchPoint(x=HALF_LENGTH - PENALTY_AREA_DEPTH, y=PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(x=HALF_LENGTH, y=PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. right bottom": [
        (0, PitchPoint(x=HALF_LENGTH - PENALTY_AREA_DEPTH, y=-PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(x=HALF_LENGTH, y=-PENALTY_AREA_WIDTH / 2)),
    ],
    "Small rect. left main": [
        (0, PitchPoint(x=-HALF_LENGTH + GOAL_AREA_DEPTH, y=-GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(x=-HALF_LENGTH + GOAL_AREA_DEPTH, y=GOAL_AREA_WIDTH / 2)),
    ],
    "Small rect. left top": [
        (0, PitchPoint(x=-HALF_LENGTH, y=GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(x=-HALF_LENGTH + GOAL_AREA_DEPTH, y=GOAL_AREA_WIDTH / 2)),
    ],
    "Small rect. left bottom": [
        (0, PitchPoint(x=-HALF_LENGTH, y=-GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(x=-HALF_LENGTH + GOAL_AREA_DEPTH, y=-GOAL_AREA_WIDTH / 2)),
    ],
    "Small rect. right main": [
        (0, PitchPoint(x=HALF_LENGTH - GOAL_AREA_DEPTH, y=-GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(x=HALF_LENGTH - GOAL_AREA_DEPTH, y=GOAL_AREA_WIDTH / 2)),
    ],
    "Small rect. right top": [
        (0, PitchPoint(x=HALF_LENGTH - GOAL_AREA_DEPTH, y=GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(x=HALF_LENGTH, y=GOAL_AREA_WIDTH / 2)),
    ],
    "Small rect. right bottom": [
        (0, PitchPoint(x=HALF_LENGTH - GOAL_AREA_DEPTH, y=-GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(x=HALF_LENGTH, y=-GOAL_AREA_WIDTH / 2)),
    ],
    "Goal left crossbar": [
        (0, PitchPoint(x=-HALF_LENGTH, y=-GOAL_WIDTH / 2)),
        (1, PitchPoint(x=-HALF_LENGTH, y=GOAL_WIDTH / 2)),
    ],
    "Goal left post left ": [
        (0, PitchPoint(x=-HALF_LENGTH, y=-GOAL_WIDTH / 2)),
        (1, PitchPoint(x=-HALF_LENGTH, y=-GOAL_WIDTH / 2)),
    ],
    "Goal left post right": [
        (0, PitchPoint(x=-HALF_LENGTH, y=GOAL_WIDTH / 2)),
        (1, PitchPoint(x=-HALF_LENGTH, y=GOAL_WIDTH / 2)),
    ],
    "Goal right crossbar": [
        (0, PitchPoint(x=HALF_LENGTH, y=-GOAL_WIDTH / 2)),
        (1, PitchPoint(x=HALF_LENGTH, y=GOAL_WIDTH / 2)),
    ],
    "Goal right post left": [
        (0, PitchPoint(x=HALF_LENGTH, y=-GOAL_WIDTH / 2)),
        (1, PitchPoint(x=HALF_LENGTH, y=-GOAL_WIDTH / 2)),
    ],
    "Goal right post right": [
        (0, PitchPoint(x=HALF_LENGTH, y=GOAL_WIDTH / 2)),
        (1, PitchPoint(x=HALF_LENGTH, y=GOAL_WIDTH / 2)),
    ],
}


def _get_circle_points(
    center_x: float, center_y: float, radius: float, n_points: int = 9
) -> List[Tuple[int, PitchPoint]]:
    """Generate circle keypoints evenly spaced around the circle."""
    points = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append((i, PitchPoint(x=x, y=y)))
    return points


PITCH_LINE_COORDINATES["Circle central"] = _get_circle_points(0, 0, CENTER_CIRCLE_RADIUS)
PITCH_LINE_COORDINATES["Circle left"] = _get_circle_points(
    -HALF_LENGTH + PENALTY_SPOT_DISTANCE, 0, CENTER_CIRCLE_RADIUS
)
PITCH_LINE_COORDINATES["Circle right"] = _get_circle_points(
    HALF_LENGTH - PENALTY_SPOT_DISTANCE, 0, CENTER_CIRCLE_RADIUS
)


class HomographyEstimator:
    """
    Estimates homography from detected line keypoints to pitch coordinates.

    Uses RANSAC for robust estimation when multiple line correspondences are available.
    Includes temporal smoothing to reduce jitter.
    """

    def __init__(
        self,
        min_correspondences: int = 4,
        min_inliers: int = 6,
        ransac_reproj_threshold: float = 3.0,
        confidence_threshold: float = 0.5,
        visibility_threshold: float = 0.5,
        smoothing_alpha: float = 0.15,  # Lower = smoother (was 0.3)
    ):
        """
        Args:
            min_correspondences: Minimum point correspondences needed for homography
            min_inliers: Minimum inliers required to accept a new homography
            ransac_reproj_threshold: RANSAC reprojection error threshold (pixels)
            confidence_threshold: Minimum line class confidence to use
            visibility_threshold: Minimum keypoint visibility to use
            smoothing_alpha: Blend factor for temporal smoothing (0=keep old, 1=use new)
        """
        self.min_correspondences = min_correspondences
        self.min_inliers = min_inliers
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.confidence_threshold = confidence_threshold
        self.visibility_threshold = visibility_threshold
        self.smoothing_alpha = smoothing_alpha

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.H_smoothed: Optional[np.ndarray] = None  # Temporally smoothed
        self.inliers: Optional[np.ndarray] = None
        self.num_inliers: int = 0

        # self-correction: temporal fallback tracking
        self.frames_since_valid: int = 0
        self.max_fallback_frames: int = 15
        self.last_valid_H: Optional[np.ndarray] = None

        from soccernet.calibration_data import LINE_CLASSES

        self.line_classes = LINE_CLASSES

    def _project_with_H(self, points: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
        """Project image points to pitch using specific homography matrix."""
        if H is None:
            return None

        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, 2)

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.hstack([points, ones])

        projected = (H @ points_h.T).T

        # Avoid division by zero
        w = projected[:, 2:3]
        if np.any(np.abs(w) < 1e-6):
            return None

        return projected[:, :2] / w

    def estimate(
        self,
        keypoints: torch.Tensor,  # [num_classes, max_points, 2] normalized coords
        visibility: torch.Tensor,  # [num_classes, max_points]
        confidence: torch.Tensor,  # [num_classes]
        image_size: Tuple[int, int],  # (height, width) of the image
    ) -> bool:
        """
        Estimate homography from detected keypoints.

        Args:
            keypoints: Predicted keypoints in normalized [0,1] coords
            visibility: Visibility scores per keypoint
            confidence: Confidence scores per line class
            image_size: (height, width) of the source image

        Returns:
            True if homography was successfully computed
        """
        h, w = image_size

        # Convert tensors to numpy
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(visibility, torch.Tensor):
            visibility = visibility.cpu().numpy()
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()

        # Collect point correspondences
        src_points = []  # Image points (pixels)
        dst_points = []  # Pitch points (meters)

        for class_idx, class_name in enumerate(self.line_classes):
            # Skip low confidence classes
            if confidence[class_idx] < self.confidence_threshold:
                continue

            # Skip classes we don't have pitch coordinates for
            if class_name not in PITCH_LINE_COORDINATES:
                continue

            pitch_coords = PITCH_LINE_COORDINATES[class_name]

            for kp_idx, pitch_point in pitch_coords:
                # Check if this keypoint is visible
                if kp_idx >= keypoints.shape[1]:
                    continue
                if visibility[class_idx, kp_idx] < self.visibility_threshold:
                    continue

                # Get image coordinates (denormalize)
                img_x = keypoints[class_idx, kp_idx, 0] * w
                img_y = keypoints[class_idx, kp_idx, 1] * h

                # Skip invalid coordinates
                if img_x < 0 or img_x > w or img_y < 0 or img_y > h:
                    continue

                src_points.append([img_x, img_y])
                dst_points.append(pitch_point.to_array())

        if len(src_points) < self.min_correspondences:
            print(f"Not enough correspondences: {len(src_points)} < {self.min_correspondences}")
            self.H = None
            self.H_inv = None
            return False

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Strategy 1: Standard RANSAC
        H1, mask1 = cv2.findHomography(src_points, dst_points, cv2.RANSAC, self.ransac_reproj_threshold)
        inliers1 = np.sum(mask1.ravel() == 1) if H1 is not None else 0

        # Strategy 2: Mirrored RANSAC (Try swapping Left/Right sides)
        # This handles cases where the model confuses left/right penalty areas
        dst_points_mirrored = dst_points.copy()
        dst_points_mirrored[:, 0] *= -1  # Negate X coordinate

        H2, mask2 = cv2.findHomography(src_points, dst_points_mirrored, cv2.RANSAC, self.ransac_reproj_threshold)
        inliers2 = np.sum(mask2.ravel() == 1) if H2 is not None else 0

        # Select best homography
        if inliers2 > inliers1 and inliers2 >= self.min_inliers:
            # print(f"  Using mirrored homography ({inliers2} vs {inliers1} inliers)")
            self.H = H2
            self.inliers = mask2.ravel() == 1
            self.num_inliers = inliers2
        else:
            self.H = H1
            self.inliers = mask1.ravel() == 1 if H1 is not None else np.zeros(len(src_points), dtype=bool)
            self.num_inliers = inliers1

        if self.H is None:
            # Self-correction: Use fallback if available
            self.frames_since_valid += 1
            if self.last_valid_H is not None and self.frames_since_valid <= self.max_fallback_frames:
                # Decay confidence: apply slight blur to old H as time passes
                decay = 1.0 - (self.frames_since_valid / self.max_fallback_frames) * 0.1
                self.H_smoothed = self.last_valid_H * decay
                return True  # Use fallback
            return False

        # Require minimum inliers for a reliable homography
        if self.num_inliers < self.min_inliers:
            print(f"Not enough inliers: {self.num_inliers} < {self.min_inliers}")
            self.frames_since_valid += 1
            # Keep previous homography if available
            if self.last_valid_H is not None and self.frames_since_valid <= self.max_fallback_frames:
                self.H_smoothed = self.last_valid_H
                return True  # Use old homography
            return False

        # Reset fallback counter on successful estimation
        self.frames_since_valid = 0

        # Check if new homography would cause extreme position changes
        if self.H_smoothed is not None:
            # Test with a few reference points
            test_points = np.array([[640, 360], [320, 540], [960, 540]], dtype=np.float32)

            # Project with old and new H
            old_pts = self._project_with_H(test_points, self.H_smoothed)
            new_pts = self._project_with_H(test_points, self.H)

            if old_pts is not None and new_pts is not None:
                # Check max displacement
                displacements = np.linalg.norm(new_pts - old_pts, axis=1)
                max_disp = np.max(displacements)

                if max_disp > 20.0:  # More than 20m jump is suspicious
                    # print(f"Rejecting homography with {max_disp:.1f}m displacement")
                    # Keep old homography
                    return True

        # Temporal smoothing: blend with previous homography
        if self.H_smoothed is None:
            self.H_smoothed = self.H.copy()
        else:
            # Blend new homography with old (exponential moving average)
            self.H_smoothed = (1 - self.smoothing_alpha) * self.H_smoothed + self.smoothing_alpha * self.H

        # Self-correction: Store as last valid for fallback
        self.last_valid_H = self.H_smoothed.copy()

        # Compute inverse for projecting pitch to image
        try:
            self.H_inv = np.linalg.inv(self.H_smoothed)
        except np.linalg.LinAlgError:
            self.H_inv = None

        print(f"Homography estimated with {self.num_inliers}/{len(src_points)} inliers")
        return True

    def project_to_pitch(
        self,
        points: np.ndarray,  # [N, 2] pixel coordinates
    ) -> Optional[np.ndarray]:
        """
        Project image points to pitch coordinates.

        Args:
            points: [N, 2] array of (x, y) pixel coordinates

        Returns:
            [N, 2] array of (x, y) pitch coordinates in meters, or None if no homography
        """
        # Use smoothed homography for stability
        H = self.H_smoothed if self.H_smoothed is not None else self.H
        if H is None:
            return None

        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, 2)

        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.hstack([points, ones])  # [N, 3]

        # Apply homography
        projected = (H @ points_h.T).T  # [N, 3]

        # Convert from homogeneous
        projected = projected[:, :2] / projected[:, 2:3]

        return projected

    def project_to_image(
        self,
        pitch_points: np.ndarray,  # [N, 2] pitch coordinates in meters
    ) -> Optional[np.ndarray]:
        """
        Project pitch coordinates back to image pixels.

        Args:
            pitch_points: [N, 2] array of (x, y) pitch coordinates

        Returns:
            [N, 2] array of (x, y) pixel coordinates, or None if no inverse homography
        """
        if self.H_inv is None:
            return None

        pitch_points = np.asarray(pitch_points, dtype=np.float32)
        if pitch_points.ndim == 1:
            pitch_points = pitch_points.reshape(1, 2)

        ones = np.ones((pitch_points.shape[0], 1), dtype=np.float32)
        points_h = np.hstack([pitch_points, ones])

        projected = (self.H_inv @ points_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]

        return projected

    def get_player_foot_position(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Get the foot position of a player from their bounding box.

        Uses bottom-center of bounding box as foot position.

        Args:
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            (x, y) pixel coordinates of foot position
        """
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2
        foot_y = y2  # Bottom of bounding box
        return foot_x, foot_y

    def project_player_to_pitch(
        self,
        bbox: List[float],
    ) -> Optional[Tuple[float, float]]:
        """
        Project a player's position to pitch coordinates.

        Args:
            bbox: [x1, y1, x2, y2] player bounding box

        Returns:
            (x, y) pitch coordinates in meters, or None if projection fails
        """
        foot_x, foot_y = self.get_player_foot_position(bbox)

        projected = self.project_to_pitch(np.array([[foot_x, foot_y]]))
        if projected is None:
            return None

        x, y = float(projected[0, 0]), float(projected[0, 1])

        # Self-correction: Validate and clamp projection coordinates
        # Pitch is 105m x 68m centered at origin: x in [-52.5, 52.5], y in [-34, 34]
        # Allow margin for players slightly off-pitch (touchlines, behind goals)
        MAX_X = HALF_LENGTH + 5.0  # 57.5m from center
        MAX_Y = HALF_WIDTH + 5.0  # 39m from center

        # Reject completely invalid projections (numerical errors)
        if abs(x) > MAX_X * 2 or abs(y) > MAX_Y * 2:
            return None

        # Clamp to extended pitch bounds (soft correction)
        x_clamped = np.clip(x, -MAX_X, MAX_X)
        y_clamped = np.clip(y, -MAX_Y, MAX_Y)

        return x_clamped, y_clamped


class PitchVisualizer:
    """
    Visualizes player positions on a 2D pitch diagram.
    Optimized for GPU acceleration when available.
    """

    def __init__(
        self,
        width: int = 1050,  # Pixels (10 pixels per meter)
        height: int = 680,
        margin: int = 50,
        scale: float = 10.0,  # pixels per meter
        use_gpu: bool = True,
    ):
        self.width = width
        self.height = height
        self.margin = margin
        self.scale = scale

        self.pitch_color = (34, 139, 34)  # green
        self.line_color = (255, 255, 255)  # white
        self.team1_color = (0, 0, 255)  # red
        self.team2_color = (255, 0, 0)  # blue
        self.referee_color = (0, 255, 255)  # yellow
        self.unknown_color = (128, 128, 128)  # gray

        # spatial topology smoothing params
        self.prev_inf_gpu = None  # GPU tensor for temporal smoothing
        self.heatmap_alpha = 0.1

        self.base_pitch = self._draw_pitch()

        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self._init_gpu_grids()

        # Pre-compute pitch mask for blending
        self._init_pitch_mask()

    def _init_gpu_grids(self):
        """Pre-compute coordinate grids on GPU for spatial dominance."""
        h_img = self.height + 2 * self.margin
        w_img = self.width + 2 * self.margin

        self.scale_factor = 0.25
        h_small = int(h_img * self.scale_factor)
        w_small = int(w_img * self.scale_factor)
        self.h_small = h_small
        self.w_small = w_small
        self.h_img = h_img
        self.w_img = w_img

        x_indices = torch.arange(w_small, device=self.device, dtype=torch.float32)
        y_indices = torch.arange(h_small, device=self.device, dtype=torch.float32)
        Y_pix, X_pix = torch.meshgrid(y_indices, x_indices, indexing='ij')

        self.X_meters = (X_pix / self.scale_factor - self.margin) / self.scale - HALF_LENGTH
        self.Y_meters = (Y_pix / self.scale_factor - self.margin) / self.scale - HALF_WIDTH

        self.inf_t0_gpu = torch.zeros((h_small, w_small), device=self.device, dtype=torch.float32)
        self.inf_t1_gpu = torch.zeros((h_small, w_small), device=self.device, dtype=torch.float32)

        if DEBUG_TIMING:
            print(f"[GPU] Initialized grids on {self.device}, shape: {h_small}x{w_small}")

    def _init_pitch_mask(self):
        """Pre-compute pitch mask for heatmap blending."""
        tl = self._pitch_to_pixel(-HALF_LENGTH, HALF_WIDTH)
        br = self._pitch_to_pixel(HALF_LENGTH, -HALF_WIDTH)

        x_min = min(tl[0], br[0])
        x_max = max(tl[0], br[0])
        y_min = min(tl[1], br[1])
        y_max = max(tl[1], br[1])

        self.pitch_mask = np.zeros((self.h_img, self.w_img), dtype=np.uint8)
        cv2.rectangle(self.pitch_mask, (x_min, y_min), (x_max, y_max), 255, -1)

        self.alpha_mask = np.zeros((self.h_img, self.w_img), dtype=np.float32)
        self.alpha_mask[self.pitch_mask > 0] = 0.4

        self.overlay_buffer = np.zeros((self.h_img, self.w_img, 3), dtype=np.uint8)

    def _pitch_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert pitch coordinates (meters) to pixel coordinates."""
        px = int(self.margin + (x + HALF_LENGTH) * self.scale)
        py = int(self.margin + (y + HALF_WIDTH) * self.scale)  # flip y-axis to reflect image coords
        return px, py

    def _draw_pitch(self) -> np.ndarray:
        """Draw the base pitch with all markings."""
        total_width = self.width + 2 * self.margin
        total_height = self.height + 2 * self.margin
        img = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        # green pitch area
        cv2.rectangle(
            img, (self.margin, self.margin), (self.margin + self.width, self.margin + self.height), self.pitch_color, -1
        )

        # outer boundary
        cv2.rectangle(
            img,
            self._pitch_to_pixel(-HALF_LENGTH, HALF_WIDTH),
            self._pitch_to_pixel(HALF_LENGTH, -HALF_WIDTH),
            self.line_color,
            2,
        )

        # halfway line
        cv2.line(img, self._pitch_to_pixel(0, HALF_WIDTH), self._pitch_to_pixel(0, -HALF_WIDTH), self.line_color, 2)

        # center circle
        center = self._pitch_to_pixel(0, 0)
        radius = int(CENTER_CIRCLE_RADIUS * self.scale)
        cv2.circle(img, center, radius, self.line_color, 2)

        # center spot
        cv2.circle(img, center, 3, self.line_color, -1)

        # penalty areas
        cv2.rectangle(
            img,
            self._pitch_to_pixel(-HALF_LENGTH, PENALTY_AREA_WIDTH / 2),
            self._pitch_to_pixel(-HALF_LENGTH + PENALTY_AREA_DEPTH, -PENALTY_AREA_WIDTH / 2),
            self.line_color,
            2,
        )
        cv2.rectangle(
            img,
            self._pitch_to_pixel(HALF_LENGTH - PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH / 2),
            self._pitch_to_pixel(HALF_LENGTH, -PENALTY_AREA_WIDTH / 2),
            self.line_color,
            2,
        )

        # goal area boxes
        cv2.rectangle(
            img,
            self._pitch_to_pixel(-HALF_LENGTH, GOAL_AREA_WIDTH / 2),
            self._pitch_to_pixel(-HALF_LENGTH + GOAL_AREA_DEPTH, -GOAL_AREA_WIDTH / 2),
            self.line_color,
            2,
        )
        cv2.rectangle(
            img,
            self._pitch_to_pixel(HALF_LENGTH - GOAL_AREA_DEPTH, GOAL_AREA_WIDTH / 2),
            self._pitch_to_pixel(HALF_LENGTH, -GOAL_AREA_WIDTH / 2),
            self.line_color,
            2,
        )

        # penalty spots
        left_spot = self._pitch_to_pixel(-HALF_LENGTH + PENALTY_SPOT_DISTANCE, 0)
        right_spot = self._pitch_to_pixel(HALF_LENGTH - PENALTY_SPOT_DISTANCE, 0)
        cv2.circle(img, left_spot, 3, self.line_color, -1)
        cv2.circle(img, right_spot, 3, self.line_color, -1)
        # penalty arcs
        cv2.ellipse(img, left_spot, (radius, radius), 0, -53, 53, self.line_color, 2)
        cv2.ellipse(img, right_spot, (radius, radius), 180, -53, 53, self.line_color, 2)

        # goal boxes
        goal_depth = 2.44
        cv2.rectangle(
            img,
            self._pitch_to_pixel(-HALF_LENGTH - goal_depth, GOAL_WIDTH / 2),
            self._pitch_to_pixel(-HALF_LENGTH, -GOAL_WIDTH / 2),
            self.line_color,
            2,
        )
        cv2.rectangle(
            img,
            self._pitch_to_pixel(HALF_LENGTH, GOAL_WIDTH / 2),
            self._pitch_to_pixel(HALF_LENGTH + goal_depth, -GOAL_WIDTH / 2),
            self.line_color,
            2,
        )

        return img

    def draw_players(
        self,
        pitch_positions: List[Tuple[float, float, int, Optional[int]]],  # [(x, y, track_id, team_id), ...]
    ) -> np.ndarray:
        """
        Draw players on the pitch.

        Args:
            pitch_positions: List of (x, y, track_id, team_id) tuples
                - x, y: pitch coordinates in meters
                - track_id: unique player ID
                - team_id: 0, 1, or None for unknown

        Returns:
            Image with players drawn
        """
        img = self.base_pitch.copy()

        for x, y, track_id, team_id in pitch_positions:
            # Skip if outside pitch bounds (with some margin)
            if abs(x) > HALF_LENGTH + 5 or abs(y) > HALF_WIDTH + 5:
                continue

            px, py = self._pitch_to_pixel(x, y)

            # Choose color based on team
            if team_id == 0:
                color = self.team1_color
            elif team_id == 1:
                color = self.team2_color
            elif team_id == 2:
                color = self.referee_color
            else:
                color = self.unknown_color

            # Draw player dot
            cv2.circle(img, (px, py), 8, color, -1)
            cv2.circle(img, (px, py), 8, (0, 0, 0), 2)  # Black border

            # Draw track ID
            cv2.putText(img, str(track_id), (px - 5, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return img

    @timed
    def draw_with_trails(
        self,
        pitch_positions: List[Tuple[float, float, float, float, int, Optional[int]]],  # x, y, vx, vy, id, team
        history: Dict[int, List[Tuple[float, float]]],  # track_id -> [(x, y), ...]
        trail_length: int = 30,
        draw_vectors: bool = True,
        draw_dominance: bool = True,
    ) -> np.ndarray:
        """
        Draw players with movement trails, directional vectors, and spatial dominance.

        Args:
            pitch_positions: Current positions with velocity (x, y, vx, vy, id, team)
            history: Historical positions per track_id
            trail_length: Maximum trail points to show
            draw_vectors: Whether to draw velocity vectors
            draw_dominance: Whether to draw spatial dominance heatmap

        Returns:
            Image with visualization
        """
        img = self.base_pitch.copy()

        # 1. Draw Spatial Dominance Heatmap
        if draw_dominance:
            self._draw_heatmap(img, pitch_positions)

        # 2. Draw trails
        self._draw_trails(img, history, trail_length)

        # 3. Draw players and vectors
        self._draw_players(img, pitch_positions, draw_vectors)

        return img

    @timed
    def _draw_heatmap(self, img: np.ndarray, pitch_positions: list):
        """Draw spatial dominance heatmap on the image."""
        # Get raw influence maps for both teams
        inf_t0, inf_t1 = self._compute_spatial_dominance(pitch_positions)

        if inf_t0 is None or inf_t1 is None:
            return

        # Normalize to probability [0, 1] (in-place)
        total = inf_t0 + inf_t1
        total += 1e-6  # avoid division by zero
        np.divide(inf_t1, total, out=inf_t1)
        np.divide(inf_t0, total, out=inf_t0)

        # Use pre-allocated overlay buffer (avoid allocation)
        self.overlay_buffer[:, :, 0] = (inf_t1 * 255).astype(np.uint8)  # Blue
        self.overlay_buffer[:, :, 1] = 0  # Green
        self.overlay_buffer[:, :, 2] = (inf_t0 * 255).astype(np.uint8)  # Red

        # Mask overlay to only show within pitch bounds
        self.overlay_buffer[self.pitch_mask == 0] = 0

        # Use OpenCV's optimized addWeighted function
        cv2.addWeighted(img, 0.6, self.overlay_buffer, 0.4, 0, dst=img)

    def _draw_trails(self, img: np.ndarray, history: Dict[int, List[Tuple[float, float]]], trail_length: int):
        """Draw movement trails for players."""
        for track_id, positions in history.items():
            if len(positions) < 2:
                continue

            recent = positions[-trail_length:]
            for i in range(len(recent) - 1):
                alpha = (i + 1) / len(recent)
                x1, y1 = recent[i]
                x2, y2 = recent[i + 1]

                if abs(x1) > HALF_LENGTH + 5 or abs(y1) > HALF_WIDTH + 5:
                    continue
                if abs(x2) > HALF_LENGTH + 5 or abs(y2) > HALF_WIDTH + 5:
                    continue

                p1 = self._pitch_to_pixel(x1, y1)
                p2 = self._pitch_to_pixel(x2, y2)

                gray = int(100 + 155 * alpha)
                cv2.line(img, p1, p2, (gray, gray, gray), 2)

    def _draw_players(self, img: np.ndarray, pitch_positions: list, draw_vectors: bool):
        """Draw player dots and velocity vectors."""
        for x, y, vx, vy, track_id, team_id in pitch_positions:
            if abs(x) > HALF_LENGTH + 5 or abs(y) > HALF_WIDTH + 5:
                continue

            px, py = self._pitch_to_pixel(x, y)

            if team_id == 0:
                color = self.team1_color
            elif team_id == 1:
                color = self.team2_color
            elif team_id == 2:
                color = self.referee_color
            else:
                color = self.unknown_color

            if draw_vectors and (abs(vx) > 0.1 or abs(vy) > 0.1):
                end_x = x + vx * 15
                end_y = y + vy * 15
                px_end, py_end = self._pitch_to_pixel(end_x, end_y)
                cv2.arrowedLine(img, (px, py), (px_end, py_end), color, 2, tipLength=0.3)

            cv2.circle(img, (px, py), 8, color, -1)
            cv2.circle(img, (px, py), 8, (0, 0, 0), 2)
            cv2.putText(img, str(track_id), (px - 5, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    @timed
    def _compute_spatial_dominance(self, players, resolution=2.0):
        """
        Compute spatial dominance map based on player positions and velocities.
        GPU-accelerated with vectorized operations.
        Returns raw influence maps for Team 0 and Team 1.
        """
        # Filter players by team and extract data
        t0_players = []
        t1_players = []

        for px, py, vx, vy, _, team_id in players:
            if team_id == 0:
                t0_players.append((px, py, vx, vy))
            elif team_id == 1:
                t1_players.append((px, py, vx, vy))

        # Compute both team influences on GPU
        inf_t0_gpu, inf_t1_gpu = self._compute_both_teams_influence_gpu(t0_players, t1_players)

        # Stack and resize both at once (single kernel call)
        stacked = torch.stack([inf_t0_gpu, inf_t1_gpu], dim=0).unsqueeze(0)  # [1, 2, H, W]
        resized = torch.nn.functional.interpolate(stacked, size=(self.h_img, self.w_img), mode='nearest').squeeze(
            0
        )  # [2, H, W]

        # Temporal smoothing on GPU before transfer
        if self.prev_inf_gpu is not None:
            resized = self.prev_inf_gpu * (1 - self.heatmap_alpha) + resized * self.heatmap_alpha

        self.prev_inf_gpu = resized

        # Single CPU transfer
        result = resized.cpu().numpy()
        return result[0], result[1]

    @timed
    def _compute_both_teams_influence_gpu(self, t0_players: List[tuple], t1_players: List[tuple]):
        """
        Compute influence maps for both teams in a single GPU kernel.
        Returns tensors on GPU to avoid intermediate transfers.
        """
        zeros = torch.zeros((self.h_small, self.w_small), device=self.device, dtype=torch.float32)

        # Prepare tensors for both teams
        if not t0_players:
            inf_t0 = zeros.clone()
        else:
            inf_t0 = self._compute_influence_tensor(t0_players)

        if not t1_players:
            inf_t1 = zeros.clone()
        else:
            inf_t1 = self._compute_influence_tensor(t1_players)

        return inf_t0, inf_t1

    def _compute_influence_tensor(self, players: List[tuple]) -> torch.Tensor:
        """
        Compute influence map for players, returning a GPU tensor.
        """
        # Convert player data to tensors
        player_data = torch.tensor(players, device=self.device, dtype=torch.float32)
        px = player_data[:, 0]
        py = player_data[:, 1]
        vx = player_data[:, 2]
        vy = player_data[:, 3]

        # Compute per-player parameters
        speed = torch.sqrt(vx**2 + vy**2)
        angle = torch.atan2(vy, vx)

        # Momentum-shifted centers
        mu_x = px + vx * 15.0
        mu_y = py + vy * 15.0

        # Anisotropic sigma
        sigma_x = 4.0 * (1 + speed * 0.5)
        sigma_y = 4.0 / (1 + speed * 0.2)

        # Pre-compute trig values
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        # Expand for broadcasting: [N] -> [N, 1, 1]
        mu_x = mu_x.view(-1, 1, 1)
        mu_y = mu_y.view(-1, 1, 1)
        sigma_x = sigma_x.view(-1, 1, 1)
        sigma_y = sigma_y.view(-1, 1, 1)
        cos_angle = cos_angle.view(-1, 1, 1)
        sin_angle = sin_angle.view(-1, 1, 1)

        # Expand grid: [H, W] -> [1, H, W]
        X = self.X_meters.unsqueeze(0)
        Y = self.Y_meters.unsqueeze(0)

        # Compute distance from each player's center: [N, H, W]
        dx = X - mu_x
        dy = Y - mu_y

        # Rotate coordinates
        dx_rot = dx * cos_angle + dy * sin_angle
        dy_rot = -dx * sin_angle + dy * cos_angle

        # Compute Gaussian for all players: [N, H, W]
        gaussian = torch.exp(-(dx_rot**2 / (2 * sigma_x**2) + dy_rot**2 / (2 * sigma_y**2)))

        # Sum over all players: [H, W]
        return gaussian.sum(dim=0)
