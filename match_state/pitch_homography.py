import cv2
import numpy as np
import torch
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple


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


class PitchPoint(BaseModel):
    """A point on the pitch in real-world coordinates (meters)."""

    x: float  # Along touchline, -52.5 (left) to 52.5 (right)
    y: float  # Along goal line, -34 (bottom) to 34 (top)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


PITCH_LINE_COORDINATES: Dict[str, List[Tuple[int, PitchPoint]]] = {
    "Side line top": [
        (0, PitchPoint(-HALF_LENGTH, HALF_WIDTH)),  # top-left corner
        (1, PitchPoint(HALF_LENGTH, HALF_WIDTH)),  # top-right corner
    ],
    "Side line bottom": [
        (0, PitchPoint(-HALF_LENGTH, -HALF_WIDTH)),  # bottom-left corner
        (1, PitchPoint(HALF_LENGTH, -HALF_WIDTH)),  # bottom-right corner
    ],
    "Side line left": [
        (0, PitchPoint(-HALF_LENGTH, -HALF_WIDTH)),  # bottom-left corner
        (1, PitchPoint(-HALF_LENGTH, HALF_WIDTH)),  # top-left corner
    ],
    "Side line right": [
        (0, PitchPoint(HALF_LENGTH, -HALF_WIDTH)),  # bottom-right corner
        (1, PitchPoint(HALF_LENGTH, HALF_WIDTH)),  # top-right corner
    ],
    "Middle line": [
        (0, PitchPoint(0, -HALF_WIDTH)),  # bottom of halfway line
        (1, PitchPoint(0, HALF_WIDTH)),  # top of halfway line
    ],
    "Big rect. left main": [
        (0, PitchPoint(-HALF_LENGTH + PENALTY_AREA_DEPTH, -PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(-HALF_LENGTH + PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. left top": [
        (0, PitchPoint(-HALF_LENGTH, PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(-HALF_LENGTH + PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. left bottom": [
        (0, PitchPoint(-HALF_LENGTH, -PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(-HALF_LENGTH + PENALTY_AREA_DEPTH, -PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. right main": [
        (0, PitchPoint(HALF_LENGTH - PENALTY_AREA_DEPTH, -PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(HALF_LENGTH - PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. right top": [
        (0, PitchPoint(HALF_LENGTH - PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(HALF_LENGTH, PENALTY_AREA_WIDTH / 2)),
    ],
    "Big rect. right bottom": [
        (0, PitchPoint(HALF_LENGTH - PENALTY_AREA_DEPTH, -PENALTY_AREA_WIDTH / 2)),
        (1, PitchPoint(HALF_LENGTH, -PENALTY_AREA_WIDTH / 2)),
    ],
    "Small rect. left main": [
        (0, PitchPoint(-HALF_LENGTH + GOAL_AREA_DEPTH, -GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(-HALF_LENGTH + GOAL_AREA_DEPTH, GOAL_AREA_WIDTH / 2)),
    ],
    "Small rect. left top": [
        (0, PitchPoint(-HALF_LENGTH, GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(-HALF_LENGTH + GOAL_AREA_DEPTH, GOAL_AREA_WIDTH / 2)),
    ],
    "Small rect. left bottom": [
        (0, PitchPoint(-HALF_LENGTH, -GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(-HALF_LENGTH + GOAL_AREA_DEPTH, -GOAL_AREA_WIDTH / 2)),
    ],
    "Small rect. right main": [
        (0, PitchPoint(HALF_LENGTH - GOAL_AREA_DEPTH, -GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(HALF_LENGTH - GOAL_AREA_DEPTH, GOAL_AREA_WIDTH / 2)),
    ],
    "Small rect. right top": [
        (0, PitchPoint(HALF_LENGTH - GOAL_AREA_DEPTH, GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(HALF_LENGTH, GOAL_AREA_WIDTH / 2)),
    ],
    "Small rect. right bottom": [
        (0, PitchPoint(HALF_LENGTH - GOAL_AREA_DEPTH, -GOAL_AREA_WIDTH / 2)),
        (1, PitchPoint(HALF_LENGTH, -GOAL_AREA_WIDTH / 2)),
    ],
    "Goal left crossbar": [
        (0, PitchPoint(-HALF_LENGTH, -GOAL_WIDTH / 2)),
        (1, PitchPoint(-HALF_LENGTH, GOAL_WIDTH / 2)),
    ],
    "Goal left post left ": [
        (0, PitchPoint(-HALF_LENGTH, -GOAL_WIDTH / 2)),
        (1, PitchPoint(-HALF_LENGTH, -GOAL_WIDTH / 2)),
    ],
    "Goal left post right": [
        (0, PitchPoint(-HALF_LENGTH, GOAL_WIDTH / 2)),
        (1, PitchPoint(-HALF_LENGTH, GOAL_WIDTH / 2)),
    ],
    "Goal right crossbar": [
        (0, PitchPoint(HALF_LENGTH, -GOAL_WIDTH / 2)),
        (1, PitchPoint(HALF_LENGTH, GOAL_WIDTH / 2)),
    ],
    "Goal right post left": [
        (0, PitchPoint(HALF_LENGTH, -GOAL_WIDTH / 2)),
        (1, PitchPoint(HALF_LENGTH, -GOAL_WIDTH / 2)),
    ],
    "Goal right post right": [
        (0, PitchPoint(HALF_LENGTH, GOAL_WIDTH / 2)),
        (1, PitchPoint(HALF_LENGTH, GOAL_WIDTH / 2)),
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
        points.append((i, PitchPoint(x, y)))
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
        smoothing_alpha: float = 0.3,
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

        # Import line classes
        from soccernet.calibration_data import LINE_CLASSES

        self.line_classes = LINE_CLASSES

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
            # print("Homography computation failed")
            return False

        # Require minimum inliers for a reliable homography
        if self.num_inliers < self.min_inliers:
            print(f"Not enough inliers: {self.num_inliers} < {self.min_inliers}")
            # Keep previous homography if available
            if self.H_smoothed is not None:
                return True  # Use old homography
            return False

        # Temporal smoothing: blend with previous homography
        if self.H_smoothed is None:
            self.H_smoothed = self.H.copy()
        else:
            # Blend new homography with old (exponential moving average)
            self.H_smoothed = (1 - self.smoothing_alpha) * self.H_smoothed + self.smoothing_alpha * self.H

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

        # Reject unrealistic positions (far outside pitch bounds)
        # Pitch is 105m x 68m, allow some margin
        if abs(x) > 60 or abs(y) > 40:
            return None

        return x, y


class PitchVisualizer:
    """
    Visualizes player positions on a 2D pitch diagram.
    """

    def __init__(
        self,
        width: int = 1050,  # Pixels (10 pixels per meter)
        height: int = 680,
        margin: int = 50,
        scale: float = 10.0,  # pixels per meter
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
        self.prev_inf_t0 = None
        self.prev_inf_t1 = None
        self.heatmap_alpha = 0.1

        self.base_pitch = self._draw_pitch()

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
            # Get raw influence maps for both teams
            inf_t0, inf_t1 = self._compute_spatial_dominance(pitch_positions)

            if inf_t0 is not None and inf_t1 is not None:
                # Normalize to probability [0, 1] to fill the pitch
                # Add small epsilon to avoid division by zero
                total = inf_t0 + inf_t1 + 1e-6
                prob_t1 = inf_t1 / total
                prob_t0 = inf_t0 / total

                # Create colored overlay
                # Team 0 (Red) -> Red Channel (2)
                # Team 1 (Blue) -> Blue Channel (0)
                # Mixed -> Purple

                overlay = np.zeros_like(img)
                overlay[:, :, 0] = (prob_t1 * 255).astype(np.uint8)  # Blue
                overlay[:, :, 2] = (prob_t0 * 255).astype(np.uint8)  # Red

                # Constant alpha for visibility
                alpha_val = 0.4

                # Mask out areas outside the pitch
                tl = self._pitch_to_pixel(-HALF_LENGTH, HALF_WIDTH)
                br = self._pitch_to_pixel(HALF_LENGTH, -HALF_WIDTH)

                # Create mask
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

                x_min = min(tl[0], br[0])
                x_max = max(tl[0], br[0])
                y_min = min(tl[1], br[1])
                y_max = max(tl[1], br[1])

                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

                # Apply mask to alpha
                alpha_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                alpha_mask[mask > 0] = alpha_val

                # Blend
                for c in range(3):
                    img[:, :, c] = (img[:, :, c] * (1 - alpha_mask) + overlay[:, :, c] * alpha_mask).astype(np.uint8)

        # 2. Draw trails
        for track_id, positions in history.items():
            if len(positions) < 2:
                continue

            recent = positions[-trail_length:]
            for i in range(len(recent) - 1):
                alpha = (i + 1) / len(recent)
                x1, y1 = recent[i]
                x2, y2 = recent[i + 1]

                p1 = self._pitch_to_pixel(x1, y1)
                p2 = self._pitch_to_pixel(x2, y2)

                if abs(x1) > HALF_LENGTH + 5 or abs(y1) > HALF_WIDTH + 5:
                    continue
                if abs(x2) > HALF_LENGTH + 5 or abs(y2) > HALF_WIDTH + 5:
                    continue

                gray = int(100 + 155 * alpha)
                cv2.line(img, p1, p2, (gray, gray, gray), 2)

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
                end_x = x + vx * 15  # 15 frames projection (~0.5s)
                end_y = y + vy * 15
                px_end, py_end = self._pitch_to_pixel(end_x, end_y)
                cv2.arrowedLine(img, (px, py), (px_end, py_end), color, 2, tipLength=0.3)

            cv2.circle(img, (px, py), 8, color, -1)
            cv2.circle(img, (px, py), 8, (0, 0, 0), 2)
            cv2.putText(img, str(track_id), (px - 5, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return img

    def _compute_spatial_dominance(self, players, resolution=2.0):
        """
        Compute spatial dominance map based on player positions and velocities.
        Returns raw influence maps for Team 0 and Team 1.
        """
        # we need to map pixels back to meters to evaluate the Gaussian
        h_img, w_img = self.height + 2 * self.margin, self.width + 2 * self.margin

        # first we create a pixel grid and downsample for speed
        scale_factor = 0.25
        h_small, w_small = int(h_img * scale_factor), int(w_img * scale_factor)

        x_indices = np.arange(w_small)
        y_indices = np.arange(h_small)
        X_pix, Y_pix = np.meshgrid(x_indices, y_indices)

        # pixel coordinates to pitch coordinates (meters)
        # is inverse of _pitch_to_pixel logic
        X_meters = (X_pix / scale_factor - self.margin) / self.scale - HALF_LENGTH
        Y_meters = (Y_pix / scale_factor - self.margin) / self.scale - HALF_WIDTH

        influence_t0 = np.zeros_like(X_meters)
        influence_t1 = np.zeros_like(X_meters)

        for px, py, vx, vy, _, team_id in players:
            if team_id not in [0, 1]:
                continue

            # gaussian ball of pixel velocity influence
            speed = np.sqrt(vx**2 + vy**2)
            angle = np.arctan2(vy, vx)  # NOTE: angle is used for directional influence

            # center based on momentum (0.5s projection)
            # this makes influence extend forward and lag behind
            mu_x = px + vx * 15.0  # 15 frames ~ 0.5s
            mu_y = py + vy * 15.0

            # base influence radius (meters)
            # elongate in direction of motion (momentum)
            sigma_x = 4.0 * (1 + speed * 0.5)
            # narrow cross-track influence as speed increases (harder to turn)
            sigma_y = 4.0 / (1 + speed * 0.2)

            # rotate grid coordinates to align with player velocity
            dx = X_meters - mu_x
            dy = Y_meters - mu_y

            # standard rotation for 2D points (i.e., rotate grid by -angle)
            dx_rot = dx * np.cos(angle) + dy * np.sin(angle)
            dy_rot = -dx * np.sin(angle) + dy * np.cos(angle)

            gaussian = np.exp(-(dx_rot**2 / (2 * sigma_x**2) + dy_rot**2 / (2 * sigma_y**2)))

            if team_id == 0:
                influence_t0 += gaussian
            else:
                influence_t1 += gaussian

        # resize back to full image size
        inf_t0_full = cv2.resize(influence_t0, (w_img, h_img))
        inf_t1_full = cv2.resize(influence_t1, (w_img, h_img))

        # temporal smoothing
        if self.prev_inf_t0 is not None:
            inf_t0_full = self.prev_inf_t0 * (1 - self.heatmap_alpha) + inf_t0_full * self.heatmap_alpha
            inf_t1_full = self.prev_inf_t1 * (1 - self.heatmap_alpha) + inf_t1_full * self.heatmap_alpha

        self.prev_inf_t0 = inf_t0_full
        self.prev_inf_t1 = inf_t1_full

        return inf_t0_full, inf_t1_full
