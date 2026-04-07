"""
3D lifting of 2D player poses to world coordinates.

Pipeline:
    1. build_camera_model(): recover K, R, t from pitch RANSAC correspondences via solvePnP
    2. PoseLift3D.lift(): ray-cast ankle pixel through K[R|t] to z=0 ground plane,
       position full skeleton relative to foot contact point
    3. PoseLift3D.refine_lbfgs(): optimize per-player translation to minimise 2D
       reprojection error (LBFGS with strong Wolfe line search)

Coordinate system:
    - World frame: pitch center = (0, 0, 0), x along length (±52.5m), y along width
      (±34m), z pointing up. Ground plane is z=0.
    - Camera frame: standard OpenCV convention (z forward, x right, y down).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from torchkick.models.pose import PoseResult


@dataclass
class CameraModel:
    """
    Per-frame calibrated camera parameters.

    All arrays use float64 for numerical stability in reprojection.
    """

    K: np.ndarray  # [3, 3] intrinsic matrix
    R: np.ndarray  # [3, 3] rotation matrix (world → camera)
    t: np.ndarray  # [3]    translation vector
    rvec: np.ndarray  # [3]    Rodrigues rotation vector
    image_size: Tuple[int, int]  # (height, width)

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project [N, 3] world points to [N, 2] pixel coordinates.

        Args:
            points_3d: [N, 3] float64 world coordinates.

        Returns:
            [N, 2] float64 pixel coordinates.
        """
        pts = points_3d.reshape(-1, 1, 3).astype(np.float64)
        projected, _ = cv2.projectPoints(pts, self.rvec, self.t, self.K, None)
        return projected.reshape(-1, 2)


def build_camera_model(
    matched_src: np.ndarray,
    matched_dst: np.ndarray,
    image_size: Tuple[int, int],
    approx_K: Optional[np.ndarray] = None,
) -> Optional[CameraModel]:
    """
    Recover camera pose from 2D image ↔ 2D pitch correspondences via PnP.

    The pitch correspondences are treated as 3D points with z=0 (all pitch
    landmarks lie on the ground plane). Uses the approximate K from homography
    decomposition as the initial intrinsics estimate, then refines via RANSAC
    + iterative PnP on inliers.

    Args:
        matched_src: [N, 2] float32 — image pixel coordinates (RANSAC inliers).
        matched_dst: [N, 2] float32 — pitch 2D coordinates in meters (x, y).
        image_size: (height, width) of the source frame.
        approx_K: [3, 3] approximate intrinsic matrix. If None, builds from
                  image dimensions assuming ~60° FOV (f = max(h, w)).

    Returns:
        CameraModel or None if fewer than 4 valid correspondences.
    """
    if matched_src is None or matched_dst is None or len(matched_src) < 4:
        return None

    h, w = image_size
    if approx_K is None:
        f = float(max(h, w))
        approx_K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    else:
        approx_K = approx_K.astype(np.float64)

    # Build 3D object points: pitch (x, y) with z=0
    object_points = np.hstack(
        [
            matched_dst.astype(np.float64),
            np.zeros((len(matched_dst), 1), dtype=np.float64),
        ]
    )  # [N, 3]
    image_points = matched_src.astype(np.float64)  # [N, 2]

    # RANSAC PnP
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            approx_K,
            None,  # no distortion
            reprojectionError=3.0,
            confidence=0.99,
            iterationsCount=200,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None

    if not success or inliers is None or len(inliers) < 4:
        return None

    # Refine on inliers only
    inlier_idx = inliers.ravel()
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points[inlier_idx],
            image_points[inlier_idx],
            approx_K,
            None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None

    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)

    return CameraModel(
        K=approx_K,
        R=R.astype(np.float64),
        t=tvec.ravel().astype(np.float64),
        rvec=rvec.ravel().astype(np.float64),
        image_size=image_size,
    )


class PoseLift3D:
    """
    Lifts 2D COCO-17 keypoints to 3D world coordinates via ray casting.

    Algorithm:
        1. Find the best ankle (highest confidence above threshold).
        2. Cast a ray from ankle 2D pixel through K[R|t] to z=0 ground plane
           → foot world position.
        3. Scale canonical human skeleton proportions to estimated player height.
        4. Position all 17 joints relative to foot contact.
        5. Optionally refine per-player translation via LBFGS reprojection minimisation.

    Args:
        use_lbfgs: Run LBFGS reprojection refinement (recommended for accuracy).
        lbfgs_max_iter: Maximum LBFGS iterations per player.
        ankle_conf_threshold: Minimum ankle confidence to use for ground contact.
        min_visible_for_lbfgs: Minimum number of visible keypoints required for LBFGS.
    """

    # Canonical human skeleton in local body frame (meters).
    # Origin at ankle midpoint, y-axis up, proportions for 1.75m player.
    # Indices follow COCO-17 ordering.
    CANONICAL_SKELETON_LOCAL: np.ndarray = np.array(
        [
            [0.00, 1.70, 0.00],  # 0  nose
            [-0.05, 1.68, 0.00],  # 1  left_eye
            [0.05, 1.68, 0.00],  # 2  right_eye
            [-0.10, 1.65, 0.00],  # 3  left_ear
            [0.10, 1.65, 0.00],  # 4  right_ear
            [-0.20, 1.42, 0.00],  # 5  left_shoulder
            [0.20, 1.42, 0.00],  # 6  right_shoulder
            [-0.38, 1.25, 0.00],  # 7  left_elbow
            [0.38, 1.25, 0.00],  # 8  right_elbow
            [-0.60, 1.00, 0.00],  # 9  left_wrist
            [0.60, 1.00, 0.00],  # 10 right_wrist
            [-0.12, 0.92, 0.00],  # 11 left_hip
            [0.12, 0.92, 0.00],  # 12 right_hip
            [-0.10, 0.48, 0.00],  # 13 left_knee
            [0.10, 0.48, 0.00],  # 14 right_knee
            [-0.10, 0.00, 0.00],  # 15 left_ankle
            [0.10, 0.00, 0.00],  # 16 right_ankle
        ],
        dtype=np.float64,
    )

    _CANONICAL_HEIGHT: float = 1.75  # metres

    def __init__(
        self,
        use_lbfgs: bool = True,
        lbfgs_max_iter: int = 50,
        ankle_conf_threshold: float = 0.3,
        min_visible_for_lbfgs: int = 4,
    ) -> None:
        self.use_lbfgs = use_lbfgs
        self.lbfgs_max_iter = lbfgs_max_iter
        self.ankle_conf_threshold = ankle_conf_threshold
        self.min_visible_for_lbfgs = min_visible_for_lbfgs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lift(
        self,
        keypoints_2d: np.ndarray,
        scores_2d: np.ndarray,
        camera: CameraModel,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """
        Lift a single player's 2D pose to 3D world coordinates.

        Args:
            keypoints_2d: [17, 2] float32 full-frame pixel (x, y).
            scores_2d: [17] float32 per-keypoint confidence.
            camera: Calibrated camera model for this frame.
            bbox: Optional [x1, y1, x2, y2] for height estimation fallback.

        Returns:
            [17, 3] float64 world coordinates (metres).
        """
        # 1. Find best ankle for ground contact
        foot_world = self._get_foot_world(keypoints_2d, scores_2d, camera, bbox)

        # 2. Estimate player height from bbox pixel height
        scale = 1.0
        if bbox is not None:
            scale = self._estimate_height_scale(bbox, camera)

        # 3. Position canonical skeleton relative to foot
        local = self.CANONICAL_SKELETON_LOCAL * scale  # [17, 3]
        kp3d = foot_world[np.newaxis, :] + local  # [17, 3]

        # 4. Optional LBFGS refinement
        if self.use_lbfgs:
            visible = scores_2d >= self.ankle_conf_threshold
            if visible.sum() >= self.min_visible_for_lbfgs:
                kp3d = self.refine_lbfgs(keypoints_2d, scores_2d, kp3d, camera)

        return kp3d

    def lift_batch(
        self,
        pose_results: List[PoseResult],
        camera: Optional[CameraModel],
    ) -> List[np.ndarray]:
        """
        Lift all players in a frame.

        Args:
            pose_results: List of PoseResult from BodyPoseDetector.
            camera: CameraModel for this frame, or None if not yet available.

        Returns:
            List of [17, 3] arrays (one per player).
            Returns zeros if camera is None or lifting fails for a player.
        """
        results = []
        for pr in pose_results:
            if camera is None:
                results.append(np.zeros((17, 3), dtype=np.float64))
                continue
            try:
                kp3d = self.lift(pr.keypoints, pr.scores, camera, bbox=pr.bbox)
            except Exception:
                kp3d = np.zeros((17, 3), dtype=np.float64)
            results.append(kp3d)
        return results

    def refine_lbfgs(
        self,
        keypoints_2d: np.ndarray,
        scores_2d: np.ndarray,
        keypoints_3d_init: np.ndarray,
        camera: CameraModel,
        trans_bounds_xy: float = 3.0,
        trans_bounds_z: float = 0.2,
    ) -> np.ndarray:
        """
        Optimise a per-player translation offset dt to minimise weighted
        2D reprojection error via LBFGS.

        The objective:
            L(dt) = sum_i scores_i * ||project(kp3d_i + dt, camera) - kp2d_i||^2

        Args:
            keypoints_2d: [17, 2] float32 pixel observations.
            scores_2d: [17] float32 confidence weights.
            keypoints_3d_init: [17, 3] float64 initial world estimate.
            camera: Calibrated camera for this frame.
            trans_bounds_xy: ±bound in x, y (metres).
            trans_bounds_z: ±bound in z (metres).

        Returns:
            Refined [17, 3] float64 world coordinates.
        """
        kp3d_t = torch.tensor(keypoints_3d_init, dtype=torch.float64)
        kp2d_t = torch.tensor(keypoints_2d, dtype=torch.float64)
        scores_t = torch.tensor(scores_2d, dtype=torch.float64)
        K_t = torch.tensor(camera.K, dtype=torch.float64)
        R_t = torch.tensor(camera.R, dtype=torch.float64)
        t_t = torch.tensor(camera.t, dtype=torch.float64)

        # Learnable translation offset, initialised at zero
        dt = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        bounds = torch.tensor([trans_bounds_xy, trans_bounds_xy, trans_bounds_z], dtype=torch.float64)

        optimizer = torch.optim.LBFGS(
            [dt],
            max_iter=self.lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            pts_shifted = kp3d_t + dt.unsqueeze(0)  # [17, 3]
            # Transform to camera space
            pts_cam = (R_t @ pts_shifted.T).T + t_t.unsqueeze(0)  # [17, 3]
            # Perspective divide
            z = pts_cam[:, 2:3].clamp(min=1e-4)
            pts_img = pts_cam[:, :2] / z  # normalised image plane
            # Apply K
            proj_x = K_t[0, 0] * pts_img[:, 0] + K_t[0, 2]
            proj_y = K_t[1, 1] * pts_img[:, 1] + K_t[1, 2]
            proj = torch.stack([proj_x, proj_y], dim=-1)  # [17, 2]
            diff = proj - kp2d_t
            loss = (scores_t * (diff**2).sum(dim=-1)).sum()
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            dt.clamp_(-bounds, bounds)

        offset = dt.detach().numpy()
        return keypoints_3d_init + offset[np.newaxis, :]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ray_ground_intersect(
        self,
        pixel_uv: np.ndarray,
        camera: CameraModel,
    ) -> Optional[np.ndarray]:
        """
        Compute the intersection of a ray through pixel_uv with the z=0 plane.

        Returns [3] world point, or None if the ray is parallel to the ground
        or the intersection is behind the camera.
        """
        u, v = float(pixel_uv[0]), float(pixel_uv[1])
        K_inv = np.linalg.inv(camera.K)
        # Ray direction in camera frame
        d_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        # Transform to world frame
        RT = camera.R.T  # world → camera inverse
        d_world = RT @ d_cam
        d_world /= np.linalg.norm(d_world)

        # Camera centre in world frame: C = -R.T @ t
        origin = -RT @ camera.t

        if abs(d_world[2]) < 1e-4:
            return None  # Ray nearly parallel to ground

        t_param = -origin[2] / d_world[2]
        if t_param < 0:
            return None  # Intersection behind camera

        return origin + t_param * d_world

    def _get_foot_world(
        self,
        keypoints_2d: np.ndarray,
        scores_2d: np.ndarray,
        camera: CameraModel,
        bbox: Optional[Tuple],
    ) -> np.ndarray:
        """
        Get the foot world position for ground contact.

        Tries ankles first; falls back to bbox bottom-centre projected via
        the 2D homography foot position (z=0).
        """
        l_score = scores_2d[15]
        r_score = scores_2d[16]

        best_ankle_uv = None
        if l_score >= self.ankle_conf_threshold or r_score >= self.ankle_conf_threshold:
            idx = 15 if l_score >= r_score else 16
            best_ankle_uv = keypoints_2d[idx]
        elif bbox is not None:
            # Fallback: bottom-centre of bounding box
            x1, y1, x2, y2 = bbox
            best_ankle_uv = np.array([(x1 + x2) / 2, y2], dtype=np.float32)

        if best_ankle_uv is not None:
            foot = self._ray_ground_intersect(best_ankle_uv, camera)
            if foot is not None:
                return foot

        # Last resort: return pitch origin
        return np.zeros(3, dtype=np.float64)

    def _estimate_height_scale(
        self,
        bbox: Tuple[float, float, float, float],
        camera: CameraModel,
    ) -> float:
        """
        Estimate player height scale from bounding box pixel height.

        Uses the camera's approximate focal length and distance to estimate
        the metric height of a player from their pixel bbox height. Clips
        to [0.8, 1.2] to prevent degenerate scaling.
        """
        x1, y1, x2, y2 = bbox
        bbox_pixel_h = float(y2 - y1)
        if bbox_pixel_h < 1:
            return 1.0

        # Bottom-centre → world foot
        foot_uv = np.array([(x1 + x2) / 2, y2], dtype=np.float32)
        foot_world = self._ray_ground_intersect(foot_uv, camera)
        if foot_world is None:
            return 1.0

        # Top-centre → world via ray (approximate — assumes player is vertical)
        head_uv = np.array([(x1 + x2) / 2, y1], dtype=np.float32)
        head_world = self._ray_ground_intersect(head_uv, camera)
        if head_world is None:
            return 1.0

        # Player height from world points (z-difference after accounting for ground slope)
        metric_h = float(np.linalg.norm(head_world - foot_world))
        if metric_h < 0.5 or metric_h > 2.5:
            return 1.0

        return float(np.clip(metric_h / self._CANONICAL_HEIGHT, 0.8, 1.2))
