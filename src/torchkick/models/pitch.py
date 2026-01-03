"""
Pitch line keypoint detection models.

This module provides HRNet-based models for detecting pitch line
keypoints and lines, enabling camera calibration and homography
estimation.

Example:
    >>> from torchkick.models.pitch import PitchLineDetector
    >>> 
    >>> detector = PitchLineDetector(
    ...     weights_kp="weights/SV_kp",
    ...     weights_lines="weights/SV_lines",
    ... )
    >>> keypoints, lines, confidence = detector.detect(frame)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml

# Type aliases
TensorLike = Union[np.ndarray, torch.Tensor]


# Pitch line coordinates for visualization
LINE_COORDINATES_3D = [
    [[0.0, 54.16, 0.0], [16.5, 54.16, 0.0]],
    [[16.5, 13.84, 0.0], [16.5, 54.16, 0.0]],
    [[16.5, 13.84, 0.0], [0.0, 13.84, 0.0]],
    [[88.5, 54.16, 0.0], [105.0, 54.16, 0.0]],
    [[88.5, 13.84, 0.0], [88.5, 54.16, 0.0]],
    [[88.5, 13.84, 0.0], [105.0, 13.84, 0.0]],
    [[0.0, 37.66, -2.44], [0.0, 30.34, -2.44]],
    [[0.0, 37.66, 0.0], [0.0, 37.66, -2.44]],
    [[0.0, 30.34, 0.0], [0.0, 30.34, -2.44]],
    [[105.0, 37.66, -2.44], [105.0, 30.34, -2.44]],
    [[105.0, 30.34, 0.0], [105.0, 30.34, -2.44]],
    [[105.0, 37.66, 0.0], [105.0, 37.66, -2.44]],
    [[52.5, 0.0, 0.0], [52.5, 68, 0.0]],
    [[0.0, 68.0, 0.0], [105.0, 68.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 68.0, 0.0]],
    [[105.0, 0.0, 0.0], [105.0, 68.0, 0.0]],
    [[0.0, 0.0, 0.0], [105.0, 0.0, 0.0]],
    [[0.0, 43.16, 0.0], [5.5, 43.16, 0.0]],
    [[5.5, 43.16, 0.0], [5.5, 24.84, 0.0]],
    [[5.5, 24.84, 0.0], [0.0, 24.84, 0.0]],
    [[99.5, 43.16, 0.0], [105.0, 43.16, 0.0]],
    [[99.5, 43.16, 0.0], [99.5, 24.84, 0.0]],
    [[99.5, 24.84, 0.0], [105.0, 24.84, 0.0]],
]


def projection_from_cam_params(params_dict: Dict) -> np.ndarray:
    """
    Compute projection matrix from camera parameters.

    Args:
        params_dict: Camera parameters with "cam_params" key containing
            x_focal_length, y_focal_length, principal_point,
            position_meters, rotation_matrix.

    Returns:
        3x4 projection matrix P.
    """
    cam_params = params_dict["cam_params"]
    x_focal_length = cam_params["x_focal_length"]
    y_focal_length = cam_params["y_focal_length"]
    principal_point = np.array(cam_params["principal_point"])
    position_meters = np.array(cam_params["position_meters"])
    rotation = np.array(cam_params["rotation_matrix"])

    # Build projection: P = K @ [R | -R @ t]
    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array(
        [
            [x_focal_length, 0, principal_point[0]],
            [0, y_focal_length, principal_point[1]],
            [0, 0, 1],
        ]
    )
    P = Q @ (rotation @ It)

    return P


def project_lines_to_image(
    frame: np.ndarray,
    P: np.ndarray,
    line_color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 3,
) -> np.ndarray:
    """
    Draw projected pitch lines on frame.

    Args:
        frame: BGR image to draw on.
        P: 3x4 projection matrix.
        line_color: BGR color for lines.
        line_width: Line thickness.

    Returns:
        Image with lines drawn.
    """
    frame = frame.copy()

    for line in LINE_COORDINATES_3D:
        w1, w2 = line
        # Convert to centered coordinates
        i1 = P @ np.array([w1[0] - 52.5, w1[1] - 34, w1[2], 1])
        i2 = P @ np.array([w2[0] - 52.5, w2[1] - 34, w2[2], 1])
        i1 /= i1[-1]
        i2 /= i2[-1]
        cv2.line(
            frame,
            (int(i1[0]), int(i1[1])),
            (int(i2[0]), int(i2[1])),
            line_color,
            line_width,
        )

    return frame


class PitchLineDetector:
    """
    Detect pitch line keypoints using HRNet.

    Uses two HRNet models: one for keypoints and one for lines.
    Supports FP16 inference and CUDA stream parallelism for speed.

    Args:
        weights_kp: Path to keypoint model weights.
        weights_lines: Path to line model weights.
        config_kp: Path to keypoint model config YAML.
        config_lines: Path to line model config YAML.
        device: Torch device string.
        use_fp16: Use FP16 for faster inference.
        kp_threshold: Confidence threshold for keypoints.
        line_threshold: Confidence threshold for lines.

    Example:
        >>> detector = PitchLineDetector(
        ...     weights_kp="models/pitch/weights/SV_kp",
        ...     weights_lines="models/pitch/weights/SV_lines",
        ... )
        >>> kp_dict, lines_dict = detector.detect(frame)
    """

    def __init__(
        self,
        weights_kp: Union[str, Path],
        weights_lines: Union[str, Path],
        config_kp: Optional[Union[str, Path]] = None,
        config_lines: Optional[Union[str, Path]] = None,
        device: str = "cuda:0",
        use_fp16: bool = True,
        kp_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
    ) -> None:
        self.device = device
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold
        self.use_fp16 = use_fp16

        weights_kp = Path(weights_kp)
        weights_lines = Path(weights_lines)

        # Default config paths
        if config_kp is None:
            config_kp = weights_kp.parent / "config" / "hrnetv2_w48.yaml"
        if config_lines is None:
            config_lines = weights_lines.parent / "config" / "hrnetv2_w48_l.yaml"

        # Load configs
        with open(config_kp, "r") as f:
            self.cfg_kp = yaml.safe_load(f)
        with open(config_lines, "r") as f:
            self.cfg_lines = yaml.safe_load(f)

        # Load models (import here to avoid circular deps)
        from models.pitch.model.cls_hrnet import get_cls_net
        from models.pitch.model.cls_hrnet_l import get_cls_net as get_cls_net_l

        self.model_kp = get_cls_net(self.cfg_kp)
        self.model_kp.load_state_dict(torch.load(weights_kp, map_location=device))
        self.model_kp.to(device)
        self.model_kp.eval()

        self.model_lines = get_cls_net_l(self.cfg_lines)
        self.model_lines.load_state_dict(torch.load(weights_lines, map_location=device))
        self.model_lines.to(device)
        self.model_lines.eval()

        # FP16 conversion
        if use_fp16 and "cuda" in device:
            self.model_kp = self.model_kp.half()
            self.model_lines = self.model_lines.half()

        # CUDA streams
        if "cuda" in device:
            self.stream_kp = torch.cuda.Stream()
            self.stream_lines = torch.cuda.Stream()
        else:
            self.stream_kp = None
            self.stream_lines = None

        self.transform = T.Resize((540, 960))

    def detect(
        self,
        frame: np.ndarray,
    ) -> Tuple[Dict, Dict]:
        """
        Detect keypoints and lines in a frame.

        Args:
            frame: BGR image.

        Returns:
            Tuple of (keypoint_dict, lines_dict) with detected features.
        """
        from models.pitch.utils.heatmap import (
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
            complete_keypoints,
            coords_to_dict,
        )

        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_np = frame_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame_np).unsqueeze(0)

        if frame_tensor.size()[-1] != 960:
            frame_tensor = self.transform(frame_tensor)

        frame_tensor = frame_tensor.to(self.device)
        if self.use_fp16:
            frame_tensor = frame_tensor.half()

        _, _, h, w = frame_tensor.size()

        # Inference
        with torch.no_grad():
            if self.stream_kp is not None:
                with torch.cuda.stream(self.stream_kp):
                    heatmaps_kp = self.model_kp(frame_tensor)
                with torch.cuda.stream(self.stream_lines):
                    heatmaps_lines = self.model_lines(frame_tensor)
                self.stream_kp.synchronize()
                self.stream_lines.synchronize()
            else:
                heatmaps_kp = self.model_kp(frame_tensor)
                heatmaps_lines = self.model_lines(frame_tensor)

        # Extract keypoints
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps_kp[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_lines[:, :-1, :, :])

        kp_dict = coords_to_dict(kp_coords, threshold=self.kp_threshold)
        lines_dict = coords_to_dict(line_coords, threshold=self.line_threshold)
        kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)

        return kp_dict, lines_dict


class PitchCalibrator:
    """
    Frame-by-frame camera calibration from pitch keypoints.

    Combines keypoint detection with camera parameter estimation.

    Args:
        detector: PitchLineDetector instance.
        pnl_refine: Use PnL refinement.

    Example:
        >>> detector = PitchLineDetector(...)
        >>> calibrator = PitchCalibrator(detector)
        >>> P = calibrator.process_frame(frame)
        >>> if P is not None:
        ...     frame_viz = project_lines_to_image(frame, P)
    """

    def __init__(
        self,
        detector: PitchLineDetector,
        pnl_refine: bool = True,
    ) -> None:
        self.detector = detector
        self.pnl_refine = pnl_refine
        self.cam = None

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Process frame and return projection matrix.

        Args:
            frame: BGR image.

        Returns:
            3x4 projection matrix, or None if calibration failed.
        """
        from models.pitch.utils.calib import FramebyFrameCalib

        h, w = frame.shape[:2]
        if self.cam is None:
            self.cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)

        kp_dict, lines_dict = self.detector.detect(frame)

        self.cam.update(kp_dict, lines_dict)
        params = self.cam.heuristic_voting(refine_lines=self.pnl_refine)

        if params is not None:
            return projection_from_cam_params(params)
        return None


__all__ = [
    "LINE_COORDINATES_3D",
    "projection_from_cam_params",
    "project_lines_to_image",
    "PitchLineDetector",
    "PitchCalibrator",
]
