"""
Pitch keypoint detection models.

HeatmapPitchDetector uses a DINOv2 ViT-S/14 backbone with a CNN heatmap
decoder to detect 32 pitch landmark keypoints (Roboflow schema) and a
broadcast-view confidence score, used to compute the pitch-to-image
homography via HomographyEstimator.

Example:
    >>> from torchkick.models.pitch import HeatmapPitchDetector
    >>> detector = HeatmapPitchDetector("weights/pitch_heatmap/best.pt")
    >>> keypoints, confidence = detector.detect(frame)
    >>> # keypoints: np.ndarray [32, 2], confidence: np.ndarray [32]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


class DINOv2PitchModel(torch.nn.Module):
    """
    DINOv2 ViT-S/14 backbone + lightweight CNN heatmap decoder for pitch
    landmark detection.

    Backbone outputs 40×40 patch tokens (for 560×560 input) which are decoded
    to 32 per-keypoint heatmaps at 320×320 resolution.  A secondary head on
    the CLS token predicts whether the frame is a broadcast view.

    Args:
        backbone_name: DINOv2 hub model name ("dinov2_vits14" or "dinov2_vitb14").
        num_keypoints: Number of output heatmap channels (default 32).

    Input:  [B, 3, 560, 560]  (ImageNet-normalised RGB)
    Output: (logits [B, 32, 320, 320],  pitch_logit [B, 1])
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        num_keypoints: int = 32,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints

        # --- backbone ---
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=True,
        )
        embed_dim: int = self.backbone.embed_dim  # 384 for ViT-S, 768 for ViT-B

        # --- pitch-presence head (CLS token → scalar logit) ---
        self.pitch_head = torch.nn.Linear(embed_dim, 1)

        # --- heatmap decoder (patch tokens → 32 heatmaps at 8× upsampling) ---
        def _block(in_ch: int, out_ch: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True),
            )

        self.decoder = torch.nn.Sequential(
            # project backbone dim → 256
            torch.nn.Conv2d(embed_dim, 256, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            # stage 1: 40×40 → 80×80
            _block(256, 256),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # stage 2: 80×80 → 160×160
            _block(256, 128),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # stage 3: 160×160 → 320×320
            _block(128, 64),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # 1×1 projection to keypoint channels (no activation — raw logits)
            torch.nn.Conv2d(64, num_keypoints, 1),
        )

        torch.nn.init.constant_(self.pitch_head.bias, 2.0)  # prior: most frames are broadcast

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            logits:       [B, 32, H/8*4, W/8*4]  raw heatmap logits (no sigmoid)
            pitch_logit:  [B, 1]                  raw broadcast-view logit
        """
        B = x.shape[0]
        features = self.backbone.forward_features(x)
        # patch tokens: [B, num_patches, embed_dim]
        patch_tokens = features["x_norm_patchtokens"]
        cls_token = features["x_norm_clstoken"]  # [B, embed_dim]

        # reshape patch tokens to spatial grid
        n_patches = patch_tokens.shape[1]
        grid_size = int(n_patches**0.5)
        spatial = patch_tokens.permute(0, 2, 1).reshape(B, -1, grid_size, grid_size)

        logits = self.decoder(spatial)  # [B, 32, 320, 320]
        pitch_logit = self.pitch_head(cls_token)  # [B, 1]
        return logits, pitch_logit


class HeatmapPitchDetector:
    """
    DINOv2+heatmap pitch landmark detector.

    Drop-in replacement for YOLOPoseKeypointDetector — same detect() interface.

    Detects 32 pitch keypoints (Roboflow schema) and a broadcast-view
    confidence score.  If the frame is not a broadcast view (pitch_conf < 0.5)
    empty arrays are returned immediately so the caller can skip homography.

    Args:
        weights_path:      Path to .pt checkpoint saved by train_pitch_heatmap.
        device:            Torch device string (default "cuda").
        conf_threshold:    Minimum heatmap peak value to accept a keypoint.
        pitch_threshold:   Minimum pitch-presence score to proceed.
        use_fp16:          Use FP16 inference on CUDA (default True).

    Example:
        >>> detector = HeatmapPitchDetector("weights/pitch_heatmap/best.pt")
        >>> keypoints, confidence = detector.detect(frame)
        >>> # keypoints: np.ndarray [32, 2], confidence: np.ndarray [32]
    """

    NUM_KEYPOINTS = 32
    INPUT_SIZE = 560
    HEATMAP_SIZE = 320
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "cuda",
        conf_threshold: float = 0.3,
        pitch_threshold: float = 0.5,
        use_fp16: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.pitch_threshold = pitch_threshold
        self.use_fp16 = use_fp16 and "cuda" in str(device)
        self._model: Optional[DINOv2PitchModel] = None
        self._load_model(str(weights_path))

    @property
    def backbone(self):
        """DINOv2 backbone for reuse by other components (e.g. GameTeamEmbedder)."""
        return self._model.backbone if self._model is not None else None

    def _load_model(self, weights_path: str) -> None:
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        backbone = cfg.get("backbone_variant", "dinov2_vits14")
        nkp = cfg.get("num_keypoints", self.NUM_KEYPOINTS)

        model = DINOv2PitchModel(backbone_name=backbone, num_keypoints=nkp)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        if self.use_fp16:
            model = model.half()
        model = model.to(self.device)
        self._model = model

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """BGR frame → [1, 3, 560, 560] normalised tensor."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rsz = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        img = rsz.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # [1, 3, H, W]
        if self.use_fp16:
            t = t.half()
        return t.to(self.device)

    @staticmethod
    def _subpixel_argmax(hm: np.ndarray) -> Tuple[float, float]:
        """3-point parabolic sub-pixel refinement on argmax neighbourhood."""
        hy, hx = np.unravel_index(np.argmax(hm), hm.shape)
        H, W = hm.shape
        dx = dy = 0.0
        if 1 <= hx < W - 1:
            a, b, c = hm[hy, hx - 1], hm[hy, hx], hm[hy, hx + 1]
            denom = a - 2 * b + c
            if abs(denom) > 1e-6:
                dx = 0.5 * (a - c) / denom
        if 1 <= hy < H - 1:
            a, b, c = hm[hy - 1, hx], hm[hy, hx], hm[hy + 1, hx]
            denom = a - 2 * b + c
            if abs(denom) > 1e-6:
                dy = 0.5 * (a - c) / denom
        return float(hx) + dx, float(hy) + dy

    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect pitch landmark keypoints in a BGR frame.

        Args:
            frame: BGR image array (any resolution).

        Returns:
            keypoints:  np.ndarray [32, 2]  pixel coordinates (x, y) in original frame.
            confidence: np.ndarray [32]     scores in [0, 1]; 0 = not visible / below threshold.
        """
        empty = (
            np.zeros((self.NUM_KEYPOINTS, 2), dtype=np.float32),
            np.zeros(self.NUM_KEYPOINTS, dtype=np.float32),
        )
        if self._model is None:
            return empty

        h_orig, w_orig = frame.shape[:2]
        inp = self._preprocess(frame)

        logits, pitch_logit = self._model(inp)

        # --- pitch-presence gate ---
        pitch_conf = float(torch.sigmoid(pitch_logit[0, 0]).cpu())
        if pitch_conf < self.pitch_threshold:
            return empty

        # --- decode heatmaps ---
        heatmaps = torch.sigmoid(logits[0]).float().cpu().numpy()  # [32, 320, 320]
        hm_w = hm_h = self.HEATMAP_SIZE

        kps = np.zeros((self.NUM_KEYPOINTS, 2), dtype=np.float32)
        conf = np.zeros(self.NUM_KEYPOINTS, dtype=np.float32)

        for k in range(self.NUM_KEYPOINTS):
            hm = heatmaps[k]
            peak = float(hm.max())
            if peak < self.conf_threshold:
                continue
            hx, hy = self._subpixel_argmax(hm)
            # scale from heatmap space to original image space
            kps[k, 0] = (hx + 0.5) / hm_w * w_orig
            kps[k, 1] = (hy + 0.5) / hm_h * h_orig
            conf[k] = peak

        return kps, conf


__all__ = [
    "ViTPoseKeypointDetector",
    "YOLOPoseKeypointDetector",
    "DINOv2PitchModel",
    "HeatmapPitchDetector",
]
