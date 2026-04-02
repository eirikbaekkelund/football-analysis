# Copyright (c) Meta Platforms, Inc. and affiliates.
# MLX port of sam3/model/sam3_tracker_utils.py
# Replaces all torch operations with mlx.core equivalents.

from typing import Optional, Tuple

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Mask / IoU helpers
# ---------------------------------------------------------------------------


def get_1d_sine_pe(pos_inds: mx.array, dim: int, temperature: float = 10000.0) -> mx.array:
    """1-D sinusoidal positional encoding for a sequence of indices."""
    assert dim % 2 == 0, "dim must be even"
    omega = mx.arange(dim // 2, dtype=mx.float32) / (dim // 2)
    omega = 1.0 / (temperature**omega)  # (dim/2,)
    pos_inds = pos_inds.astype(mx.float32)
    out = pos_inds[:, None] * omega[None, :]  # (N, dim/2)
    emb = mx.concatenate([mx.sin(out), mx.cos(out)], axis=-1)  # (N, dim)
    return emb


def get_activation_fn(activation: str):
    if activation == "relu":
        return mx.maximum
    if activation == "gelu":
        import mlx.nn as nn

        return nn.gelu
    raise ValueError(f"Unknown activation: {activation}")


def select_closest_cond_frames(
    frame_idx: int,
    cond_frame_outputs: dict,
    max_cond_frame_num: int,
) -> Tuple[dict, dict]:
    """
    Among *cond_frame_outputs* (a {frame_idx: output} dict) pick at most
    *max_cond_frame_num* frames closest (by distance) to *frame_idx*.
    Returns (selected, unselected).
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        return cond_frame_outputs, {}

    indices = sorted(cond_frame_outputs.keys())
    # Sort by absolute distance to frame_idx, break ties by recency
    indices = sorted(indices, key=lambda t: (abs(t - frame_idx), -t))
    selected_indices = set(indices[:max_cond_frame_num])

    selected = {t: cond_frame_outputs[t] for t in selected_indices}
    unselected = {t: cond_frame_outputs[t] for t in cond_frame_outputs if t not in selected_indices}
    return selected, unselected


def get_next_point(
    sorted_pts_values: mx.array,
    point_labels: mx.array,
    last_pt_idx: int,
) -> Tuple[Optional[mx.array], Optional[mx.array], int]:
    """
    Iterate over sorted point prompts returning the next point/label pair.
    Returns (point, label, next_idx).  Returns (None, None, last_pt_idx)
    when exhausted.
    """
    if last_pt_idx >= sorted_pts_values.shape[0]:
        return None, None, last_pt_idx
    pt = sorted_pts_values[last_pt_idx : last_pt_idx + 1]  # (1, 2)
    lb = point_labels[last_pt_idx : last_pt_idx + 1]  # (1,)
    return pt, lb, last_pt_idx + 1


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def _sigmoid(x: mx.array) -> mx.array:
    return mx.sigmoid(x)


def fill_holes_in_mask_scores(mask: mx.array, max_area: int) -> mx.array:
    """
    Fill small holes (connected components of 0s) in a binary mask if they
    are smaller than *max_area*.  Operates on a numpy copy then returns mx.
    Falls back to a no-op when scipy is unavailable.
    """
    if max_area <= 0:
        return mask

    try:
        from scipy import ndimage  # optional dependency
    except ImportError:
        return mask

    mask_np = np.array(mask).astype(bool)
    # label background (0) regions
    labeled, num = ndimage.label(~mask_np)
    for i in range(1, num + 1):
        region = labeled == i
        if region.sum() < max_area:
            mask_np[region] = True
    return mx.array(mask_np.astype(np.float32))


def apply_non_overlapping_constraints(
    pred_masks: mx.array,  # (num_obj, H, W)  float logits
) -> mx.array:
    """
    Resolve overlapping predictions by assigning each pixel to the object
    with the highest logit.  All other objects get a very negative logit
    at that pixel.
    """
    # winner takes all per pixel
    winner = mx.argmax(pred_masks, axis=0, keepdims=True)  # (1, H, W)
    obj_idx = mx.arange(pred_masks.shape[0])[:, None, None]  # (N,1,1)
    mask = (winner == obj_idx).astype(mx.float32)  # (N, H, W)
    LARGE_NEG = -1024.0
    out = pred_masks * mask + LARGE_NEG * (1.0 - mask)
    return out


# ---------------------------------------------------------------------------
# Box / point conversion helpers
# ---------------------------------------------------------------------------


def box_xyxy_to_xywh(boxes: mx.array) -> mx.array:
    """Convert (x1,y1,x2,y2) to (cx,cy,w,h)."""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return mx.stack([cx, cy, w, h], axis=-1)


def concat_points(
    old_point_inputs: Optional[dict],
    new_coords: mx.array,
    new_labels: mx.array,
) -> dict:
    """Append new point/label arrays to an existing prompt dict."""
    if old_point_inputs is None:
        return {"point_coords": new_coords, "point_labels": new_labels}
    coords = mx.concatenate([old_point_inputs["point_coords"], new_coords], axis=1)
    labels = mx.concatenate([old_point_inputs["point_labels"], new_labels], axis=1)
    return {"point_coords": coords, "point_labels": labels}


# ---------------------------------------------------------------------------
# EDT / Morphology fallbacks (CPU)
# ---------------------------------------------------------------------------


def edt_cpu(mask: mx.array) -> mx.array:
    """Compute Euclidean distance transform for a binary mask on CPU.

    Returns per-pixel distances (float32). Tries scipy.ndimage first,
    falls back to OpenCV if available.
    """
    mask_np = np.array(mask).astype(np.uint8)
    # Ensure binary (0/1)
    mask_np = (mask_np > 0).astype(np.uint8)

    # Try scipy
    try:
        from scipy import ndimage

        out = ndimage.distance_transform_edt(mask_np)
        return mx.array(out.astype(np.float32))
    except Exception:
        pass

    # Fallback to OpenCV
    try:
        import cv2

        # OpenCV expects non-zero pixels as object (uint8 image)
        img = (mask_np * 255).astype('uint8')
        out = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=0)
        return mx.array(out.astype(np.float32))
    except Exception:
        raise ImportError("EDT requires scipy or opencv. Install scipy (`pip install scipy`) or opencv-python.")


def binary_erosion(mask: mx.array, iterations: int = 1) -> mx.array:
    """Binary erosion (CPU) using scipy if available, else no-op."""
    try:
        from scipy import ndimage

        mask_np = np.array(mask).astype(bool)
        out = ndimage.binary_erosion(mask_np, iterations=iterations)
        return mx.array(out.astype(np.float32))
    except Exception:
        return mask


def binary_dilation(mask: mx.array, iterations: int = 1) -> mx.array:
    """Binary dilation (CPU) using scipy if available, else no-op."""
    try:
        from scipy import ndimage

        mask_np = np.array(mask).astype(bool)
        out = ndimage.binary_dilation(mask_np, iterations=iterations)
        return mx.array(out.astype(np.float32))
    except Exception:
        return mask
