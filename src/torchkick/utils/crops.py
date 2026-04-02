"""Shared crop utility used by inference.py and annotation modules."""

from __future__ import annotations

import numpy as np


def crop_box(frame_rgb: np.ndarray, box) -> np.ndarray:
    """
    Crop a bounding box region from a frame.

    Args:
        frame_rgb: RGB (or BGR) image array.
        box: Bounding box [x1, y1, x2, y2].

    Returns:
        Cropped region as a uint8 numpy array.
        Returns a black 128 x 64 placeholder if the box is degenerate.
    """
    h, w = frame_rgb.shape[:2]
    x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
    x2, y2 = int(min(w, box[2])), int(min(h, box[3]))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((128, 64, 3), dtype=np.uint8)
    return frame_rgb[y1:y2, x1:x2].copy()


__all__ = ["crop_box"]
