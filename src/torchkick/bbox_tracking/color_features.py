"""
Color feature extraction for team classification.

This module provides functions for extracting color features from
player bounding boxes for team assignment.

Example:
    >>> from torchkick.tracking import get_jersey_color_feature
    >>> 
    >>> feature = get_jersey_color_feature(frame_rgb, player_box)
    >>> # feature is a 6D vector: [shirt_L, shirt_a, shirt_b, shorts_L, shorts_a, shorts_b]
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def get_jersey_color_feature(
    image_rgb: np.ndarray,
    box: List[float],
) -> np.ndarray:
    """
    Extract compact 6D color feature from shirt and shorts regions.

    Strategy:
    1. Extract upper half (shirt, 15-45%) and lower half (shorts, 50-75%)
    2. Dampen green pixels with soft weighting
    3. Return mean Lab values for each region

    Args:
        image_rgb: HxWx3 RGB image.
        box: [x1, y1, x2, y2] bounding box of player.

    Returns:
        6D feature: [shirt_L, shirt_a, shirt_b, shorts_L, shorts_a, shorts_b].

    Example:
        >>> feature = get_jersey_color_feature(frame, [100, 200, 150, 350])
        >>> print(feature.shape)  # (6,)
    """
    x1, y1, x2, y2 = map(int, box)
    h_img, w_img = image_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    bbox_h = y2 - y1
    bbox_w = x2 - x1

    if bbox_h < 15 or bbox_w < 5:
        return np.zeros(6, dtype=np.float32)

    # Crop horizontal middle 40% to avoid arms/edges
    crop_x1 = x1 + int(bbox_w * 0.30)
    crop_x2 = x2 - int(bbox_w * 0.30)

    def extract_region_mean(ry1: int, ry2: int) -> np.ndarray:
        crop = image_rgb[ry1:ry2, crop_x1:crop_x2]
        if crop.size == 0:
            return np.zeros(3, dtype=np.float32)

        # Convert to HSV for green detection
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32) / 255.0

        # Soft green dampening based on hue distance from green (60)
        green_center = 60
        hue_dist = np.minimum(
            np.abs(h - green_center),
            180 - np.abs(h - green_center),
        )
        weight = np.clip(hue_dist / 30.0, 0, 1) * (1 - s * 0.5)

        # Convert to Lab
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float32)
        weight_flat = weight.flatten()

        valid = weight_flat > 0.2
        if valid.sum() < 5:
            return np.zeros(3, dtype=np.float32)

        w = weight_flat[valid]
        pixels = lab[valid]
        mean_lab = np.average(pixels, axis=0, weights=w)
        return mean_lab.astype(np.float32)

    # Shirt region: 10-50% height
    shirt_y1 = y1 + int(bbox_h * 0.10)
    shirt_y2 = y1 + int(bbox_h * 0.50)
    shirt_feat = extract_region_mean(shirt_y1, shirt_y2)

    # Shorts region: 40-90% height
    shorts_y1 = y1 + int(bbox_h * 0.40)
    shorts_y2 = y1 + int(bbox_h * 0.90)
    shorts_feat = extract_region_mean(shorts_y1, shorts_y2)

    return np.concatenate([shirt_feat, shorts_feat])


def get_dominant_color_feature(
    image_rgb: np.ndarray,
    box: List[float],
) -> np.ndarray:
    """
    Extract robust 6D color feature using K-Means clustering.

    More robust to pose variations (bending, falling) than fixed crops.

    Strategy:
    1. Extract player crop (center 50% width)
    2. Filter out green background pixels
    3. Cluster remaining into 2 groups (shirt, shorts)
    4. Sort by vertical position
    5. Return Lab means for each cluster

    Args:
        image_rgb: HxWx3 RGB image.
        box: [x1, y1, x2, y2] bounding box.

    Returns:
        6D feature: [shirt_L, shirt_a, shirt_b, shorts_L, shorts_a, shorts_b].
    """
    from sklearn.cluster import MiniBatchKMeans

    x1, y1, x2, y2 = map(int, box)
    h_img, w_img = image_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    bbox_h = y2 - y1
    bbox_w = x2 - x1

    if bbox_h < 20 or bbox_w < 10:
        return np.zeros(6, dtype=np.float32)

    # Center 50% width
    crop_x1 = x1 + int(bbox_w * 0.25)
    crop_x2 = x2 - int(bbox_w * 0.25)

    crop = image_rgb[y1:y2, crop_x1:crop_x2]
    if crop.size == 0:
        return np.zeros(6, dtype=np.float32)

    # Convert to HSV for green filtering
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    # Non-green mask
    green_low, green_high = 35, 85
    saturation_thresh = 50
    non_green_mask = ~((h >= green_low) & (h <= green_high) & (s > saturation_thresh))

    # Get non-green pixels with positions
    ys, xs = np.where(non_green_mask)
    if len(ys) < 20:
        return np.zeros(6, dtype=np.float32)

    # Get colors in Lab space
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
    lab_pixels = lab[non_green_mask].astype(np.float32)

    # Cluster into 2 groups
    try:
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=3)
        labels = kmeans.fit_predict(lab_pixels)

        # Get mean y position for each cluster
        cluster_y = [np.mean(ys[labels == i]) for i in range(2)]

        # Sort by y (top = shirt, bottom = shorts)
        sort_idx = np.argsort(cluster_y)

        centers = kmeans.cluster_centers_[sort_idx]
        return np.concatenate(centers).astype(np.float32)

    except Exception:
        return np.zeros(6, dtype=np.float32)


__all__ = [
    "get_jersey_color_feature",
    "get_dominant_color_feature",
]
