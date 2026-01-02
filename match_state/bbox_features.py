import cv2
import numpy as np
from typing import List
from sklearn.cluster import MiniBatchKMeans
from utils.timing import timed


@timed
def get_jersey_color_feature(image_rgb: np.ndarray, box: List[float]) -> np.ndarray:
    """
    Extracts a compact 6D color feature from shirt and shorts regions.

    Strategy:
    1. Extract upper half (shirt, 15-45%) and lower half (shorts, 50-75%)
    2. Dampen green pixels with soft weighting instead of hard masking
    3. Return mean Lab values: [shirt_L, shirt_a, shirt_b, shorts_L, shorts_a, shorts_b]

    Args:
        image_rgb (np.ndarray): HxWx3 RGB image
        box (List[float]): [x1, y1, x2, y2] bounding box of player

    Returns:
        np.ndarray: 6D color feature vector
    """
    x1, y1, x2, y2 = map(int, box)
    h_img, w_img, _ = image_rgb.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    bbox_h = y2 - y1
    bbox_w = x2 - x1

    if bbox_h < 15 or bbox_w < 5:
        return np.zeros(6, dtype=np.float32)

    # horizontal middle 40% to avoid arms/edges (30% trim each side)
    crop_x1 = x1 + int(bbox_w * 0.30)
    crop_x2 = x2 - int(bbox_w * 0.30)

    def extract_region_mean(ry1, ry2):
        crop = image_rgb[ry1:ry2, crop_x1:crop_x2]
        if crop.size == 0:
            return np.zeros(3, dtype=np.float32)

        # convert to HSV for green detection
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32) / 255.0

        # soft green dampening: weight based on hue distance from green (60)
        green_center = 60
        hue_dist = np.minimum(np.abs(h - green_center), 180 - np.abs(h - green_center))
        weight = np.clip(hue_dist / 30.0, 0, 1) * (1 - s * 0.5)

        # convert to Lab
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float32)
        weight_flat = weight.flatten()

        # Weighted mean
        valid = weight_flat > 0.2
        if valid.sum() < 5:
            return np.zeros(3, dtype=np.float32)

        w = weight_flat[valid]
        pixels = lab[valid]
        mean_lab = np.average(pixels, axis=0, weights=w)
        return mean_lab.astype(np.float32)

    # shirt region: 10-50% height (upper body)
    shirt_y1 = y1 + int(bbox_h * 0.10)
    shirt_y2 = y1 + int(bbox_h * 0.50)
    shirt_feat = extract_region_mean(shirt_y1, shirt_y2)

    # shorts/legs region: 40-90% height (lower body, overlaps a bit)
    shorts_y1 = y1 + int(bbox_h * 0.40)
    shorts_y2 = y1 + int(bbox_h * 0.90)
    shorts_feat = extract_region_mean(shorts_y1, shorts_y2)

    return np.concatenate([shirt_feat, shorts_feat])


@timed
def get_dominant_color_feature(image_rgb: np.ndarray, box: List[float]) -> np.ndarray:
    """
    Extracts a robust 6D color feature using K-Means clustering on non-green pixels.

    Strategy:
    1. Extract player crop (center 50% width).
    2. Filter out green background pixels.
    3. Cluster remaining pixels into 2 groups (Shirt, Shorts).
    4. Sort clusters by vertical position (Upper=Shirt, Lower=Shorts).
    5. Return [shirt_L, shirt_a, shirt_b, shorts_L, shorts_a, shorts_b].

    This is more robust to pose variations (bending, falling) than fixed crops.
    """
    x1, y1, x2, y2 = map(int, box)
    h_img, w_img, _ = image_rgb.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    bbox_h = y2 - y1
    bbox_w = x2 - x1

    if bbox_h < 15 or bbox_w < 5:
        return np.zeros(6, dtype=np.float32)

    # Crop center 50% width to avoid background
    crop_x1 = x1 + int(bbox_w * 0.25)
    crop_x2 = x2 - int(bbox_w * 0.25)

    crop = image_rgb[y1:y2, crop_x1:crop_x2]
    if crop.size == 0:
        return np.zeros(6, dtype=np.float32)

    # 1. Filter Green
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0

    green_center = 60
    hue_dist = np.minimum(np.abs(h - green_center), 180 - np.abs(h - green_center))
    # Harder mask for clustering: keep pixels far from green or low saturation (white/black)
    # Keep if hue_dist > 20 OR saturation < 0.25
    mask = (hue_dist > 20) | (s < 0.25)

    if mask.sum() < 50:
        # Fallback to simple mean if not enough pixels
        return get_jersey_color_feature(image_rgb, box)

    # 2. Prepare data for K-Means
    # We want to cluster based on Color (Lab) AND Position (Y)
    # But primarily Color. Let's cluster on Lab first.
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab).astype(np.float32)

    valid_pixels = lab[mask]
    valid_coords = np.argwhere(mask)  # (y, x) relative to crop

    # Need at least some pixels for K-Means
    if len(valid_pixels) < 10:
        return get_jersey_color_feature(image_rgb, box)

    # Subsample if too many pixels to speed up
    if len(valid_pixels) > 1000:
        indices = np.random.choice(len(valid_pixels), 1000, replace=False)
        valid_pixels = valid_pixels[indices]
        valid_coords = valid_coords[indices]

    # 3. K-Means (K=2)
    try:
        kmeans = MiniBatchKMeans(n_clusters=2, n_init=3, batch_size=min(256, len(valid_pixels)), random_state=42).fit(
            valid_pixels
        )
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
    except Exception:
        return get_jersey_color_feature(image_rgb, box)

    # 4. Determine which cluster is Upper (Shirt) and Lower (Shorts)
    # Calculate mean Y position for each cluster
    y_coords = valid_coords[:, 0]

    mean_y_0 = y_coords[labels == 0].mean() if (labels == 0).any() else 0
    mean_y_1 = y_coords[labels == 1].mean() if (labels == 1).any() else bbox_h

    if mean_y_0 < mean_y_1:
        shirt_feat = centers[0]
        shorts_feat = centers[1]
    else:
        shirt_feat = centers[1]
        shorts_feat = centers[0]

    return np.concatenate([shirt_feat, shorts_feat])
