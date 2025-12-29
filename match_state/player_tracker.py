import cv2
import numpy as np
from typing import List
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment


def get_jersey_color_feature(image_rgb: np.ndarray, box: List[float]) -> np.ndarray:
    """
    Extracts a robust color feature from the player's chest region.

    Strategy:
    1. Focus on chest region (top 25%-65% height, middle 40% width) - avoids head, legs, grass edges
    2. Aggressively mask out green pixels (the pitch)
    3. Use percentile-based Lab features to capture the color distribution
       - This handles stripes, patterns, and is robust to outliers
       - Uses L (lightness) to distinguish white vs black
       - Uses a, b for chromaticity (red-green, blue-yellow)

    Args:
        image_rgb (np.ndarray): HxWx3 RGB image
        box (List[float]): [x1, y1, x2, y2] bounding box of player

    Returns:
        np.ndarray: 9D color feature vector
    """
    x1, y1, x2, y2 = map(int, box)
    h_img, w_img, _ = image_rgb.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    bbox_h = y2 - y1
    bbox_w = x2 - x1

    if bbox_h < 10 or bbox_w < 5:
        return np.zeros(9)

    chest_y1 = y1 + int(bbox_h * 0.10)
    chest_y2 = y1 + int(bbox_h * 0.90)
    chest_x1 = x1 + int(bbox_w * 0.30)
    chest_x2 = x2 - int(bbox_w * 0.30)

    chest_crop = image_rgb[chest_y1:chest_y2, chest_x1:chest_x2]
    if chest_crop.size == 0:
        return np.zeros(9)

    hsv = cv2.cvtColor(chest_crop, cv2.COLOR_RGB2HSV)

    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)

    lab_crop = cv2.cvtColor(chest_crop, cv2.COLOR_RGB2Lab)
    valid_pixels = lab_crop[non_green_mask > 0]

    if len(valid_pixels) < 10:
        center_crop = image_rgb[y1 + bbox_h // 3 : y1 + 2 * bbox_h // 3, x1 + bbox_w // 3 : x2 - bbox_w // 3]
        if center_crop.size == 0:
            return np.zeros(9)
        lab_center = cv2.cvtColor(center_crop, cv2.COLOR_RGB2Lab)
        valid_pixels = lab_center.reshape(-1, 3)

    if len(valid_pixels) == 0:
        return np.zeros(9)

    L = valid_pixels[:, 0]
    a = valid_pixels[:, 1]
    b = valid_pixels[:, 2]

    feature = np.array(
        [
            np.percentile(L, 10),
            np.percentile(L, 50),
            np.percentile(L, 90),
            np.percentile(a, 10),
            np.percentile(a, 50),
            np.percentile(a, 90),
            np.percentile(b, 10),
            np.percentile(b, 50),
            np.percentile(b, 90),
        ],
        dtype=np.float32,
    )

    return feature


class KalmanTrack:
    """
    Single object track with Kalman Filter and jersey color feature.
    Args:
        box (List[float]): initial bounding box [x1, y1, x2, y2]
        track_id (int): unique track ID
        color_feature (np.ndarray): initial jersey color feature
    """

    def __init__(self, box: List[float], track_id: int, color_feature: np.ndarray):
        self.id = track_id
        self.color_feature = color_feature
        self.team_id = None
        self.team_belief = 0.0  # range(-1.0, 1.0) for team assignment confidence
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        # kalman filter with 7 state variables and 4 measurements
        # state = [x, y, s, r, vx, vy, vs] (x,y center, scale area, ratio, velocities)
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], np.float32
        )

        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self._update_kf_state(box)

    def _update_kf_state(self, box: List[float]) -> None:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        s = w * h
        r = w / float(h)
        self.kf.statePost = np.array([[cx], [cy], [s], [r], [0], [0], [0]], np.float32)

    def predict(self) -> List[float]:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, box: List[float], color_feature: np.ndarray) -> None:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        s = w * h
        r = w / float(h)

        measurement = np.array([[cx], [cy], [s], [r]], np.float32)
        self.kf.correct(measurement)

        self.hits += 1
        self.time_since_update = 0

        # exp MA for color adaption to light changes
        self.color_feature = 0.90 * self.color_feature + 0.10 * color_feature

    def get_state(self) -> List[float]:
        """Returns the current bounding box estimate [x1, y1, x2, y2]"""
        x = self.kf.statePost
        cx, cy, s, r = x[0].item(), x[1].item(), x[2].item(), x[3].item()

        # NOTE: safety check for negative area or ratio
        if s <= 0 or r <= 0:
            # fallback to last known good state or just return 0-box
            # ideally we should have the last valid box
            return [0, 0, 0, 0]

        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return [x1, y1, x2, y2]


class PlayerTracker:
    """
    Multi-object tracker with jersey color-based team clustering.

    Args:
        max_age (int): frames to keep "alive" without detections
        iou_threshold (float): minimum IoU for matching detections to tracks
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: List[KalmanTrack] = []
        self.frame_id = 0
        self.next_id = 1

        self.gmm = None
        self.team_centers = None
        self.color_buffer: List[np.ndarray] = []
        self.fitting_complete = False

    def _iou(self, box_a: List[float], box_b: List[float]) -> float:
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def update(self, detections: List[List[float]], image_rgb: np.ndarray) -> None:
        """
        detections: List of [x1, y1, x2, y2, score]
        """
        self.frame_id += 1

        for t in self.tracks:
            t.predict()

        det_features = []
        for det in detections:
            color = get_jersey_color_feature(image_rgb, det[:4])
            det_features.append({'box': det[:4], 'color': color, 'score': det[4]})

        matches = []
        unmatched_tracks = []
        unmatched_dets = []

        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for t, track in enumerate(self.tracks):
                pred_box = track.get_state()
                for d, det in enumerate(detections):
                    iou_matrix[t, d] = self._iou(pred_box, det[:4])

            # NOTE: maximize IoU -> minimize -IoU for Hungarian algo.
            cost_matrix = -iou_matrix
            row_inds, col_inds = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_inds, col_inds):
                if iou_matrix[r, c] < self.iou_threshold:
                    unmatched_tracks.append(r)
                    unmatched_dets.append(c)
                else:
                    matches.append((r, c))

            for t in range(len(self.tracks)):
                if t not in row_inds:
                    unmatched_tracks.append(t)
            for d in range(len(detections)):
                if d not in col_inds:
                    unmatched_dets.append(d)
        else:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))

        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(det_features[d_idx]['box'], det_features[d_idx]['color'])

        for d_idx in unmatched_dets:
            new_track = KalmanTrack(det_features[d_idx]['box'], self.next_id, det_features[d_idx]['color'])
            self.tracks.append(new_track)
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # team clustering logic with contrastive inference
        # NOTE: color features gets collected from active tracks
        valid_colors = [
            t.color_feature for t in self.tracks if t.time_since_update == 0 and np.linalg.norm(t.color_feature) > 0
        ]

        if not self.fitting_complete:
            self.color_buffer.extend(valid_colors)

            if len(self.color_buffer) > 300:
                print("Fitting Team Colors with GMM...")
                data = np.array(self.color_buffer)

                # NOTE: we might want to refactor to 3 clusters for referee detection
                self.gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=3)
                self.gmm.fit(data)

                self.team_centers = self.gmm.means_
                self.fitting_complete = True
                self.color_buffer = []
        else:
            active_tracks = [t for t in self.tracks if np.linalg.norm(t.color_feature) > 0]

            if len(active_tracks) > 0:
                features = np.array([t.color_feature for t in active_tracks])

                predictions = self.gmm.predict(features)
                probs = self.gmm.predict_proba(features)

                for i, t in enumerate(active_tracks):
                    pred = predictions[i]
                    confidence = probs[i].max()

                    if confidence < 0.6:
                        continue
                    dist = np.linalg.norm(t.color_feature - self.team_centers[pred])
                    if dist > 80:
                        continue

                    vote = -1.0 if pred == 0 else 1.0

                    if t.team_belief == 0.0:
                        t.team_belief = vote
                    else:
                        alpha = 0.1 * confidence
                        t.team_belief = (1 - alpha) * t.team_belief + alpha * vote

                    t.team_id = 1 if t.team_belief > 0 else 0

        results = []
        for t in self.tracks:
            if t.hits >= 3 and t.time_since_update < 5:
                res_box = t.get_state()
                results.append({'box': res_box, 'id': t.id, 'team': t.team_id})

        return results
