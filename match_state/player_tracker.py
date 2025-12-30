import cv2
import numpy as np
from typing import List
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment


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


class TeamClassifier:
    """
    Classifies players into teams using pairwise color distance clustering.

    Approach:
    - First detect linesmen by position (near touchlines using homography)
    - Extract 6D color features (shirt + shorts mean Lab)
    - Initialize clusters once, then refine with EMA (never re-seed)
    - Detect referee as the biggest color outlier (excluding linesmen)
    - Use 5s sliding window voting for temporal stability
    - All officials (referee + linesmen) are classified as REFEREE
    """

    TEAM_0 = 0
    TEAM_1 = 1
    REFEREE = 2  # Includes both main referee and linesmen

    # Pitch dimensions (FIFA standard)
    HALF_WIDTH = 34.0  # Half of pitch width (goal line to center)

    # Debug flag
    DEBUG = True

    def __init__(self, fps: int = 30, window_seconds: float = 5.0):
        """
        Args:
            fps: Video frame rate for temporal window calculation
            window_seconds: Seconds of history for sliding window voting
        """
        self.fps = fps
        self.window_size = int(fps * window_seconds)

        # Track state: features, vote history, position history
        self.track_state = {}  # track_id -> dict

        # Team color centers (updated dynamically)
        self.team_centers = [None, None]  # [team0_center, team1_center]
        self.centers_initialized = False

        self.frame_idx = 0

    def _color_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Euclidean distance between two 6D color features."""
        if feat1 is None or feat2 is None:
            return float('inf')
        if np.linalg.norm(feat1) == 0 or np.linalg.norm(feat2) == 0:
            return float('inf')
        return float(np.linalg.norm(feat1 - feat2))

    def _initialize_centers(self, track_ids: List[int], linesman_ids: set) -> bool:
        """
        Initialize cluster centers ONCE using the two most distant tracks.
        Excludes linesmen from initialization.
        Returns True if successfully initialized.
        """
        # Filter out linesmen
        candidate_ids = [tid for tid in track_ids if tid not in linesman_ids]

        if len(candidate_ids) < 4:
            return False

        # Get features for all tracks
        features = {}
        for tid in candidate_ids:
            if tid in self.track_state:
                feat = self.track_state[tid]['color_feature']
                if np.linalg.norm(feat) > 0:
                    features[tid] = feat

        if len(features) < 4:
            return False

        tids = list(features.keys())
        feats = np.array([features[t] for t in tids])

        # Find two most distant tracks as initial seeds
        max_dist = 0
        seed1, seed2 = 0, 1
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                d = np.linalg.norm(feats[i] - feats[j])
                if d > max_dist:
                    max_dist = d
                    seed1, seed2 = i, j

        # Require minimum separation
        if max_dist < 15:
            return False

        # Initialize centers
        center0 = feats[seed1].copy()
        center1 = feats[seed2].copy()

        # Simple k-means: 5 iterations for better initial fit
        for _ in range(5):
            cluster0, cluster1 = [], []
            for feat in feats:
                d0 = np.linalg.norm(feat - center0)
                d1 = np.linalg.norm(feat - center1)
                if d0 < d1:
                    cluster0.append(feat)
                else:
                    cluster1.append(feat)

            if cluster0:
                center0 = np.mean(cluster0, axis=0)
            if cluster1:
                center1 = np.mean(cluster1, axis=0)

        self.team_centers = [center0, center1]
        self.centers_initialized = True
        print(f"Team centers initialized. Separation: {max_dist:.1f}")
        return True

    def _update_centers_incremental(self, track_ids: List[int], exclude_ids: set) -> None:
        """
        Incrementally update centers using EMA from confidently classified tracks.
        Excludes referee, linesmen, and tracks with mixed voting history.
        """
        if not self.centers_initialized:
            return

        team0_feats = []
        team1_feats = []

        for tid in track_ids:
            if tid in exclude_ids:
                continue
            if tid not in self.track_state:
                continue

            state = self.track_state[tid]
            feat = state['color_feature']
            if np.linalg.norm(feat) == 0:
                continue

            # Only use tracks with strong voting history (>70% for one team)
            votes = [v for _, v in state['vote_history'] if v in [self.TEAM_0, self.TEAM_1]]
            if len(votes) < 15:
                continue

            counts = [votes.count(0), votes.count(1)]
            total = sum(counts)
            if total == 0:
                continue

            majority_ratio = max(counts) / total
            if majority_ratio < 0.7:
                continue

            if counts[0] > counts[1]:
                team0_feats.append(feat)
            else:
                team1_feats.append(feat)

        # Update centers with slow EMA
        alpha = 0.03  # Very slow update
        if team0_feats:
            new_center0 = np.mean(team0_feats, axis=0)
            self.team_centers[0] = (1 - alpha) * self.team_centers[0] + alpha * new_center0
        if team1_feats:
            new_center1 = np.mean(team1_feats, axis=0)
            self.team_centers[1] = (1 - alpha) * self.team_centers[1] + alpha * new_center1

    def _detect_linesmen(self, active_ids: List[int], pitch_positions: dict = None) -> set:
        """
        Detect linesmen based on position history.
        Linesmen are consistently near/outside the touchlines (|y| >= HALF_WIDTH - 2).
        There is exactly ONE linesman per touchline (near side y<0, far side y>0).
        Returns set of track_ids identified as linesmen (will be classified as REFEREE).
        """
        # Touchline is at 34m, linesmen typically stand 0-2m outside
        touchline_threshold = self.HALF_WIDTH - 2  # 32m from center
        strict_touchline = self.HALF_WIDTH - 0.5  # 33.5m - definitely outside pitch

        # Collect candidates for each side
        near_candidates = []  # (track_id, score, avg_y, near_ratio, outside_ratio) for y < 0
        far_candidates = []  # (track_id, score, avg_y, near_ratio, outside_ratio) for y > 0

        for tid in active_ids:
            if tid not in self.track_state:
                continue

            state = self.track_state[tid]
            pos_history = state.get('position_history', [])

            # For early detection: use current position if not enough history
            current_y = None
            if pitch_positions and tid in pitch_positions:
                _, current_y = pitch_positions[tid]

            # If we have very little history but current position is clearly on touchline
            if len(pos_history) < 10:
                if current_y is not None and abs(current_y) >= strict_touchline:
                    # Immediate linesman candidate based on current position
                    candidate = (tid, 1.0, abs(current_y), 1.0, 1.0)  # High score for being clearly outside
                    if current_y < 0:
                        near_candidates.append(candidate)
                    else:
                        far_candidates.append(candidate)
                continue

            # Calculate average y (signed) to determine which side
            avg_y_signed = np.mean([y for _, y in pos_history])
            avg_y_abs = np.mean([abs(y) for _, y in pos_history])

            # Count how many positions are near touchlines
            near_touchline_count = sum(1 for _, y in pos_history if abs(y) >= touchline_threshold)
            ratio = near_touchline_count / len(pos_history)

            # Count how many are strictly outside the pitch
            outside_pitch_count = sum(1 for _, y in pos_history if abs(y) >= strict_touchline)
            outside_ratio = outside_pitch_count / len(pos_history)

            # Basic linesman criteria: position AND ratio check
            position_ok = avg_y_abs >= (self.HALF_WIDTH - 2.0)  # >= 32.0m
            ratio_ok = ratio > 0.75 or outside_ratio > 0.25

            if position_ok and ratio_ok:
                # Score combines outside_ratio and near_ratio for ranking
                score = outside_ratio * 2 + ratio
                candidate = (tid, score, avg_y_abs, ratio, outside_ratio)
                if avg_y_signed < 0:
                    near_candidates.append(candidate)
                else:
                    far_candidates.append(candidate)

        # Pick best candidate per side (prefer higher score)
        linesman_ids = set()

        for side_name, candidates in [("NEAR", near_candidates), ("FAR", far_candidates)]:
            if not candidates:
                continue

            # Sort by score (desc)
            candidates.sort(key=lambda c: c[1], reverse=True)
            best = candidates[0]
            tid, score, avg_y, ratio, outside_ratio = best
            linesman_ids.add(tid)

            if self.DEBUG and self.frame_idx % 30 == 0:
                print(
                    f"[DEBUG] Track {tid} -> LINESMAN ({side_name}): avg|y|={avg_y:.1f}m, near={ratio:.2f}, outside={outside_ratio:.2f}"
                )

        return linesman_ids

        return linesman_ids

    def _detect_referee(self, active_ids: List[int], exclude_ids: set) -> int:
        """
        Find the track that is most distant from both team centers.
        Excludes linesmen. Returns track_id of the main referee, or -1 if none detected.
        """
        if not self.centers_initialized:
            return -1

        max_min_dist = 0
        referee_id = -1
        threshold = 18  # Minimum distance to be considered outlier

        for tid in active_ids:
            if tid in exclude_ids:
                continue
            if tid not in self.track_state:
                continue
            feat = self.track_state[tid]['color_feature']
            if np.linalg.norm(feat) == 0:
                continue

            d0 = self._color_distance(feat, self.team_centers[0])
            d1 = self._color_distance(feat, self.team_centers[1])
            min_dist = min(d0, d1)

            if min_dist > max_min_dist and min_dist > threshold:
                max_min_dist = min_dist
                referee_id = tid

        if self.DEBUG and referee_id >= 0 and self.frame_idx % 30 == 0:
            print(f"[DEBUG] Track {referee_id} -> REFEREE (pitch): min_dist={max_min_dist:.1f}")

        return referee_id

    def _detect_outside_pitch(self, active_ids: List[int], exclude_ids: set, pitch_positions: dict) -> set:
        """
        Detect tracks that are currently or consistently outside the pitch boundaries.
        These are coaches, cameramen, ball boys, etc. - not players.
        Returns set of track_ids to be classified as non-players (REFEREE).
        """
        outside_ids = set()

        # Pitch boundaries
        max_x = 52.5 + 3  # HALF_LENGTH + 3m margin
        max_y = self.HALF_WIDTH + 1  # 35m - outside touchline

        # Strict boundary - definitely not a player
        strict_max_y = self.HALF_WIDTH + 2  # 36m - way outside

        for tid in active_ids:
            if tid in exclude_ids:  # Skip already classified linesmen
                continue
            if tid not in self.track_state:
                continue

            # Check 1: Current position - IMMEDIATE classification if way outside
            if pitch_positions and tid in pitch_positions:
                x, y = pitch_positions[tid]
                if abs(y) > strict_max_y or abs(x) > max_x + 2:
                    outside_ids.add(tid)
                    if self.DEBUG and self.frame_idx % 30 == 0:
                        print(f"[DEBUG] Track {tid} -> OUTSIDE PITCH (current): x={x:.1f}m, y={y:.1f}m")
                    continue

            state = self.track_state[tid]
            pos_history = state.get('position_history', [])

            if len(pos_history) < 10:  # Reduced from 15 for faster detection
                continue

            # Check 2: Historical position - if >40% outside, classify as non-player
            outside_count = sum(1 for x, y in pos_history if abs(x) > max_x or abs(y) > max_y)
            outside_ratio = outside_count / len(pos_history)

            if outside_ratio > 0.40:
                outside_ids.add(tid)
                if self.DEBUG and self.frame_idx % 30 == 0:
                    avg_x = np.mean([abs(x) for x, _ in pos_history])
                    avg_y = np.mean([abs(y) for _, y in pos_history])
                    print(
                        f"[DEBUG] Track {tid} -> OUTSIDE PITCH (history): avg|x|={avg_x:.1f}m, avg|y|={avg_y:.1f}m, outside={outside_ratio:.2f}"
                    )

        return outside_ids

    def update(
        self,
        tracks: List[dict],
        image_rgb: np.ndarray,
        pitch_positions: dict = None,  # track_id -> (x, y) for linesman detection
    ) -> List[dict]:
        """
        Update team classification for the given tracks.

        Args:
            tracks: List of dicts with 'id', 'box' keys.
            image_rgb: Current frame.
            pitch_positions: Dict of track_id -> (x, y) pitch coords in meters.

        Returns:
            List of tracks with 'team' key (0, 1, 2=referee, 3=linesman).
        """
        self.frame_idx += 1
        active_ids = []

        # Phase 1: Extract features and update track state
        for track in tracks:
            track_id = track['id']
            box = track['box']
            active_ids.append(track_id)

            color_feat = get_jersey_color_feature(image_rgb, box)

            if track_id not in self.track_state:
                self.track_state[track_id] = {
                    'color_feature': color_feat,
                    'vote_history': [],  # [(frame_idx, team_vote), ...]
                    'position_history': [],  # [(x, y), ...]
                    'last_seen': self.frame_idx,
                }
            else:
                # EMA for color feature
                old_feat = self.track_state[track_id]['color_feature']
                if np.linalg.norm(color_feat) > 0:
                    if np.linalg.norm(old_feat) > 0:
                        self.track_state[track_id]['color_feature'] = 0.85 * old_feat + 0.15 * color_feat
                    else:
                        self.track_state[track_id]['color_feature'] = color_feat
                self.track_state[track_id]['last_seen'] = self.frame_idx

            # Update position history if we have pitch coordinates
            if pitch_positions and track_id in pitch_positions:
                x, y = pitch_positions[track_id]
                self.track_state[track_id]['position_history'].append((x, y))
                # Keep last 5 seconds of position history
                max_pos_history = self.window_size
                if len(self.track_state[track_id]['position_history']) > max_pos_history:
                    self.track_state[track_id]['position_history'] = self.track_state[track_id]['position_history'][
                        -max_pos_history:
                    ]

        # Phase 2: Detect linesmen based on position (near touchlines)
        linesman_ids = self._detect_linesmen(active_ids, pitch_positions)

        # Phase 3: Detect people outside pitch (coaches, cameramen, etc.)
        outside_pitch_ids = self._detect_outside_pitch(active_ids, linesman_ids, pitch_positions)

        # Phase 4: Initialize centers ONCE (excluding linesmen and outside-pitch people)
        exclude_from_init = linesman_ids | outside_pitch_ids
        if not self.centers_initialized:
            self._initialize_centers(active_ids, exclude_from_init)

        # Phase 5: Detect main referee by color (excluding linesmen and outside-pitch)
        referee_id = self._detect_referee(active_ids, exclude_from_init)

        # All officials/non-players (linesmen + main referee + outside pitch)
        all_officials = linesman_ids | outside_pitch_ids
        if referee_id >= 0:
            all_officials.add(referee_id)

        # Phase 5: Assign votes based on distance to centers
        if self.centers_initialized:
            for tid in active_ids:
                if tid not in self.track_state:
                    continue

                state = self.track_state[tid]
                feat = state['color_feature']

                if np.linalg.norm(feat) == 0:
                    continue

                # Determine vote - linesmen and referee all become REFEREE
                if tid in all_officials:
                    vote = self.REFEREE
                else:
                    d0 = self._color_distance(feat, self.team_centers[0])
                    d1 = self._color_distance(feat, self.team_centers[1])
                    vote = self.TEAM_0 if d0 < d1 else self.TEAM_1

                state['vote_history'].append((self.frame_idx, vote))

                # Trim history to window size
                cutoff = self.frame_idx - self.window_size
                state['vote_history'] = [(f, v) for f, v in state['vote_history'] if f > cutoff]

        # Phase 6: Update centers incrementally (excluding all officials)
        if self.frame_idx % 15 == 0:
            self._update_centers_incremental(active_ids, all_officials)

        # Phase 7: Assign final team based on majority vote in window
        for track in tracks:
            tid = track['id']
            if tid not in self.track_state:
                track['team'] = self.TEAM_0
                continue

            state = self.track_state[tid]
            votes = [v for _, v in state['vote_history']]

            if not votes:
                track['team'] = self.TEAM_0
            else:
                # Count votes: [team0, team1, referee]
                counts = [0, 0, 0]
                for v in votes:
                    if v < len(counts):
                        counts[v] += 1
                track['team'] = int(np.argmax(counts))

                if self.DEBUG and self.frame_idx % 90 == 0 and track['team'] == self.REFEREE:
                    print(
                        f"[DEBUG] Track {tid} final -> REFEREE (votes: T0={counts[0]}, T1={counts[1]}, REF={counts[2]})"
                    )

        # Cleanup stale tracks
        stale_ids = [tid for tid, state in self.track_state.items() if self.frame_idx - state['last_seen'] > 300]
        for tid in stale_ids:
            del self.track_state[tid]

        return tracks


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
                results.append(
                    {
                        'box': res_box,
                        'id': t.id,
                        'team': t.team_id,
                        'foot_position': ((res_box[0] + res_box[2]) / 2, res_box[3]),
                    }
                )

        return results
