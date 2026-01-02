import numpy as np
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, ConfigDict
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture


MAX_PLAYER_SPEED_MS = 12.0  # m/s (generous for sprints)
TYPICAL_PLAYER_SPEED_MS = 6.0  # m/s
PENALTY_AREA_X = 52.5 - 16.5  # 36m from center = inside penalty area
HALF_LENGTH = 52.5  # m
HALF_WIDTH = 34.0  # m


class TrackObservation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    frame_idx: int
    box: np.ndarray
    pitch_pos: Optional[Tuple[float, float]] = None
    color_feature: Optional[np.ndarray] = None


class TrackData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    track_id: int
    observations: List[TrackObservation] = []
    smoothed_positions: Optional[np.ndarray] = None
    smoothed_frames: Optional[np.ndarray] = None
    role: Optional[str] = None
    team: Optional[int] = None
    player_id: Optional[int] = None
    frame_team_votes: Optional[List[int]] = None

    def duration_frames(self) -> int:
        return len(self.observations)

    def duration_seconds(self, fps: float) -> float:
        return self.duration_frames() / fps

    def mean_pitch_position(self) -> Optional[Tuple[float, float]]:
        """Average position on pitch."""
        positions = [o.pitch_pos for o in self.observations if o.pitch_pos is not None]
        if not positions:
            return None
        arr = np.array(positions)
        return (arr[:, 0].mean(), arr[:, 1].mean())

    def pitch_position_stats(self) -> Dict:
        """Get position statistics for role inference."""
        positions = [o.pitch_pos for o in self.observations if o.pitch_pos is not None]
        if not positions:
            return {}
        arr = np.array(positions)
        return {
            'mean_x': arr[:, 0].mean(),
            'mean_y': arr[:, 1].mean(),
            'std_x': arr[:, 0].std(),
            'std_y': arr[:, 1].std(),
            'min_x': arr[:, 0].min(),
            'max_x': arr[:, 0].max(),
            'min_y': arr[:, 1].min(),
            'max_y': arr[:, 1].max(),
            'n_samples': len(positions),
        }


class TrajectoryStore:
    """
    Stores all track data and provides methods for smoothing and analysis.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.tracks: Dict[int, TrackData] = {}
        self.total_frames = 0
        self.frame_homographies: Dict[int, np.ndarray] = {}  # Cache H_inv per frame

    def add_observation(
        self,
        track_id: int,
        frame_idx: int,
        box: np.ndarray,
        pitch_pos: Optional[Tuple[float, float]] = None,
        color_feature: Optional[np.ndarray] = None,
    ):
        """Add a single observation for a track."""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackData(track_id=track_id)

        obs = TrackObservation(
            frame_idx=frame_idx,
            box=box,
            pitch_pos=pitch_pos,
            color_feature=color_feature,
        )
        self.tracks[track_id].observations.append(obs)
        self.total_frames = max(self.total_frames, frame_idx + 1)

    def get_track(self, track_id: int) -> Optional[TrackData]:
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> List[TrackData]:
        return list(self.tracks.values())

    def get_long_tracks(self, min_frames: int = 30) -> List[TrackData]:
        """Get tracks with at least min_frames observations."""
        return [t for t in self.tracks.values() if t.duration_frames() >= min_frames]


class FallbackProjector:
    """
    Projects bounding boxes to 2D pitch coordinates when homography fails.

    Uses the relative positions and sizes of bounding boxes to estimate
    2D positions. Works by:
    1. Building a reference frame from frames WITH valid homography
    2. For frames without homography, estimate positions from bbox
       relative to known reference points (anchors).

    Key assumptions:
    - Larger bboxes = closer to camera = lower Y pitch coordinate
    - X position in image correlates with X pitch position (after lens correction)
    - Players maintain relative distances frame-to-frame
    """

    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Calibration from reference frames
        self.ref_scale_x = None  # pixels/meter ratio
        self.ref_scale_y = None  # bbox_height -> y_pitch mapping
        self.ref_offset_x = None
        self.ref_offset_y = None

        # Reference points: list of (bbox, pitch_pos) pairs
        self.reference_points: List[Tuple[np.ndarray, Tuple[float, float]]] = []

    def add_reference(self, box: np.ndarray, pitch_pos: Tuple[float, float]):
        """Add a bbox -> pitch_pos correspondence for calibration."""
        self.reference_points.append((box.copy(), pitch_pos))

        # Re-calibrate after accumulating enough points
        if len(self.reference_points) >= 50:
            self._calibrate()

    def _calibrate(self):
        """Compute calibration parameters from reference points."""
        if len(self.reference_points) < 20:
            return

        # Extract features and targets
        bboxes = []
        pitches = []
        for box, pitch in self.reference_points[-200:]:  # Use last 200
            bboxes.append(box)
            pitches.append(pitch)

        bboxes = np.array(bboxes)
        pitches = np.array(pitches)

        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2  # center x

        # Fit linear model: pitch_x ~ cx, pitch_y ~ cy, h
        # Using simple linear regression

        # X: center_x in image -> pitch_x
        # Normalize cx to [0, 1]
        cx_norm = cx / self.image_width

        # Linear fit for X
        A_x = np.column_stack([cx_norm, np.ones_like(cx_norm)])
        result = np.linalg.lstsq(A_x, pitches[:, 0], rcond=None)
        if len(result[0]) >= 2:
            self.ref_scale_x = result[0][0]
            self.ref_offset_x = result[0][1]
        # Try combined feature: bottom of bbox correlates with depth
        bottom = bboxes[:, 3] / self.image_height  # normalized bottom y

        A_y = np.column_stack([bottom, np.ones_like(bottom)])
        result = np.linalg.lstsq(A_y, pitches[:, 1], rcond=None)
        if len(result[0]) >= 2:
            self.ref_scale_y = result[0][0]
            self.ref_offset_y = result[0][1]

    def project(self, box: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Project a bounding box to pitch coordinates using calibrated model.

        Returns None if not calibrated.
        """
        if self.ref_scale_x is None or self.ref_scale_y is None:
            return None

        cx = (box[0] + box[2]) / 2
        bottom = box[3]

        cx_norm = cx / self.image_width
        bottom_norm = bottom / self.image_height

        pitch_x = self.ref_scale_x * cx_norm + self.ref_offset_x
        pitch_y = self.ref_scale_y * bottom_norm + self.ref_offset_y

        pitch_x = np.clip(pitch_x, -self.pitch_length / 2, self.pitch_length / 2)
        pitch_y = np.clip(pitch_y, -self.pitch_width / 2, self.pitch_width / 2)

        return (float(pitch_x), float(pitch_y))

    def is_calibrated(self) -> bool:
        return self.ref_scale_x is not None and self.ref_scale_y is not None


class SimpleIoUTracker:
    """
    Lightweight IoU-based tracker for offline analysis.

    Only does detection-to-track association. No team assignment, no GMM,
    no Kalman filtering. Just simple IoU matching + track persistence.

    Team/identity assignment happens later in the offline pipeline.
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Dict] = {}  # track_id -> {box, age}
        self.next_id = 1

    def _iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, detections: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        """
        Update tracks with new detections.

        Args:
            detections: List of [x1, y1, x2, y2] or [x1, y1, x2, y2, score] boxes

        Returns:
            List of (track_id, box) tuples for active tracks
        """
        from scipy.optimize import linear_sum_assignment

        # Convert detections to numpy arrays, strip score if present
        det_boxes = []
        for det in detections:
            box = np.array(det[:4])
            det_boxes.append(box)

        # Increment age for all tracks
        for tid in self.tracks:
            self.tracks[tid]['age'] += 1

        if not det_boxes:
            # No detections - just age out tracks
            self._prune_old_tracks()
            return [(tid, t['box']) for tid, t in self.tracks.items() if t['age'] <= 1]

        if not self.tracks:
            # No existing tracks - create new ones for all detections
            results = []
            for box in det_boxes:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'box': box, 'age': 0}
                results.append((tid, box))
            return results

        # Build cost matrix (negative IoU for Hungarian)
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(det_boxes)))

        for i, tid in enumerate(track_ids):
            for j, det_box in enumerate(det_boxes):
                cost_matrix[i, j] = -self._iou(self.tracks[tid]['box'], det_box)

        # Hungarian matching
        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_inds, col_inds):
            if -cost_matrix[r, c] >= self.iou_threshold:
                tid = track_ids[r]
                self.tracks[tid]['box'] = det_boxes[c]
                self.tracks[tid]['age'] = 0
                matched_tracks.add(tid)
                matched_dets.add(c)

        # Create new tracks for unmatched detections
        for j, det_box in enumerate(det_boxes):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'box': det_box, 'age': 0}

        # Prune old tracks
        self._prune_old_tracks()

        # Return active tracks (age <= 1 means recently matched)
        return [(tid, t['box']) for tid, t in self.tracks.items() if t['age'] <= 1]

    def _prune_old_tracks(self):
        """Remove tracks that haven't been seen for max_age frames."""
        to_remove = [tid for tid, t in self.tracks.items() if t['age'] > self.max_age]
        for tid in to_remove:
            del self.tracks[tid]


class TrajectorySmootherVelocityConstrained:
    """
    Smooths 2D trajectories with physical velocity constraints.

    Uses:
    1. Outlier rejection (impossible jumps)
    2. Light Gaussian smoothing
    3. Velocity clamping
    """

    def __init__(
        self,
        fps: float = 30.0,
        max_speed_ms: float = MAX_PLAYER_SPEED_MS,
        smooth_sigma: float = 2.0,  # frames - reduced for less aggressive smoothing
        outlier_threshold_ms: float = 15.0,  # Reject jumps faster than this
    ):
        self.fps = fps
        self.max_speed_ms = max_speed_ms
        self.max_speed_per_frame = max_speed_ms / fps  # m/frame
        self.smooth_sigma = smooth_sigma
        self.outlier_threshold = outlier_threshold_ms / fps  # m/frame

    def _remove_outliers(self, positions: np.ndarray, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove positions that represent impossible jumps."""
        if len(positions) < 3:
            return positions, frames

        # Calculate velocities
        valid_mask = np.ones(len(positions), dtype=bool)

        for i in range(1, len(positions) - 1):
            dt_prev = max(1, frames[i] - frames[i - 1])
            dt_next = max(1, frames[i + 1] - frames[i])

            # Velocity from prev to current
            dx1 = positions[i, 0] - positions[i - 1, 0]
            dy1 = positions[i, 1] - positions[i - 1, 1]
            v1 = np.sqrt(dx1**2 + dy1**2) / dt_prev

            # Velocity from current to next
            dx2 = positions[i + 1, 0] - positions[i, 0]
            dy2 = positions[i + 1, 1] - positions[i, 1]
            v2 = np.sqrt(dx2**2 + dy2**2) / dt_next

            # If both velocities are extreme, this point is likely an outlier
            if v1 > self.outlier_threshold and v2 > self.outlier_threshold:
                valid_mask[i] = False

        return positions[valid_mask], frames[valid_mask]

    def _interpolate_gaps(
        self, positions: np.ndarray, frames: np.ndarray, total_frames: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fill gaps in trajectory with linear interpolation."""
        if len(positions) < 2:
            return positions, frames

        # Create dense frame array
        min_frame, max_frame = frames[0], frames[-1]
        dense_frames = np.arange(min_frame, max_frame + 1)

        # Interpolate x and y separately
        x_interp = np.interp(dense_frames, frames, positions[:, 0])
        y_interp = np.interp(dense_frames, frames, positions[:, 1])

        return np.stack([x_interp, y_interp], axis=1), dense_frames

    def smooth_track(self, track: TrackData, interpolate: bool = True) -> bool:
        """
        Smooth a single track's trajectory.
        Returns True if successful, False if insufficient data.
        """
        # Extract pitch positions
        positions = []
        frames = []

        for obs in track.observations:
            if obs.pitch_pos is not None:
                positions.append(obs.pitch_pos)
                frames.append(obs.frame_idx)

        if len(positions) < 5:
            return False

        positions = np.array(positions)
        frames = np.array(frames)

        # Step 0: Remove outliers (impossible jumps)
        positions, frames = self._remove_outliers(positions, frames)

        if len(positions) < 5:
            return False

        # Step 0.5: Interpolate gaps for smoother results
        if interpolate:
            positions, frames = self._interpolate_gaps(positions, frames, track.observations[-1].frame_idx)

        # Step 1: Gaussian smooth for noise reduction
        x_smooth = gaussian_filter1d(positions[:, 0], sigma=self.smooth_sigma)
        y_smooth = gaussian_filter1d(positions[:, 1], sigma=self.smooth_sigma)

        # Step 2: Velocity clamping (forward pass)
        for i in range(1, len(x_smooth)):
            dt = max(1, frames[i] - frames[i - 1])
            max_dist = self.max_speed_per_frame * dt

            dx = x_smooth[i] - x_smooth[i - 1]
            dy = y_smooth[i] - y_smooth[i - 1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > max_dist:
                # Clamp to max velocity
                scale = max_dist / dist
                x_smooth[i] = x_smooth[i - 1] + dx * scale
                y_smooth[i] = y_smooth[i - 1] + dy * scale

        # Step 3: Backward pass for bidirectional smoothing
        for i in range(len(x_smooth) - 2, -1, -1):
            dt = max(1, frames[i + 1] - frames[i])
            max_dist = self.max_speed_per_frame * dt

            dx = x_smooth[i] - x_smooth[i + 1]
            dy = y_smooth[i] - y_smooth[i + 1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > max_dist:
                scale = max_dist / dist
                x_smooth[i] = x_smooth[i + 1] + dx * scale
                y_smooth[i] = y_smooth[i + 1] + dy * scale

        # Step 4: Final Gaussian pass for extra smoothness
        x_smooth = gaussian_filter1d(x_smooth, sigma=self.smooth_sigma / 2)
        y_smooth = gaussian_filter1d(y_smooth, sigma=self.smooth_sigma / 2)

        # Store results
        track.smoothed_positions = np.stack([x_smooth, y_smooth], axis=1)
        track.smoothed_frames = frames

        return True

    def smooth_all(self, store: TrajectoryStore, min_frames: int = 10) -> int:
        """Smooth all tracks in the store. Returns count of successfully smoothed."""
        count = 0
        for track in store.tracks.values():
            if track.duration_frames() >= min_frames:
                if self.smooth_track(track):
                    count += 1
        return count


class IdentityAssigner:
    """
    Assigns player identities using color clustering + spatial priors.

    Strategy:
    1. Per-frame color voting - classify each frame independently
    2. Majority vote across track lifetime to determine final identity
    3. Use spatial position to identify goalies (penalty area)
    4. Use spatial position to identify linesmen (sidelines)
    5. Use spatial position to identify referee (center, mobile)
    6. Enforce 22 players + 1 referee constraint for 2D pitch view
    """

    def __init__(self, fps: float = 30.0, debug: bool = True):
        self.fps = fps
        self.debug = debug
        self.gmm = None  # GMM for per-frame voting
        self.team_a_cluster = None
        self.team_b_cluster = None
        self.ref_cluster = None

    def assign_roles(self, store: TrajectoryStore) -> Dict[int, Dict]:
        """
        Assign roles and teams to all tracks.

        Returns:
            Dict mapping track_id -> {'role': str, 'team': int, 'player_id': int}
        """
        tracks = store.get_long_tracks(min_frames=30)

        if self.debug:
            print(f"[IdentityAssigner] Processing {len(tracks)} long tracks")

        track_stats = {}
        for track in tracks:
            stats = track.pitch_position_stats()
            if stats:
                track_stats[track.track_id] = stats

        if not track_stats:
            return {}

        goalie_candidates = self._find_goalie_candidates(track_stats)
        linesman_candidates = self._find_linesman_candidates(track_stats, exclude=goalie_candidates)
        remaining_ids = [
            tid for tid in track_stats.keys() if tid not in goalie_candidates and tid not in linesman_candidates
        ]

        team_assignments, referee_id = self._cluster_by_color(store, remaining_ids)
        goalie_teams = self._assign_goalie_teams(goalie_candidates, track_stats, team_assignments)

        results = {}

        for tid in track_stats.keys():
            track = store.get_track(tid)

            if tid in goalie_candidates:
                results[tid] = {
                    'role': 'goalie',
                    'team': goalie_teams.get(tid, -1),
                    'player_id': 1,
                }
                track.role = 'goalie'
                track.team = goalie_teams.get(tid, -1)

            elif tid in linesman_candidates:
                results[tid] = {
                    'role': 'linesman',
                    'team': -1,
                    'player_id': None,
                }
                track.role = 'linesman'
                track.team = -1

            elif tid == referee_id:
                results[tid] = {
                    'role': 'referee',
                    'team': -1,
                    'player_id': None,
                }
                track.role = 'referee'
                track.team = -1

            else:
                team = team_assignments.get(tid, 0)
                results[tid] = {
                    'role': 'player',
                    'team': team,
                    'player_id': None,  # need jersey OCR for this
                }
                track.role = 'player'
                track.team = team

        if self.debug:
            self._print_summary(results, track_stats)

        return results

    def _find_goalie_candidates(self, track_stats: Dict) -> set:
        """Find tracks that are predominantly in penalty areas."""
        candidates = set()

        for tid, stats in track_stats.items():
            mean_x = stats['mean_x']
            std_x = stats['std_x']

            # fair assumption criteria
            # 1. Average position in penalty area (|x| > 36m)
            # 2. Low x-variance (stays in area)
            # 3. Enough samples

            in_left_penalty = mean_x < -PENALTY_AREA_X and std_x < 10
            in_right_penalty = mean_x > PENALTY_AREA_X and std_x < 10

            if (in_left_penalty or in_right_penalty) and stats['n_samples'] > 50:
                candidates.add(tid)
                if self.debug:
                    side = "LEFT" if in_left_penalty else "RIGHT"
                    print(f"[DEBUG] Track {tid} -> GOALIE candidate ({side}): mean_x={mean_x:.1f}m, std_x={std_x:.1f}m")

        return candidates

    def _find_linesman_candidates(self, track_stats: Dict, exclude: set) -> set:
        """Find tracks that are predominantly on sidelines."""
        candidates = set()

        for tid, stats in track_stats.items():
            if tid in exclude:
                continue

            mean_y = stats['mean_y']
            std_y = stats['std_y']

            # fair assumption criteria
            # 1. Average position near sideline (|y| > 32m)
            # 2. Low y-variance (stays on line)
            # 3. High x-variance (runs along line)

            near_sideline = abs(mean_y) > 32
            low_y_var = std_y < 5

            if near_sideline and low_y_var and stats['n_samples'] > 50:
                candidates.add(tid)
                if self.debug:
                    side = "TOP" if mean_y > 0 else "BOTTOM"
                    print(
                        f"[DEBUG] Track {tid} -> LINESMAN candidate ({side}): mean_y={mean_y:.1f}m, std_y={std_y:.1f}m"
                    )

        return candidates

    def _cluster_by_color(self, store: TrajectoryStore, track_ids: List[int]) -> Tuple[Dict[int, int], Optional[int]]:
        """
        Cluster tracks by color features using per-frame voting with majority.
        Returns (team_assignments, referee_id).

        Approach:
        1. Fit GMM with 2 clusters (just the two teams) on ALL color observations
        2. Per-frame voting: classify each frame to team 0 or 1
        3. Identify referee as the track whose color is most OUTLIER from both teams
           (high Mahalanobis distance from both cluster centers)
        4. Linesmen are near sidelines, center referee is central
        """
        if not track_ids:
            return {}, None

        if not self._fit_color_gmm_2_clusters(store, track_ids):
            return {tid: 0 for tid in track_ids}, None

        team_assignments = {}
        track_outlier_scores = {}  # track_id -> average outlier score

        for tid in track_ids:
            track = store.get_track(tid)
            if not track:
                continue

            majority_team, confidence, outlier_score = self._vote_per_frame_with_outlier(track)
            team_assignments[tid] = majority_team
            track_outlier_scores[tid] = outlier_score

            if self.debug and outlier_score > 1.5:
                print(f"[DEBUG] Track {tid}: team={majority_team}, conf={confidence:.2f}, outlier={outlier_score:.2f}")

        # Step 3: Find referee = track with highest outlier score that's NOT near sidelines
        referee_id = None
        best_outlier = 0.0

        for tid, outlier_score in track_outlier_scores.items():
            track = store.get_track(tid)
            if not track:
                continue

            stats = track.pitch_position_stats()
            if not stats:
                continue

            # Exclude sideline tracks (linesmen)
            mean_y = abs(stats.get('mean_y', 0))
            if mean_y > 30:  # Near sideline = linesman, not center referee
                if self.debug and outlier_score > 1.0:
                    print(f"[DEBUG] Track {tid} -> LINESMAN (|mean_y|={mean_y:.1f}m, outlier={outlier_score:.2f})")
                continue

            # Also require some mobility (referee runs around)
            mobility = stats.get('std_x', 0) + stats.get('std_y', 0)
            if mobility < 2.0:  # Too static = probably not referee
                continue

            # Referee should have high outlier score AND be on the pitch
            if outlier_score > best_outlier:
                best_outlier = outlier_score
                referee_id = tid

        if referee_id is not None:
            team_assignments[referee_id] = -1  # Mark as referee
            if self.debug:
                print(f"[DEBUG] Selected referee: Track {referee_id} (outlier={best_outlier:.2f})")

        return team_assignments, referee_id

    def _fit_color_gmm_2_clusters(self, store: TrajectoryStore, track_ids: List[int]) -> bool:
        """
        Fit GMM with just 2 clusters (the two teams).
        Referee will be detected as color outlier, not as a separate cluster.
        """
        if not track_ids:
            return False

        all_features = []
        for tid in track_ids:
            track = store.get_track(tid)
            if not track:
                continue
            for obs in track.observations:
                if obs.color_feature is not None and np.linalg.norm(obs.color_feature) > 0:
                    all_features.append(obs.color_feature)

        if len(all_features) < 100:
            if self.debug:
                print(f"[DEBUG] Not enough color samples ({len(all_features)}) for GMM")
            return False

        X = np.array(all_features, dtype=np.float64)

        try:
            # Only 2 clusters for the two teams
            self.gmm = GaussianMixture(
                n_components=2,
                covariance_type='diag',
                random_state=42,
                n_init=10,
                reg_covar=1e-3,
            )
            self.gmm.fit(X)

            # Store cluster means and covariances for outlier detection
            self.team_means = self.gmm.means_
            self.team_covs = self.gmm.covariances_

            labels = self.gmm.predict(X)
            cluster_counts = np.bincount(labels, minlength=2)

            # Larger cluster = team A, smaller = team B (arbitrary)
            sorted_clusters = np.argsort(cluster_counts)[::-1]
            self.team_a_cluster = sorted_clusters[0]
            self.team_b_cluster = sorted_clusters[1]
            self.ref_cluster = None  # No dedicated referee cluster

            if self.debug:
                print(f"[DEBUG] GMM fitted on {len(all_features)} color samples (2 clusters)")
                print(f"[DEBUG] Team A cluster={self.team_a_cluster} ({cluster_counts[self.team_a_cluster]} samples)")
                print(f"[DEBUG] Team B cluster={self.team_b_cluster} ({cluster_counts[self.team_b_cluster]} samples)")

            return True

        except Exception as e:
            if self.debug:
                print(f"[DEBUG] GMM fitting failed: {e}")
            return False

    def _vote_per_frame_with_outlier(self, track: TrackData) -> Tuple[int, float, float]:
        """
        Per-frame voting with outlier score calculation.

        Returns:
            (majority_team, confidence, outlier_score)

        outlier_score = average Mahalanobis distance from BOTH team centroids.
        High score = color doesn't fit either team well = likely referee.
        """
        if self.gmm is None:
            return 0, 0.0, 0.0

        votes = []
        outlier_scores = []

        for obs in track.observations:
            if obs.color_feature is None or np.linalg.norm(obs.color_feature) == 0:
                continue

            feat = obs.color_feature.reshape(1, -1)
            probs = self.gmm.predict_proba(feat)[0]
            label = np.argmax(probs)
            conf = probs[label]

            # Compute outlier score: how far from both clusters?
            # Use negative log probability as distance metric
            max_prob = max(probs)
            outlier = -np.log(max_prob + 1e-10)  # High when max_prob is low
            outlier_scores.append(outlier)

            # Map cluster to team
            if label == self.team_a_cluster:
                team = 0
            else:
                team = 1

            votes.append((team, conf))

        if not votes:
            return 0, 0.0, 0.0

        # Count votes
        team_scores = {0: 0.0, 1: 0.0}
        for team, conf in votes:
            team_scores[team] += conf

        majority_team = max(team_scores, key=team_scores.get)
        total_votes = sum(team_scores.values())
        confidence = team_scores[majority_team] / total_votes if total_votes > 0 else 0.0

        # Average outlier score
        avg_outlier = np.mean(outlier_scores) if outlier_scores else 0.0

        # Store per-frame votes
        track.frame_team_votes = [v[0] for v in votes]

        return majority_team, confidence, avg_outlier

    def _assign_goalie_teams(self, goalie_ids: set, track_stats: Dict, team_assignments: Dict) -> Dict[int, int]:
        """Assign goalies to teams based on which side they defend."""
        goalie_teams = {}

        # Figure out which team defends which side
        # Count average x position of each team's players
        team_0_x = []
        team_1_x = []

        for tid, team in team_assignments.items():
            if tid in track_stats:
                if team == 0:
                    team_0_x.append(track_stats[tid]['mean_x'])
                elif team == 1:
                    team_1_x.append(track_stats[tid]['mean_x'])

        # Determine which team is on which side (in first half)
        # This is a heuristic - in reality teams switch at half time
        team_0_avg_x = np.mean(team_0_x) if team_0_x else 0
        team_1_avg_x = np.mean(team_1_x) if team_1_x else 0

        for gid in goalie_ids:
            if gid in track_stats:
                goalie_x = track_stats[gid]['mean_x']

                # Assign to the team whose outfield players are on the OPPOSITE side
                # (goalies defend their own goal)
                if goalie_x < 0:
                    # Left goalie -> team whose players are on right
                    goalie_teams[gid] = 0 if team_0_avg_x > team_1_avg_x else 1
                else:
                    # Right goalie -> team whose players are on left
                    goalie_teams[gid] = 0 if team_0_avg_x < team_1_avg_x else 1

                if self.debug:
                    print(f"[DEBUG] Goalie {gid} assigned to Team {goalie_teams[gid]}")

        return goalie_teams

    def _print_summary(self, results: Dict, track_stats: Dict):
        """Print summary of assignments."""
        role_counts = defaultdict(int)
        team_counts = defaultdict(int)

        for tid, info in results.items():
            role_counts[info['role']] += 1
            if info['team'] in [0, 1]:
                team_counts[info['team']] += 1

        print("\n[IdentityAssigner] Summary:")
        print(f"  Roles: {dict(role_counts)}")
        print(f"  Teams: Team 0: {team_counts[0]}, Team 1: {team_counts[1]}")


class PlayerSlot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    slot_id: int
    team: int
    position: Tuple[float, float]
    last_observed_frame: int
    assigned_track_ids: List[int] = []
    color_feature: Optional[np.ndarray] = None
    is_goalie: bool = False


class PitchSlotManager:
    """
    Manages fixed 11v11 + 1 optional referee slot(s) on the 2D homographic pitch.
    Responsible for assigning tracks to slots based on identity assignments
    and merging fragmented tracks that likely belong to the same player.
    """

    def __init__(self, fps: float = 30.0, debug: bool = False):
        self.fps = fps
        self.debug = debug
        self.slots: Dict[str, PlayerSlot] = {}

        for team in [0, 1]:
            for i in range(11):
                slot_key = f"T{team}_{i}"
                self.slots[slot_key] = PlayerSlot(
                    slot_id=i,
                    team=team,
                    position=(0.0, 0.0),
                    last_observed_frame=-1,
                )

        self.slots["REF"] = PlayerSlot(
            slot_id=0,
            team=-1,
            position=(0.0, 0.0),
            last_observed_frame=-1,
        )
        self.track_to_slot: Dict[int, str] = {}
        self.frame_slot_positions: Dict[int, Dict[str, Tuple[float, float]]] = defaultdict(dict)

    def initialize_from_assignments(
        self,
        store: TrajectoryStore,
        assignments: Dict[int, Dict],
    ):
        """
        Initialize slots from identity assignments.

        Merges fragmented tracks that likely represent the same player based on:
        1. Same team assignment
        2. Temporal non-overlap (tracks don't exist at same time)
        3. Spatial proximity (last pos of track A close to first pos of track B)
        """
        if self.debug:
            print("\n[PitchSlotManager] Initializing slots from assignments...")

        # Group tracks by team and role
        team_tracks: Dict[int, List[Tuple[int, TrackData, Dict]]] = {0: [], 1: []}
        referee_tracks = []

        for tid, info in assignments.items():
            track = store.get_track(tid)
            if not track:
                continue

            role = info.get('role', 'unknown')
            team = info.get('team', -1)

            if role == 'linesman':
                continue  # Skip linesmen

            if role == 'referee':
                referee_tracks.append((tid, track, info))
            elif team in [0, 1]:
                team_tracks[team].append((tid, track, info))

        # Merge fragmented tracks for each team
        for team in [0, 1]:
            tracks = team_tracks[team]
            merged_groups = self._merge_fragmented_tracks(tracks, store)

            # Sort merged groups by total duration
            merged_groups.sort(
                key=lambda g: sum(store.get_track(tid).duration_frames() for tid in g if store.get_track(tid)),
                reverse=True,
            )

            # Assign top 11 groups to slots
            for i, group in enumerate(merged_groups[:11]):
                slot_key = f"T{team}_{i}"

                # Find the main track (longest in group)
                main_track = None
                main_duration = 0
                for tid in group:
                    track = store.get_track(tid)
                    if track and track.duration_frames() > main_duration:
                        main_duration = track.duration_frames()
                        main_track = track

                if not main_track:
                    continue

                # Assign all tracks in group to this slot
                for tid in group:
                    self.track_to_slot[tid] = slot_key
                    self.slots[slot_key].assigned_track_ids.append(tid)

                # Get position from main track
                pos = main_track.mean_pitch_position()
                if pos:
                    self.slots[slot_key].position = pos

                # Check if any track in group is a goalie
                for tid in group:
                    info = assignments.get(tid, {})
                    if info.get('role') == 'goalie':
                        self.slots[slot_key].is_goalie = True
                        break

                # Compute average color from all tracks in group
                all_colors = []
                for tid in group:
                    track = store.get_track(tid)
                    if track:
                        colors = [o.color_feature for o in track.observations if o.color_feature is not None]
                        all_colors.extend(colors)
                if all_colors:
                    self.slots[slot_key].color_feature = np.mean(all_colors, axis=0)

                if self.debug:
                    total_dur = sum(store.get_track(tid).duration_frames() for tid in group if store.get_track(tid))
                    print(f"  Slot {slot_key}: {len(group)} tracks merged, total {total_dur} frames")

        # Assign referee
        if referee_tracks:
            tid, track, info = referee_tracks[0]
            self.track_to_slot[tid] = "REF"
            self.slots["REF"].assigned_track_ids.append(tid)

            if self.debug:
                print(f"  Slot REF: Track {tid}")

        if self.debug:
            print(f"[PitchSlotManager] Initialized {len(self.track_to_slot)} track->slot mappings")

    def _merge_fragmented_tracks(
        self,
        tracks: List[Tuple[int, TrackData, Dict]],
        store: TrajectoryStore,
        max_gap_frames: int = 30,  # Max frames between tracks to merge
        max_distance_px: float = 150.0,  # Max pixels between bbox centers
        max_height_ratio: float = 1.5,  # Max ratio between bbox heights
    ) -> List[List[int]]:
        """
        Merge fragmented tracks that likely represent the same player.

        Uses 3D (IMAGE SPACE) matching - more robust to homography errors:
        1. Temporal gap (tracks don't overlap)
        2. Bbox center proximity (in pixels)
        3. Bbox height similarity (similar size = similar depth)
        """
        if not tracks:
            return []

        # Get track info using BBOX coordinates (image space)
        track_info = []
        for tid, track, info in tracks:
            if not track.observations:
                continue

            # Get first and last observations (any observation, not just with pitch_pos)
            first_obs = track.observations[0]
            last_obs = track.observations[-1]

            # Compute bbox centers and heights
            first_box = first_obs.box
            last_box = last_obs.box

            first_center = ((first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2)
            last_center = ((last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2)
            first_height = first_box[3] - first_box[1]
            last_height = last_box[3] - last_box[1]

            track_info.append(
                {
                    'tid': tid,
                    'first_frame': first_obs.frame_idx,
                    'last_frame': last_obs.frame_idx,
                    'first_center': first_center,
                    'last_center': last_center,
                    'first_height': first_height,
                    'last_height': last_height,
                    'duration': track.duration_frames(),
                }
            )

        # Sort by first frame
        track_info.sort(key=lambda x: x['first_frame'])

        # Union-find structure for merging
        parent = {t['tid']: t['tid'] for t in track_info}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # For each track, try to connect to a previous track
        for i, curr in enumerate(track_info):
            best_match = None
            best_score = float('inf')

            for j in range(i):
                prev = track_info[j]

                # Check temporal gap
                gap = curr['first_frame'] - prev['last_frame']
                if gap < 0 or gap > max_gap_frames:
                    continue

                # Check spatial distance in image space (pixels)
                dx = curr['first_center'][0] - prev['last_center'][0]
                dy = curr['first_center'][1] - prev['last_center'][1]
                dist = np.sqrt(dx**2 + dy**2)

                if dist > max_distance_px:
                    continue

                # Check height similarity (size consistency)
                height_ratio = max(curr['first_height'], prev['last_height']) / max(
                    min(curr['first_height'], prev['last_height']), 1
                )
                if height_ratio > max_height_ratio:
                    continue

                # Score: prefer small gap and small distance, penalize height mismatch
                score = gap + dist * 0.5 + (height_ratio - 1) * 50
                if score < best_score:
                    best_score = score
                    best_match = prev['tid']

            if best_match is not None:
                union(curr['tid'], best_match)

        # Build groups from union-find
        groups: Dict[int, List[int]] = defaultdict(list)
        for t in track_info:
            root = find(t['tid'])
            groups[root].append(t['tid'])

        return list(groups.values())

    def process_frame(
        self,
        frame_idx: int,
        observations: List[Dict],  # [{track_id, pitch_pos, team, role, color_feature}]
        max_match_distance: float = 15.0,  # meters
    ):
        """
        Process a single frame's observations and update slot positions.

        For tracks not in our mapping, try to match to existing slots.
        Keep slots at last position when not observed.
        """
        # Update slots from known track mappings
        observed_slots = set()

        for obs in observations:
            tid = obs['track_id']
            pos = obs.get('pitch_pos')
            team = obs.get('team', -1)

            if pos is None:
                continue

            # Check if we already know this track
            if tid in self.track_to_slot:
                slot_key = self.track_to_slot[tid]
                self.slots[slot_key].position = pos
                self.slots[slot_key].last_observed_frame = frame_idx
                observed_slots.add(slot_key)
            else:
                # Try to match to an unobserved slot of the same team
                best_slot = None
                best_dist = max_match_distance

                for slot_key, slot in self.slots.items():
                    if slot.team != team:
                        continue
                    if slot_key in observed_slots:
                        continue
                    if slot.last_observed_frame < 0:
                        continue  # Never observed, skip

                    # Distance to last known position
                    dx = pos[0] - slot.position[0]
                    dy = pos[1] - slot.position[1]
                    dist = np.sqrt(dx**2 + dy**2)

                    if dist < best_dist:
                        best_dist = dist
                        best_slot = slot_key

                if best_slot:
                    # Assign this track to the slot
                    self.track_to_slot[tid] = best_slot
                    self.slots[best_slot].assigned_track_ids.append(tid)
                    self.slots[best_slot].position = pos
                    self.slots[best_slot].last_observed_frame = frame_idx
                    observed_slots.add(best_slot)

        # Store positions for this frame
        for slot_key, slot in self.slots.items():
            if slot.last_observed_frame >= 0:  # Only slots that have been observed
                self.frame_slot_positions[frame_idx][slot_key] = slot.position

    def get_frame_positions(self, frame_idx: int) -> List[Dict]:
        """
        Get all slot positions for a frame.
        Returns list of {slot_key, position, team, is_goalie}.
        """
        result = []

        for slot_key, slot in self.slots.items():
            if slot.last_observed_frame >= 0:
                # Use stored position for this frame, or last known
                if frame_idx in self.frame_slot_positions and slot_key in self.frame_slot_positions[frame_idx]:
                    pos = self.frame_slot_positions[frame_idx][slot_key]
                else:
                    pos = slot.position

                result.append(
                    {
                        'slot_key': slot_key,
                        'position': pos,
                        'team': slot.team,
                        'is_goalie': slot.is_goalie,
                        'slot_id': slot.slot_id,
                    }
                )

        return result

    def build_all_frame_positions(self, store: TrajectoryStore, total_frames: int):
        """
        Build frame positions for all frames using SMOOTHED track data.
        Interpolates between observations for smooth movement.
        """
        if self.debug:
            print(f"\n[PitchSlotManager] Building positions for {total_frames} frames...")

        # Build lookup from smoothed data when available, otherwise raw observations
        # Format: tid -> {frame_idx -> (x, y)}
        track_frame_positions: Dict[int, Dict[int, Tuple[float, float]]] = {}

        for tid in self.track_to_slot:
            track = store.get_track(tid)
            if not track:
                continue

            track_frame_positions[tid] = {}

            # Prefer smoothed positions if available (from Pass 2)
            if track.smoothed_positions is not None and track.smoothed_frames is not None:
                for i, frame_idx in enumerate(track.smoothed_frames):
                    pos = (track.smoothed_positions[i, 0], track.smoothed_positions[i, 1])
                    track_frame_positions[tid][int(frame_idx)] = pos
            else:
                # Fallback to raw observations
                for obs in track.observations:
                    if obs.pitch_pos is not None:
                        track_frame_positions[tid][obs.frame_idx] = obs.pitch_pos

        # For each slot, build interpolated positions across all frames
        for tid, slot_key in self.track_to_slot.items():
            if tid not in track_frame_positions:
                continue

            frame_pos = track_frame_positions[tid]
            if not frame_pos:
                continue

            # Get sorted frames
            frames = sorted(frame_pos.keys())
            if not frames:
                continue

            # Initialize slot position
            first_pos = frame_pos[frames[0]]
            self.slots[slot_key].position = first_pos
            self.slots[slot_key].last_observed_frame = frames[0]

            # Interpolate for all frames in range
            min_frame, max_frame = frames[0], frames[-1]

            for frame_idx in range(min_frame, max_frame + 1):
                if frame_idx in frame_pos:
                    # Have exact position
                    pos = frame_pos[frame_idx]
                else:
                    # Interpolate between nearest known frames
                    prev_frame = max(f for f in frames if f < frame_idx)
                    next_frame = min(f for f in frames if f > frame_idx)

                    if prev_frame is not None and next_frame is not None:
                        t = (frame_idx - prev_frame) / (next_frame - prev_frame)
                        prev_pos = frame_pos[prev_frame]
                        next_pos = frame_pos[next_frame]
                        pos = (
                            prev_pos[0] + t * (next_pos[0] - prev_pos[0]),
                            prev_pos[1] + t * (next_pos[1] - prev_pos[1]),
                        )
                    else:
                        pos = self.slots[slot_key].position

                self.slots[slot_key].position = pos
                self.slots[slot_key].last_observed_frame = frame_idx
                self.frame_slot_positions[frame_idx][slot_key] = pos

            # Extend last known position for frames after track ends
            for frame_idx in range(max_frame + 1, total_frames):
                self.frame_slot_positions[frame_idx][slot_key] = self.slots[slot_key].position

        if self.debug:
            obs_count = sum(len(positions) for positions in self.frame_slot_positions.values())
            print(f"[PitchSlotManager] Built {obs_count} slot-frame observations")
