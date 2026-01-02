"""
Offline Tracker with 2D Trajectory Smoothing and Identity Assignment

This module implements a multi-pass offline analysis pipeline:
1. Pass 1: Detection + Tracking + Homography projection
2. Pass 2: Bidirectional trajectory smoothing with velocity constraints
3. Pass 3: Global identity assignment (11v11 + goalies + referee)

Physical constraints:
- Max player speed: ~10 m/s (sprint) = ~36 km/h
- Typical cruising: ~4-6 m/s
- Pitch dimensions: 105m x 68m
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture


# Physical constants
MAX_PLAYER_SPEED_MS = 12.0  # m/s (generous for sprints)
TYPICAL_PLAYER_SPEED_MS = 6.0  # m/s
PENALTY_AREA_X = 52.5 - 16.5  # 36m from center = inside penalty area
HALF_LENGTH = 52.5  # m
HALF_WIDTH = 34.0  # m


@dataclass
class TrackObservation:
    """Single observation of a track at a specific frame."""

    frame_idx: int
    box: np.ndarray  # [x1, y1, x2, y2] in image coords
    pitch_pos: Optional[Tuple[float, float]] = None  # (x, y) in meters
    color_feature: Optional[np.ndarray] = None  # 6D Lab feature


@dataclass
class TrackData:
    """All data for a single track across the video."""

    track_id: int
    observations: List[TrackObservation] = field(default_factory=list)

    # Computed after smoothing (now done AFTER 2D projection)
    smoothed_positions: Optional[np.ndarray] = None  # (N, 2) array in pitch coords
    smoothed_frames: Optional[np.ndarray] = None  # frame indices

    # Assigned after clustering (using majority vote across lifetime)
    role: Optional[str] = None  # 'player', 'goalie', 'referee', 'linesman', 'other'
    team: Optional[int] = None  # 0 or 1 for players/goalies, -1 for officials
    player_id: Optional[int] = None  # 1-11 within team

    # Per-frame team votes for majority voting
    frame_team_votes: Optional[List[int]] = None  # List of team votes per frame

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

        # Compute bbox features
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2  # center x
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2  # center y
        h = bboxes[:, 3] - bboxes[:, 1]  # height

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

        # Y: bbox height + center_y -> pitch_y
        # Larger bbox = closer to camera = smaller pitch_y (if camera at bottom)
        # Or vice versa depending on camera position
        cy_norm = cy / self.image_height
        h_norm = h / self.image_height

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
        cy = (box[1] + box[3]) / 2
        bottom = box[3]

        # Apply calibration
        cx_norm = cx / self.image_width
        bottom_norm = bottom / self.image_height

        pitch_x = self.ref_scale_x * cx_norm + self.ref_offset_x
        pitch_y = self.ref_scale_y * bottom_norm + self.ref_offset_y

        # Clamp to pitch bounds
        pitch_x = np.clip(pitch_x, -self.pitch_length / 2, self.pitch_length / 2)
        pitch_y = np.clip(pitch_y, -self.pitch_width / 2, self.pitch_width / 2)

        return (float(pitch_x), float(pitch_y))

    def is_calibrated(self) -> bool:
        return self.ref_scale_x is not None and self.ref_scale_y is not None


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
        self.gmm = None  # Fitted GMM for per-frame voting
        self.team_a_cluster = None
        self.team_b_cluster = None
        self.ref_cluster = None

    def assign_roles(self, store: TrajectoryStore) -> Dict[int, Dict]:
        """
        Assign roles and teams to all tracks.

        Returns:
            Dict mapping track_id -> {'role': str, 'team': int, 'player_id': int}
        """
        tracks = store.get_long_tracks(min_frames=30)  # At least 1 second

        if self.debug:
            print(f"[IdentityAssigner] Processing {len(tracks)} long tracks")

        # Step 1: Compute position stats for each track
        track_stats = {}
        for track in tracks:
            stats = track.pitch_position_stats()
            if stats:
                track_stats[track.track_id] = stats

        if not track_stats:
            return {}

        # Step 2: Identify goalies by position
        goalie_candidates = self._find_goalie_candidates(track_stats)

        # Step 3: Identify linesmen by position (sidelines)
        linesman_candidates = self._find_linesman_candidates(track_stats, exclude=goalie_candidates)

        # Step 4: Cluster remaining tracks by color
        remaining_ids = [
            tid for tid in track_stats.keys() if tid not in goalie_candidates and tid not in linesman_candidates
        ]

        team_assignments, referee_id = self._cluster_by_color(store, remaining_ids)

        # Step 5: Assign goalies to teams based on which side they're on
        goalie_teams = self._assign_goalie_teams(goalie_candidates, track_stats, team_assignments)

        # Step 6: Build final assignments
        results = {}

        for tid in track_stats.keys():
            track = store.get_track(tid)

            if tid in goalie_candidates:
                results[tid] = {
                    'role': 'goalie',
                    'team': goalie_teams.get(tid, -1),
                    'player_id': 1,  # Goalies are usually #1
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
                    'player_id': None,  # Would need jersey OCR for this
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

            # Goalie criteria:
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

            # Linesman criteria:
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

    def _fit_color_gmm(self, store: TrajectoryStore, track_ids: List[int]) -> bool:
        """
        Fit GMM on ALL color observations across all tracks.
        This creates a global color model for per-frame voting.
        """
        if not track_ids:
            return False

        # Collect ALL color features from all frames
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

        # Fit GMM with 3 clusters (2 teams + 1 referee/other)
        try:
            self.gmm = GaussianMixture(
                n_components=3,
                covariance_type='diag',
                random_state=42,
                n_init=10,
                reg_covar=1e-3,
            )
            self.gmm.fit(X)
            labels = self.gmm.predict(X)

            # Find cluster sizes
            cluster_counts = np.bincount(labels, minlength=3)
            sorted_clusters = np.argsort(cluster_counts)[::-1]

            self.team_a_cluster = sorted_clusters[0]
            self.team_b_cluster = sorted_clusters[1]
            self.ref_cluster = sorted_clusters[2]

            if self.debug:
                print(f"[DEBUG] GMM fitted on {len(all_features)} color samples")
                print(f"[DEBUG] Team A cluster={self.team_a_cluster} ({cluster_counts[self.team_a_cluster]} samples)")
                print(f"[DEBUG] Team B cluster={self.team_b_cluster} ({cluster_counts[self.team_b_cluster]} samples)")
                print(f"[DEBUG] Referee cluster={self.ref_cluster} ({cluster_counts[self.ref_cluster]} samples)")

            return True

        except Exception as e:
            if self.debug:
                print(f"[DEBUG] GMM fitting failed: {e}")
            return False

    def _vote_per_frame(self, track: TrackData) -> Tuple[int, float]:
        """
        Do per-frame voting for a track using the fitted GMM.
        Returns (majority_team, confidence) where:
          - majority_team: 0, 1, or -1 (referee)
          - confidence: fraction of frames voting for majority
        """
        if self.gmm is None:
            return 0, 0.0

        votes = []  # List of (team, confidence) per frame

        for obs in track.observations:
            if obs.color_feature is None or np.linalg.norm(obs.color_feature) == 0:
                continue

            feat = obs.color_feature.reshape(1, -1)
            probs = self.gmm.predict_proba(feat)[0]
            label = np.argmax(probs)
            conf = probs[label]

            # Map cluster to team
            if label == self.team_a_cluster:
                team = 0
            elif label == self.team_b_cluster:
                team = 1
            else:
                team = -1  # Referee cluster

            votes.append((team, conf))

        if not votes:
            return 0, 0.0

        # Count votes for each team (weighted by confidence)
        team_scores = {0: 0.0, 1: 0.0, -1: 0.0}
        for team, conf in votes:
            team_scores[team] += conf

        # Majority vote
        majority_team = max(team_scores, key=team_scores.get)
        total_votes = sum(team_scores.values())
        confidence = team_scores[majority_team] / total_votes if total_votes > 0 else 0.0

        # Store per-frame votes on the track for backtracking
        track.frame_team_votes = [v[0] for v in votes]

        return majority_team, confidence

    def _cluster_by_color(self, store: TrajectoryStore, track_ids: List[int]) -> Tuple[Dict[int, int], Optional[int]]:
        """
        Cluster tracks by color features using per-frame voting with majority.
        Returns (team_assignments, referee_id).

        New approach:
        1. Fit GMM on ALL color observations (global color model)
        2. Per-frame voting: classify each frame independently
        3. Majority vote across track lifetime to determine final identity
        4. Pick referee from tracks with high referee-cluster votes
        """
        if not track_ids:
            return {}, None

        # Step 1: Fit global GMM on all color observations
        if not self._fit_color_gmm(store, track_ids):
            # Fallback to simple assignment
            return {tid: 0 for tid in track_ids}, None

        # Step 2: Per-frame voting for each track
        team_assignments = {}
        referee_candidates = []  # (track_id, ref_vote_fraction)

        for tid in track_ids:
            track = store.get_track(tid)
            if not track:
                continue

            majority_team, confidence = self._vote_per_frame(track)

            if majority_team == -1 and confidence > 0.3:
                # Strong referee signal
                referee_candidates.append((tid, confidence))
                team_assignments[tid] = -1  # Tentatively referee
            elif confidence > 0.5:
                team_assignments[tid] = majority_team
            else:
                # Low confidence - assign to team with most votes anyway
                team_assignments[tid] = majority_team if majority_team >= 0 else 0

        # Step 3: Pick single best CENTER referee based on mobility + central position
        # Exclude tracks near sidelines (those are linesmen, not center referee)
        referee_id = None
        if referee_candidates:
            best_score = -float('inf')
            for tid, ref_conf in referee_candidates:
                track = store.get_track(tid)
                stats = track.pitch_position_stats() if track else {}
                if stats:
                    # Exclude if near sideline (linesmen are near sidelines)
                    mean_y = abs(stats.get('mean_y', 0))
                    if mean_y > 30:  # Near sideline - skip, this is a linesman
                        continue

                    # Center referee score: high mobility + central position + not near sidelines
                    mobility = stats.get('std_x', 0) + stats.get('std_y', 0)
                    centrality = 1.0 / (1.0 + abs(stats.get('mean_x', 50)))
                    y_centrality = 1.0 / (1.0 + mean_y / 10)  # Prefer center of pitch
                    score = mobility * centrality * y_centrality * (1 + ref_conf)

                    if score > best_score:
                        best_score = score
                        referee_id = tid

            # Reassign other referee candidates to nearest team
            for tid, _ in referee_candidates:
                if tid != referee_id:
                    # Re-vote without referee cluster
                    track = store.get_track(tid)
                    votes_0 = sum(1 for v in track.frame_team_votes if v == 0)
                    votes_1 = sum(1 for v in track.frame_team_votes if v == 1)
                    team_assignments[tid] = 0 if votes_0 >= votes_1 else 1

        if self.debug and referee_id:
            print(f"[DEBUG] Selected referee: Track {referee_id}")

        return team_assignments, referee_id

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


@dataclass
class PlayerSlot:
    """A fixed slot for a player on the 2D pitch view."""

    slot_id: int  # 0-10 for team, or special IDs for referee
    team: int  # 0 or 1, -1 for referee
    position: Tuple[float, float]  # Current (x, y) on pitch
    last_observed_frame: int  # Last frame this slot was updated
    assigned_track_ids: List[int] = field(default_factory=list)  # All tracks assigned to this slot
    color_feature: Optional[np.ndarray] = None  # Average color for this slot
    is_goalie: bool = False


class PitchSlotManager:
    """
    Manages fixed 11v11 + 1 referee slots on the 2D pitch.

    Key features:
    - Maintains exactly 11 slots per team + 1 referee (23 total)
    - Maps fragmented tracks to consistent slot IDs
    - Keeps players at last known position when not observed
    - Uses position + color + team to match tracks to slots
    """

    def __init__(self, fps: float = 30.0, debug: bool = False):
        self.fps = fps
        self.debug = debug

        # 11 slots per team + 1 referee
        self.slots: Dict[str, PlayerSlot] = {}

        # Initialize team slots
        for team in [0, 1]:
            for i in range(11):
                slot_key = f"T{team}_{i}"
                self.slots[slot_key] = PlayerSlot(
                    slot_id=i,
                    team=team,
                    position=(0.0, 0.0),
                    last_observed_frame=-1,
                )

        # Referee slot
        self.slots["REF"] = PlayerSlot(
            slot_id=0,
            team=-1,
            position=(0.0, 0.0),
            last_observed_frame=-1,
        )

        # Track ID -> Slot key mapping
        self.track_to_slot: Dict[int, str] = {}

        # Frame-by-frame slot positions for output
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
        max_distance: float = 10.0,  # Max meters between end of track A and start of track B
    ) -> List[List[int]]:
        """
        Merge fragmented tracks that likely represent the same player.

        Uses greedy matching: for each track, find the best previous track to connect to.
        """
        if not tracks:
            return []

        # Get track info: (tid, first_frame, last_frame, first_pos, last_pos)
        track_info = []
        for tid, track, info in tracks:
            if not track.observations:
                continue

            first_obs = None
            last_obs = None
            for obs in track.observations:
                if obs.pitch_pos is not None:
                    if first_obs is None:
                        first_obs = obs
                    last_obs = obs

            if first_obs and last_obs:
                track_info.append(
                    {
                        'tid': tid,
                        'first_frame': first_obs.frame_idx,
                        'last_frame': last_obs.frame_idx,
                        'first_pos': first_obs.pitch_pos,
                        'last_pos': last_obs.pitch_pos,
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

                # Check spatial distance
                dx = curr['first_pos'][0] - prev['last_pos'][0]
                dy = curr['first_pos'][1] - prev['last_pos'][1]
                dist = np.sqrt(dx**2 + dy**2)

                if dist > max_distance:
                    continue

                # Score: prefer small gap and small distance
                score = gap + dist * 2
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
        Build frame positions for all frames using track observations.
        Slots stay at last known position until updated.
        """
        if self.debug:
            print(f"\n[PitchSlotManager] Building positions for {total_frames} frames...")

        # Build a quick lookup: (tid, frame_idx) -> pitch_pos
        observation_lookup: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for tid in self.track_to_slot:
            track = store.get_track(tid)
            if track:
                for obs in track.observations:
                    if obs.pitch_pos is not None:
                        observation_lookup[(tid, obs.frame_idx)] = obs.pitch_pos

        # Initialize slot positions from first observation
        for tid, slot_key in self.track_to_slot.items():
            track = store.get_track(tid)
            if track:
                for obs in track.observations:
                    if obs.pitch_pos is not None:
                        self.slots[slot_key].position = obs.pitch_pos
                        self.slots[slot_key].last_observed_frame = obs.frame_idx
                        break

        # Process each frame
        for frame_idx in range(total_frames):
            for tid, slot_key in self.track_to_slot.items():
                # Check if we have an observation at this frame
                key = (tid, frame_idx)
                if key in observation_lookup:
                    pos = observation_lookup[key]
                    self.slots[slot_key].position = pos
                    self.slots[slot_key].last_observed_frame = frame_idx
                    self.frame_slot_positions[frame_idx][slot_key] = pos
                else:
                    # No observation - use last known position if we have one
                    if self.slots[slot_key].last_observed_frame >= 0:
                        self.frame_slot_positions[frame_idx][slot_key] = self.slots[slot_key].position

        if self.debug:
            obs_count = sum(len(positions) for positions in self.frame_slot_positions.values())
            print(f"[PitchSlotManager] Built {obs_count} slot-frame observations")
