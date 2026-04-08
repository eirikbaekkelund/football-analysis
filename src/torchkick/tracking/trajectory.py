"""
Trajectory storage and IMM Kalman smoothing.

IMMKalmanSmoother replaces the old post-hoc Gaussian smoother with an
online Interacting Multiple Models (IMM) Kalman filter that runs two
concurrent dynamic models:

  - CV  (constant velocity, low process noise): steady-state running/jogging
  - Man (maneuvering, high process noise): acceleration, sharp direction changes

At each observation the filter blends both models weighted by likelihood,
automatically switching to the maneuvering model during sprints and cuts and
back to CV for straight-line running.  Gap filling uses the fused Kalman
velocity estimate (linear extrapolation) which is physically correct and
avoids the cubic-spline overshoot artefacts of the previous approach.

Example:
    >>> from torchkick.tracking import TrajectoryStore, TrajectorySmoother
    >>> store = TrajectoryStore(fps=30.0)
    >>> store.add_observation(track_id=1, frame_idx=0, box=box, pitch_pos=(0, 0))
    >>> smoother = TrajectorySmoother(fps=30.0)
    >>> smoother.smooth_all(store)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from torchkick.tracking.models import (
    MAX_PLAYER_SPEED_MS,
    TrackData,
    TrackObservation,
)


class TrajectoryStore:
    """
    Storage for all track data with analysis utilities.

    Accumulates observations for multiple tracks and provides
    methods for querying and filtering tracks.

    Args:
        fps: Video frame rate.

    Example:
        >>> store = TrajectoryStore(fps=30.0)
        >>> store.add_observation(1, 0, box, pitch_pos=(0, 0))
        >>> track = store.get_track(1)
    """

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps
        self.tracks: Dict[int, TrackData] = {}
        self.total_frames = 0
        self.frame_homographies: Dict[int, np.ndarray] = {}
        self.frame_keypoints: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # frame → (kps [32,2], eff_conf [32])

    def add_observation(
        self,
        track_id: int,
        frame_idx: int,
        box: np.ndarray,
        conf: Optional[float] = None,
        pitch_pos: Optional[Tuple[float, float]] = None,
        rep_mask: Optional[np.ndarray] = None,
        pose_2d: Optional[np.ndarray] = None,
        pose_2d_scores: Optional[np.ndarray] = None,
        pose_3d: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a single observation for a track.

        Args:
            track_id: Unique track identifier.
            frame_idx: Frame number.
            box: Bounding box [x1, y1, x2, y2].
            pitch_pos: Optional pitch position in meters.
            rep_mask: Optional representative mask (stored every N frames).
            pose_2d: Optional [17, 2] COCO keypoints in full-frame pixel space.
            pose_2d_scores: Optional [17] per-keypoint confidence scores.
            pose_3d: Optional [17, 3] keypoints in world coordinates (metres).
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackData(track_id=track_id)

        obs = TrackObservation(
            frame_idx=frame_idx,
            box=box,
            conf=conf,
            pitch_pos=pitch_pos,
            rep_mask=rep_mask,
            pose_2d=pose_2d,
            pose_2d_scores=pose_2d_scores,
            pose_3d=pose_3d,
        )
        self.tracks[track_id].observations.append(obs)
        self.total_frames = max(self.total_frames, frame_idx + 1)

    def get_track(self, track_id: int) -> Optional[TrackData]:
        """Get track by ID, or None if not found."""
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> List[TrackData]:
        """Get all tracks."""
        return list(self.tracks.values())

    def get_long_tracks(self, min_frames: int = 30) -> List[TrackData]:
        """
        Get tracks with at least min_frames observations.

        Args:
            min_frames: Minimum number of frames required.

        Returns:
            List of qualifying TrackData objects.
        """
        return [t for t in self.tracks.values() if t.duration_frames() >= min_frames]


class IMMKalmanSmoother:
    """
    Interacting Multiple Models (IMM) Kalman smoother for 2D player trajectories.

    Maintains two concurrent dynamic models over the 4D state [x, y, vx, vy]:

      Model 0 — CV  (constant velocity): small velocity process noise (σ_v=0.3 m/s).
                Correct for straight-line running and jogging.
      Model 1 — Man (maneuvering):       large velocity process noise (σ_v=2.0 m/s).
                Correct for direction changes, acceleration bursts, sharp cuts.

    At each observation the filter blends both estimates weighted by measurement
    likelihood.  The maneuvering model activates automatically during fast
    direction changes and returns to near-zero probability once the player is
    running steadily again.

    Gap filling between observation frames uses the fused Kalman velocity
    estimate for linear extrapolation — physically correct and free from the
    cubic-spline overshoot artefacts of the old Gaussian smoother.

    Args:
        fps: Video frame rate (Hz).
        max_speed_ms: Hard speed cap in m/s (applied per step).
        smooth_sigma: Ignored; kept for backward-compatible call sites.

    Example:
        >>> smoother = IMMKalmanSmoother(fps=30.0)
        >>> smoother.smooth_all(store, min_frames=10)
    """

    # Velocity process-noise std for the two models (m/s per root-frame)
    _CV_VEL_STD: float = 0.3
    _MAN_VEL_STD: float = 2.0
    # Measurement noise std (m) — captures homography uncertainty
    _MEAS_STD: float = 0.5
    # Markov model-switch probability per frame
    _SWITCH_PROB: float = 0.1

    def __init__(
        self,
        fps: float = 30.0,
        max_speed_ms: float = MAX_PLAYER_SPEED_MS,
        smooth_sigma: float = 2.0,  # unused; API compatibility with old TrajectorySmoother
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / fps
        self.max_speed_per_frame = max_speed_ms * self.dt

        dt = self.dt
        # State-transition matrix F: [x, y, vx, vy]
        self._F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        # Measurement matrix H: observe [x, y] only
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        # Measurement noise covariance
        self._R = np.eye(2, dtype=np.float64) * self._MEAS_STD**2
        # Per-model process-noise covariances
        self._Q = [
            np.diag([0.0, 0.0, self._CV_VEL_STD**2, self._CV_VEL_STD**2]),
            np.diag([0.0, 0.0, self._MAN_VEL_STD**2, self._MAN_VEL_STD**2]),
        ]
        # Initial state covariance
        self._P0 = np.diag([1.0, 1.0, 5.0, 5.0]).astype(np.float64)
        # Markov transition matrix [2×2]
        p = self._SWITCH_PROB
        self._Pi = np.array([[1 - p, p], [p, 1 - p]], dtype=np.float64)
        # Initial mode probabilities
        self._mu0 = np.array([0.5, 0.5], dtype=np.float64)

    # ------------------------------------------------------------------
    # Core IMM step
    # ------------------------------------------------------------------

    def _imm_step(
        self,
        xs: List[np.ndarray],
        Ps: List[np.ndarray],
        mu: np.ndarray,
        z: np.ndarray,
        dt_scale: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """
        One IMM Kalman filter step.

        Args:
            xs: Per-model state vectors [4] each.
            Ps: Per-model covariance matrices [4×4] each.
            mu: Mode probabilities [2].
            z:  Measurement [x, y].
            dt_scale: Number of frames since last observation (handles gaps).

        Returns:
            (xs_new, Ps_new, mu_new, x_fused)
        """
        n = len(xs)

        # Predicted mode probabilities c̄_j = Σ_i Π_{ij} μ_i
        c_bar = self._Pi.T @ mu  # [n]
        # Mixing weights μ_{i|j} = Π_{ij} μ_i / c̄_j
        mu_ij = (self._Pi * mu[:, np.newaxis]) / np.maximum(c_bar[np.newaxis, :], 1e-300)

        # Scale F and Q for actual time gap
        F = self._F.copy()
        F[0, 2] = self.dt * dt_scale
        F[1, 3] = self.dt * dt_scale

        # Mixed initial conditions for each model
        x_mix = [sum(mu_ij[i, j] * xs[i] for i in range(n)) for j in range(n)]
        P_mix = [
            sum(mu_ij[i, j] * (Ps[i] + np.outer(xs[i] - x_mix[j], xs[i] - x_mix[j])) for i in range(n))
            for j in range(n)
        ]

        xs_new: List[np.ndarray] = []
        Ps_new: List[np.ndarray] = []
        likelihoods: List[float] = []

        for j in range(n):
            Q_j = self._Q[j] * dt_scale

            # Predict
            x_p = F @ x_mix[j]
            P_p = F @ P_mix[j] @ F.T + Q_j

            # Update
            S = self._H @ P_p @ self._H.T + self._R
            K = P_p @ self._H.T @ np.linalg.solve(S, np.eye(2))
            innov = z - self._H @ x_p
            x_u = x_p + K @ innov
            P_u = (np.eye(4) - K @ self._H) @ P_p

            # Measurement likelihood via log-det for numerical stability
            try:
                sign, logdet = np.linalg.slogdet(S)
                if sign > 0:
                    maha = float(innov.T @ np.linalg.solve(S, innov))
                    L = np.exp(-0.5 * maha - 0.5 * logdet - np.log(2 * np.pi))
                else:
                    L = 1e-300
            except np.linalg.LinAlgError:
                L = 1e-300

            xs_new.append(x_u)
            Ps_new.append(P_u)
            likelihoods.append(max(float(L), 1e-300))

        # Update mode probabilities
        L_arr = np.array(likelihoods)
        mu_raw = L_arr * c_bar
        mu_sum = mu_raw.sum()
        mu_new = mu_raw / mu_sum if mu_sum > 1e-300 else np.full(n, 1.0 / n)

        # Fused state estimate
        x_fused = sum(mu_new[j] * xs_new[j] for j in range(n))

        return xs_new, Ps_new, mu_new, x_fused

    # ------------------------------------------------------------------
    # Public API (mirrors TrajectorySmoother)
    # ------------------------------------------------------------------

    def smooth_track(self, track: TrackData, interpolate: bool = True) -> bool:
        """
        Smooth a single track's trajectory via sequential IMM Kalman filtering.

        Args:
            track: TrackData to process in-place.
            interpolate: Fill gaps between observations using Kalman velocity.

        Returns:
            True if successful, False if insufficient data.
        """
        obs_with_pos = [(obs.frame_idx, obs.pitch_pos) for obs in track.observations if obs.pitch_pos is not None]
        if len(obs_with_pos) < 5:
            return False

        obs_with_pos.sort(key=lambda x: x[0])
        frames = [o[0] for o in obs_with_pos]
        positions = [o[1] for o in obs_with_pos]

        # Initialise from first two observations
        if len(positions) >= 2:
            df = max(1, frames[1] - frames[0])
            vx = (positions[1][0] - positions[0][0]) / (df * self.dt)
            vy = (positions[1][1] - positions[0][1]) / (df * self.dt)
        else:
            vx, vy = 0.0, 0.0

        x0 = np.array([positions[0][0], positions[0][1], vx, vy], dtype=np.float64)
        xs = [x0.copy(), x0.copy()]
        Ps = [self._P0.copy(), self._P0.copy()]
        mu = self._mu0.copy()

        # (frame, x, y, vx, vy) at each observation
        smoothed: List[Tuple[int, float, float, float, float]] = [(frames[0], positions[0][0], positions[0][1], vx, vy)]

        for i in range(1, len(obs_with_pos)):
            frame_idx, pos = obs_with_pos[i]
            dt_scale = float(max(1, frame_idx - frames[i - 1]))
            z = np.array([pos[0], pos[1]], dtype=np.float64)

            xs, Ps, mu, x_fused = self._imm_step(xs, Ps, mu, z, dt_scale)

            # Hard speed constraint
            px, py = smoothed[-1][1], smoothed[-1][2]
            dx, dy = x_fused[0] - px, x_fused[1] - py
            dist = np.hypot(dx, dy)
            max_dist = self.max_speed_per_frame * dt_scale
            if dist > max_dist and dist > 0:
                scale = max_dist / dist
                x_fused[0] = px + dx * scale
                x_fused[1] = py + dy * scale

            smoothed.append((frame_idx, x_fused[0], x_fused[1], x_fused[2], x_fused[3]))

        # Build dense frame array (gap-fill via Kalman velocity)
        if interpolate:
            all_frames: List[int] = []
            all_positions: List[Tuple[float, float]] = []
            for i, (f, x, y, vxi, vyi) in enumerate(smoothed):
                all_frames.append(f)
                all_positions.append((x, y))
                if i < len(smoothed) - 1:
                    next_f = smoothed[i + 1][0]
                    for gf in range(f + 1, next_f):
                        t = gf - f
                        all_frames.append(gf)
                        all_positions.append((x + vxi * t * self.dt, y + vyi * t * self.dt))
        else:
            all_frames = [s[0] for s in smoothed]
            all_positions = [(s[1], s[2]) for s in smoothed]

        track.smoothed_positions = np.array(all_positions, dtype=np.float32)
        track.smoothed_frames = np.array(all_frames)
        return True

    def smooth_all(self, store: TrajectoryStore, min_frames: int = 10) -> int:
        """
        Smooth all qualifying tracks in the store.

        Args:
            store: TrajectoryStore to process.
            min_frames: Minimum frames required for smoothing.

        Returns:
            Number of successfully smoothed tracks.
        """
        count = 0
        for track in store.tracks.values():
            if track.duration_frames() >= min_frames:
                if self.smooth_track(track):
                    count += 1
        return count


# Backward-compatible alias — existing code that imports TrajectorySmoother still works
TrajectorySmoother = IMMKalmanSmoother


__all__ = [
    "TrajectoryStore",
    "IMMKalmanSmoother",
    "TrajectorySmoother",
]
