"""
Match analysis inference pipeline.

Runs YOLO detection + BotSORT tracking, YOLO-pose pitch keypoint estimation
for 2D homography, and appearance-based team assignment.

Output: annotated video with team-coloured bounding boxes + 2D pitch minimap.

Example:
    >>> from torchkick.inference import run_analysis
    >>> output = run_analysis(
    ...     video_path="match.mp4",
    ...     yolo_weights="weights/yolo11l_football/best.pt",
    ...     pitch_weights="weights/keypoints/best.pt",
    ... )
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from torchkick.tracking import (
    TrajectoryStore,
    TrajectorySmoother,
    IdentityAssigner,
    HomographyEstimator,
    KeypointTracker,
    PitchVisualizer,
    PITCH_LINE_COORDINATES,
    HALF_LENGTH,
    HALF_WIDTH,
)
from torchkick.utils import VideoReader, VideoWriter, ProgressTracker, generate_output_path
from torchkick.utils.crops import crop_box as _crop_box


# ---------------------------------------------------------------------------
# Pass 0: Team centroid calibration (first 2 min)
# ---------------------------------------------------------------------------


def calibrate_team_centroids(
    video_path: str,
    detector,
    embedder,
    calibration_duration: float = 120.0,
    n_sample_frames: int = 40,
    conf: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Fit k=3 team centroids from randomly sampled frames in the first
    ``calibration_duration`` seconds.

    Samples ``n_sample_frames`` random frames, runs YOLO detection, embeds
    all player crops with ``embedder``, then fits k-means k=3.  The smallest
    cluster is remapped to label 2 (referee convention).

    Args:
        video_path: Input video path.
        detector: ``ultralytics.YOLO`` model.
        embedder: ``DINOv2ReIDEmbedder`` or ``SigLIPTeamEmbedder`` with
            ``embed(crops) -> [N, D]``.
        calibration_duration: Seconds to sample from (default 120).
        n_sample_frames: Number of frames to randomly sample.
        conf: Detection confidence threshold.

    Returns:
        ``np.ndarray [3, D]`` centroids (team0, team1, ref) or None if
        fewer than 10 crops are collected.
    """
    print(f"Calibrating team centroids from first {calibration_duration:.0f}s …")
    all_crops: List[np.ndarray] = []

    with VideoReader(video_path, max_duration=calibration_duration) as reader:
        total = reader.max_frames
        sampled_set = set(random.sample(range(total), min(n_sample_frames, total)))

        for frame_idx, frame_bgr in enumerate(reader):
            if frame_idx not in sampled_set:
                continue
            results = detector(frame_bgr, verbose=False, conf=conf, classes=[0])
            if results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            for box in results[0].boxes.xyxy.cpu().numpy():
                crop = _crop_box(frame_rgb, box.tolist())
                if crop is not None and crop.size > 0:
                    all_crops.append(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    if len(all_crops) < 10:
        print(f"[warn] Only {len(all_crops)} crops for calibration — skipping.")
        return None

    print(f"  Embedding {len(all_crops)} crops …")
    embeddings = embedder.embed(all_crops)

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    counts = np.bincount(labels, minlength=3)
    ref_cluster = int(np.argmin(counts))
    centers = km.cluster_centers_.copy()
    if ref_cluster != 2:
        centers[[2, ref_cluster]] = centers[[ref_cluster, 2]]

    team_counts = list(counts)
    print(
        f"  Clusters: team0={team_counts[0 if ref_cluster != 0 else 2]}, "
        f"team1={team_counts[1 if ref_cluster != 1 else 2]}, ref={team_counts[ref_cluster]}"
    )
    return centers.astype(np.float32)


# ---------------------------------------------------------------------------
# Pass 1: Detection, tracking, projection
# ---------------------------------------------------------------------------


def detect_and_project(
    video_path: str,
    detector,
    pitch_kp_detector,
    max_duration: Optional[float] = None,
    homography_interval: int = 1,
    siglip_embedder=None,
    reid_embedder=None,
    reid_interval: int = 5,
    conf: float = 0.5,
) -> TrajectoryStore:
    """
    Pass 1: YOLO detection + BotSORT tracking + 2D pitch projection.

    Runs YOLO with BotSORT on every frame.  Every ``reid_interval`` frames,
    player crops are embedded with ``siglip_embedder`` (or ``reid_embedder``
    if provided) and stored in observations for later team clustering.

    Args:
        video_path: Input video path.
        detector: ``ultralytics.YOLO`` model.
        pitch_kp_detector: Pitch keypoint detector (YOLO-pose) for homography.
            Pass None to skip homography estimation.
        max_duration: Maximum duration in seconds.
        homography_interval: Frames between homography updates.
        siglip_embedder: Zero-shot team embedder (used when reid_embedder is None).
        reid_embedder: Fine-tuned ReID embedder (takes priority over siglip).
        reid_interval: Frames between embedding updates.
        conf: Detection confidence threshold.

    Returns:
        ``TrajectoryStore`` with all observations.
    """
    print("=" * 60)
    print("PASS 1: Detection + Tracking + Projection")
    print("=" * 60)

    homography = HomographyEstimator(
        min_correspondences=6,
        confidence_threshold=0.3,
        visibility_threshold=0.3,
        use_kalman=False,
    )
    kp_tracker = KeypointTracker()
    _embedder = reid_embedder or siglip_embedder

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata
        store = TrajectoryStore(fps=meta.fps)
        progress = ProgressTracker(reader.max_frames, log_interval=100)

        for frame_idx, frame_bgr in enumerate(reader):
            # Homography update
            if frame_idx % homography_interval == 0 and pitch_kp_detector is not None:
                kps, conf_kps = pitch_kp_detector.detect(frame_bgr)
                kps_smooth, eff_conf = kp_tracker.update(kps, conf_kps)
                ok = homography.estimate(kps_smooth, eff_conf, eff_conf, frame_bgr.shape[:2])
                if ok and homography.H_inv is not None:
                    store.frame_homographies[frame_idx] = homography.H_inv.copy()

            # Detection + tracking
            results = detector.track(
                frame_bgr,
                persist=True,
                tracker="botsort.yaml",
                verbose=False,
                classes=[0],
                conf=conf,
            )

            if results[0].boxes.id is None:
                progress.update()
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Store observations
            for box, track_id in zip(boxes, track_ids):
                pitch_pos = homography.project_player_to_pitch(box.tolist())
                store.add_observation(
                    track_id=track_id,
                    frame_idx=frame_idx,
                    box=box,
                    pitch_pos=pitch_pos,
                )

            # ReID embeddings every reid_interval frames
            if _embedder is not None and frame_idx % reid_interval == 0 and len(boxes) > 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                crops_bgr = []
                valid_pairs = []
                for box, track_id in zip(boxes, track_ids):
                    crop = _crop_box(frame_rgb, box.tolist())
                    if crop is not None and crop.size > 0:
                        crops_bgr.append(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                        valid_pairs.append(track_id)

                if crops_bgr:
                    embeddings = _embedder.embed(crops_bgr)
                    for track_id, emb in zip(valid_pairs, embeddings):
                        track = store.get_track(track_id)
                        if track and track.observations and track.observations[-1].frame_idx == frame_idx:
                            track.observations[-1].reid_embedding = emb

            progress.update()
            if progress.should_log():
                print(progress.status())

        store.total_frames = frame_idx + 1

    print(f"Complete: {len(store.tracks)} tracks over {store.total_frames} frames")
    return store


# ---------------------------------------------------------------------------
# Pass 1.5: Track re-linking via ReID
# ---------------------------------------------------------------------------


def relink_tracks(
    store: TrajectoryStore,
    similarity_threshold: float = 0.7,
    max_gap_frames: int = 150,
    max_distance_m: float = 15.0,
) -> int:
    """
    Merge fragmented track IDs using ReID cosine similarity.

    Merges pairs (A, B) where B starts shortly after A ends, they are
    spatially close at the junction, and their mean ReID embeddings are
    similar.  Skips tracks without stored ReID embeddings.

    Returns:
        Number of track merges performed.
    """
    print("=" * 60)
    print("PASS 1.5: Track Re-linking")
    print("=" * 60)

    track_info: Dict[int, tuple] = {}
    for tid, track in store.tracks.items():
        if not track.observations:
            continue
        obs_sorted = sorted(track.observations, key=lambda o: o.frame_idx)
        embeddings = [o.reid_embedding for o in obs_sorted if o.reid_embedding is not None]
        if not embeddings:
            continue
        mean_embed = np.mean(embeddings, axis=0).astype(np.float64)
        mean_embed /= np.linalg.norm(mean_embed) + 1e-8
        last_pos = next((o.pitch_pos for o in reversed(obs_sorted) if o.pitch_pos is not None), None)
        first_pos = next((o.pitch_pos for o in obs_sorted if o.pitch_pos is not None), None)
        track_info[tid] = (obs_sorted[0].frame_idx, obs_sorted[-1].frame_idx, mean_embed, last_pos, first_pos)

    parent: Dict[int, int] = {tid: tid for tid in track_info}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tids_sorted = sorted(track_info, key=lambda t: track_info[t][1])
    merges = 0

    for i, tid_a in enumerate(tids_sorted):
        first_a, last_a, embed_a, _, last_pos_a = track_info[tid_a]
        for tid_b in tids_sorted[i + 1 :]:
            first_b, _, embed_b, first_pos_b, _ = track_info[tid_b]
            gap = first_b - last_a
            if gap < 0 or gap > max_gap_frames:
                continue
            if find(tid_a) == find(tid_b):
                continue
            if last_pos_a is not None and first_pos_b is not None:
                if float(np.hypot(last_pos_a[0] - first_pos_b[0], last_pos_a[1] - first_pos_b[1])) > max_distance_m:
                    continue
            if float(np.dot(embed_a, embed_b)) < similarity_threshold:
                continue
            parent[find(tid_b)] = find(tid_a)
            merges += 1

    if merges > 0:
        root_groups: Dict[int, List[int]] = defaultdict(list)
        for tid in track_info:
            root_groups[find(tid)].append(tid)
        for root, members in root_groups.items():
            if len(members) <= 1:
                continue
            root_track = store.tracks[root]
            for tid in members:
                if tid == root:
                    continue
                if tid in store.tracks:
                    root_track.observations.extend(store.tracks[tid].observations)
                    del store.tracks[tid]
            root_track.observations.sort(key=lambda o: o.frame_idx)

    print(f"Complete: {merges} merges ({len(store.tracks)} tracks remaining)")
    return merges


# ---------------------------------------------------------------------------
# Pass 2: Trajectory smoothing
# ---------------------------------------------------------------------------


def smooth_trajectories(store: TrajectoryStore) -> int:
    """Pass 2: Gaussian-smooth all trajectories. Returns number smoothed."""
    print("=" * 60)
    print("PASS 2: Trajectory Smoothing")
    print("=" * 60)
    smoother = TrajectorySmoother(fps=store.fps, max_speed_ms=12.0, smooth_sigma=7.0)
    count = smoother.smooth_all(store, min_frames=10)
    print(f"Complete: {count} trajectories smoothed")
    return count


# ---------------------------------------------------------------------------
# Pass 3: Identity / team assignment
# ---------------------------------------------------------------------------


def assign_identities(
    store: TrajectoryStore,
    reid_embedder=None,
    siglip_embedder=None,
    team_centroids: Optional[np.ndarray] = None,
) -> Dict:
    """
    Pass 3: Team classification and goalkeeper detection.

    Uses per-track ReID embeddings (stored during Pass 1) for k=3 clustering
    (home, away, referee).  Goalkeeper inference uses position isolation:
    the most extreme-x track on each side with a ≥5 m gap from the next
    player is the GK; looser threshold when the track is an embedding outlier
    from both team centroids (different-coloured jersey).

    Args:
        store: TrajectoryStore with observations.
        reid_embedder: Fine-tuned ``DINOv2ReIDEmbedder`` (optional).
        siglip_embedder: Zero-shot ``SigLIPTeamEmbedder`` fallback.
        team_centroids: ``[3, D]`` centroids from ``calibrate_team_centroids()``.

    Returns:
        Dict mapping track_id → ``{'role', 'team'}``.
    """
    print("=" * 60)
    print("PASS 3: Identity Assignment")
    print("=" * 60)
    assigner = IdentityAssigner(
        fps=store.fps,
        embedder=reid_embedder,
        siglip_embedder=siglip_embedder,
        team_centroids=team_centroids,
        debug=True,
    )
    assignments = assigner.assign_roles(store)
    print(f"Complete: {len(assignments)} identities assigned")
    return assignments


# ---------------------------------------------------------------------------
# Pass 4: Visualization
# ---------------------------------------------------------------------------

_TEAM_COLORS = {
    0: (0, 0, 255),  # team 0 — red
    1: (255, 50, 50),  # team 1 — blue
    -1: (0, 255, 0),  # unknown — green
}
_REF_COLOR = (50, 50, 50)  # dark grey
_GK_COLOR_T0 = (0, 255, 255)  # cyan
_GK_COLOR_T1 = (255, 255, 0)  # yellow


def render_visualization(
    video_path: str,
    store: TrajectoryStore,
    assignments: Dict,
    max_duration: Optional[float] = None,
    draw_overlay: bool = True,
) -> str:
    """
    Pass 4: Render annotated video with 2D pitch minimap.

    Draws team-coloured bounding boxes on the video frame and a pitch
    minimap showing player positions as coloured dots.

    Args:
        video_path: Input video path.
        store: TrajectoryStore with all data.
        assignments: ``track_id → {'role', 'team'}`` from ``assign_identities()``.
        max_duration: Maximum duration in seconds.
        draw_overlay: Draw pitch line overlay on video frame.

    Returns:
        Path to output video.
    """
    print("=" * 60)
    print("PASS 4: Visualization")
    print("=" * 60)

    pitch_viz = PitchVisualizer()

    # Build per-frame lookup
    frame_obs: Dict[int, list] = defaultdict(list)
    for track_id, track in store.tracks.items():
        info = assignments.get(track_id, {"role": "unknown", "team": -1})
        for obs in track.observations:
            frame_obs[obs.frame_idx].append(
                {
                    "box": obs.box,
                    "pitch_pos": obs.pitch_pos,
                    "role": info.get("role", "unknown"),
                    "team": info.get("team", -1),
                    "track_id": track_id,
                }
            )

    output_path = generate_output_path(video_path, prefix="torchkick_analysis", duration=max_duration)
    current_H_inv = None

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata
        pitch_h, pitch_w = pitch_viz.base_pitch.shape[:2]
        scale = meta.height / pitch_h
        output_w = meta.width + int(pitch_w * scale)
        progress = ProgressTracker(reader.max_frames, log_interval=100)

        with VideoWriter(output_path, meta.fps, (output_w, meta.height)) as writer:
            for frame_idx, frame_bgr in enumerate(reader):
                obs_list = frame_obs.get(frame_idx, [])

                # Update homography for overlay
                if draw_overlay:
                    for check_idx in range(frame_idx, -1, -1):
                        if check_idx in store.frame_homographies:
                            current_H_inv = store.frame_homographies[check_idx]
                            break
                    if current_H_inv is not None:
                        frame_bgr = _draw_pitch_overlay(frame_bgr, current_H_inv)

                # Draw bounding boxes
                for obs in obs_list:
                    box = obs["box"]
                    role = obs["role"]
                    team = obs["team"]
                    x1, y1, x2, y2 = map(int, box)

                    if role == "goalie":
                        color = _GK_COLOR_T0 if team == 0 else _GK_COLOR_T1
                        label = f"GK{obs['track_id']}"
                    elif role in ("referee", "linesman"):
                        color = _REF_COLOR
                        label = "REF"
                    else:
                        color = _TEAM_COLORS.get(team, _TEAM_COLORS[-1])
                        label = str(obs["track_id"])

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # Build pitch minimap via PitchVisualizer (margin-correct positioning)
                minimap_entries = []
                for obs in obs_list:
                    pos = obs.get("pitch_pos")
                    if pos is None:
                        continue
                    role = obs["role"]
                    team = obs["team"]
                    if role == "goalie":
                        # Encode GK as team offset 10 — handled below
                        minimap_entries.append((pos[0], pos[1], obs["track_id"], team + 10))
                    elif role in ("referee", "linesman"):
                        minimap_entries.append((pos[0], pos[1], obs["track_id"], 2))
                    else:
                        minimap_entries.append((pos[0], pos[1], obs["track_id"], team))

                pitch_img = pitch_viz.base_pitch.copy()
                for x, y, tid, team_code in minimap_entries:
                    if abs(x) > HALF_LENGTH + 5 or abs(y) > HALF_WIDTH + 5:
                        continue
                    px_dot, py_dot = pitch_viz._pitch_to_pixel(x, y)  # margin-aware
                    if team_code >= 10:
                        dot_color = _GK_COLOR_T0 if (team_code - 10) == 0 else _GK_COLOR_T1
                    elif team_code == 2:
                        dot_color = _REF_COLOR
                    else:
                        dot_color = _TEAM_COLORS.get(team_code, _TEAM_COLORS[-1])
                    cv2.circle(pitch_img, (px_dot, py_dot), 5, dot_color, -1)
                    cv2.circle(pitch_img, (px_dot, py_dot), 5, (0, 0, 0), 1)

                pitch_scaled = cv2.resize(pitch_img, (int(pitch_w * scale), meta.height))
                writer.write(np.hstack([frame_bgr, pitch_scaled]))

                progress.update()
                if progress.should_log():
                    print(f"Pass 4: {progress.status()}")

    print(f"Complete: {output_path}")
    return output_path


def _draw_pitch_overlay(
    frame: np.ndarray,
    H_inv: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw pitch line wireframe on a frame using inverse homography."""
    frame_viz = frame.copy()
    h, w = frame.shape[:2]
    for _, points in PITCH_LINE_COORDINATES.items():
        pitch_pts = np.array([p.to_array() for _, p in points], dtype=np.float32)
        ones = np.ones((len(pitch_pts), 1), dtype=np.float32)
        projected = (H_inv @ np.hstack([pitch_pts, ones]).T).T
        projected = projected[:, :2] / projected[:, 2:3]
        valid = [(int(p[0]), int(p[1])) for p in projected if -500 < p[0] < w + 500 and -500 < p[1] < h + 500]
        if len(valid) >= 2:
            for i in range(len(valid) - 1):
                cv2.line(frame_viz, valid[i], valid[i + 1], color, thickness)
    return frame_viz


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    video_path: str,
    yolo_weights: str,
    pitch_weights: Optional[str] = None,
    reid_weights: Optional[str] = None,
    duration: Optional[float] = None,
    homography_interval: int = 1,
    reid_interval: int = 5,
    conf: float = 0.5,
    draw_overlay: bool = True,
    device: Optional[str] = None,
) -> str:
    """
    Run complete match analysis pipeline.

    Passes:
        0. Calibrate team centroids from first 2 min (random frame sample)
        1. YOLO detection + BotSORT tracking + 2D pitch projection
        1.5. Re-link fragmented tracks via ReID cosine similarity
        2. Smooth trajectories
        3. Assign team identities + goalkeeper detection
        4. Render annotated video with pitch minimap

    Args:
        video_path: Input video path.
        yolo_weights: Path to YOLO player detection weights.
        pitch_weights: Path to YOLO-pose pitch keypoint weights.
            If None, homography/2D projection is skipped.
        reid_weights: Path to fine-tuned ReID checkpoint.
            If None, SigLIP zero-shot team embedder is used automatically.
        duration: Maximum duration in seconds (None = full video).
        homography_interval: Frames between homography updates.
        reid_interval: Frames between embedding extraction.
        conf: Detection confidence threshold.
        draw_overlay: Draw pitch line wireframe on video.
        device: Device string (auto-detected when None).

    Returns:
        Path to output video.

    Example:
        >>> output = run_analysis(
        ...     video_path="match.mp4",
        ...     yolo_weights="weights/yolo11l_football/best.pt",
        ...     pitch_weights="weights/keypoints/best.pt",
        ... )
    """
    if device:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}")

    # Load YOLO detector
    from ultralytics import YOLO

    detector = YOLO(yolo_weights)
    detector.to(dev)

    # Load pitch keypoint detector (DINOv2 heatmap)
    pitch_kp_detector = None
    if pitch_weights is not None:
        from torchkick.models.pitch import HeatmapPitchDetector

        pitch_kp_detector = HeatmapPitchDetector(weights_path=pitch_weights, device=str(dev))
        print(f"Pitch keypoint detector loaded: {pitch_weights}")

    # Load ReID embedder
    reid_embedder = None
    if reid_weights is not None:
        try:
            from torchkick.models.reid import DINOv2ReIDEmbedder

            reid_embedder = DINOv2ReIDEmbedder(weights_path=reid_weights, device=str(dev))
            print(f"ReID embedder loaded: {reid_weights}")
        except Exception as e:
            print(f"[warn] Could not load ReID embedder: {e}")

    # Zero-shot SigLIP fallback
    siglip_embedder = None
    if reid_embedder is None:
        try:
            from torchkick.models.reid import SigLIPTeamEmbedder

            siglip_embedder = SigLIPTeamEmbedder(device=str(dev))
            print("SigLIP zero-shot team embedder loaded")
        except Exception as e:
            print(f"[warn] SigLIP unavailable: {e}")

    # Pass 0: Calibrate team centroids
    team_centroids = None
    _calib_embedder = reid_embedder or siglip_embedder
    if _calib_embedder is not None:
        try:
            team_centroids = calibrate_team_centroids(video_path, detector, _calib_embedder, conf=conf)
        except Exception as e:
            print(f"[warn] Centroid calibration failed: {e}")

    # Pass 1: Detection + tracking + projection
    store = detect_and_project(
        video_path,
        detector,
        pitch_kp_detector,
        max_duration=duration,
        homography_interval=homography_interval,
        siglip_embedder=siglip_embedder,
        reid_embedder=reid_embedder,
        reid_interval=reid_interval,
        conf=conf,
    )

    # Pass 1.5: Re-link fragmented tracks
    relink_tracks(store)

    # Pass 2: Smooth trajectories
    smooth_trajectories(store)

    # Pass 3: Identity assignment
    assignments = assign_identities(
        store,
        reid_embedder=reid_embedder,
        siglip_embedder=siglip_embedder,
        team_centroids=team_centroids,
    )

    # Pass 4: Render
    output_path = render_visualization(video_path, store, assignments, max_duration=duration, draw_overlay=draw_overlay)

    print("\n" + "=" * 60)
    print(f"ANALYSIS COMPLETE  →  {output_path}")
    print("=" * 60)
    return output_path
