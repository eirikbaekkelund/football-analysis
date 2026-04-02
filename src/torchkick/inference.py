"""
Match analysis inference pipeline.

This module provides the main inference pipeline for football match
analysis, combining player detection, tracking, homography estimation,
team classification, and visualization.


Example:
    CLI usage:
    
    ```bash
    # Basic inference
    torchkick analyze --video match.mp4 --output output.mp4
    
    # With specific model
    torchkick analyze --video match.mp4 --model yolo --duration 60
    
    # Full pipeline with all options
    torchkick analyze \\
        --video match.mp4 \\
        --model fcnn \\
        --duration 120 \\
        --homography-interval 1 \\
        --dominance
    ```
    
    Python API:
    
    >>> from torchkick.inference import run_analysis
    >>> 
    >>> output = run_analysis(
    ...     video_path="match.mp4",
    ...     model_type="yolo",
    ...     duration=60.0,
    ... )
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from torchkick.tracking import (
    TrajectoryStore,
    TrajectorySmoother,
    ByteTracker,
    MaskIoUTracker,
    IdentityAssigner,
    PitchSlotManager,
    HomographyEstimator,
    PitchVisualizer,
    PITCH_LINE_COORDINATES,
)
from torchkick.utils import (
    VideoReader,
    VideoWriter,
    ProgressTracker,
    generate_output_path,
)


from torchkick.utils.crops import crop_box as _crop_box


def load_models(
    device: torch.device,
    model_path: str,
    model_type: str,
    reid_weights: Optional[str] = None,
    pitch_weights: Optional[str] = None,
    pitch_detector_type: str = "yolo",
) -> Tuple:
    """
    Load detection and homography models, and optionally the ReID embedder.

    Args:
        device: Torch device.
        model_path: Path to detection model weights.
        model_type: "yolo", "fcnn", "rtdetr", "rfdetr", "sam3_mlx", or "sam3_pytorch".
        reid_weights: Optional path to ReID student checkpoint. When provided,
            a ``DINOv2ReIDEmbedder`` is included in the return tuple.
        pitch_weights: Optional path to pitch keypoint model weights.
            When provided, loads a ``YOLOPoseKeypointDetector`` or
            ``ViTPoseKeypointDetector`` for homography estimation.
            When None, homography estimation is skipped.
        pitch_detector_type: "yolo" (faster, default) or "vitpose" (more accurate).

    Returns:
        ``(detector, pitch_kp_detector, reid_embedder)`` — ``pitch_kp_detector``
        and ``reid_embedder`` are None when the corresponding weights are not given.
    """
    # Load pitch keypoint detector (optional)
    pitch_kp_detector = None
    if pitch_weights is not None:
        from torchkick.models.pitch import ViTPoseKeypointDetector, YOLOPoseKeypointDetector

        if pitch_detector_type == "vitpose":
            pitch_kp_detector = ViTPoseKeypointDetector(
                weights_path=pitch_weights,
                device=str(device),
            )
        else:
            pitch_kp_detector = YOLOPoseKeypointDetector(
                weights_path=pitch_weights,
                device=str(device),
            )

    # Load detection model
    if model_type == "yolo":
        from ultralytics import YOLO

        detector = YOLO(model_path)
        detector.to(device)
    elif model_type == "fcnn":
        from torchkick.training import get_player_detector_model

        detector = get_player_detector_model(num_classes=2)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            detector.load_state_dict(checkpoint['model_state_dict'])
        else:
            detector.load_state_dict(checkpoint)
        detector.to(device)
        detector.eval()
    elif model_type == "rtdetr":
        from torchkick.models import RTDETRDetector

        detector = RTDETRDetector(
            weights_path=model_path,  # None → uses base HuggingFace weights
            device=str(device),
        )
    elif model_type == "rfdetr":
        from torchkick.models import RFDETRDetector

        detector = RFDETRDetector(
            weights_path=model_path,  # None → downloads pretrained weights
            device=str(device),
        )
    elif model_type == "sam3_mlx":
        # MLX-based SAM3 lightweight predictor (CPU/MLX backend)
        from torchkick.models.sam_3._mlx.videopredictor import Sam3VideoPredictorMLX

        # resolve assets relative to package
        script_dir = Path(__file__).parent / "models" / "sam_3" / "_mlx"
        bpe_path = str((script_dir / "assets" / "bpe.txt.gz").resolve())
        checkpoint_path = str((script_dir / "assets" / "model.safetensors").resolve())

        detector = Sam3VideoPredictorMLX.from_assets(bpe_path=bpe_path, checkpoint_path=checkpoint_path)
    elif model_type == "sam3_pytorch":
        # Defer loading to the external sam3 pytorch implementation (requires GPU/triton)
        from sam3 import Sam3VideoPredictor

        predictor = Sam3VideoPredictor.from_pretrained("facebook/sam3")
        detector = predictor
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Use 'yolo', 'fcnn', 'rtdetr', 'rfdetr', 'sam3_mlx' or 'sam3_pytorch'."
        )

    reid_embedder = None
    if reid_weights is not None:
        try:
            from torchkick.models.reid import DINOv2ReIDEmbedder

            reid_embedder = DINOv2ReIDEmbedder(
                weights_path=reid_weights,
                device=str(device),
            )
            print(f"ReID embedder loaded from {reid_weights}")
        except Exception as e:
            print(f"[warn] Could not load ReID embedder: {e}. Continuing without ReID.")

    return detector, pitch_kp_detector, reid_embedder


def _detect_and_project_sam3_mlx(
    video_path: str,
    predictor: "Sam3VideoPredictorMLX",
    homography: HomographyEstimator,
    store: TrajectoryStore,
    progress: ProgressTracker,
):
    """Run MLX-based Sam3 predictor over a video and populate the TrajectoryStore.

    Iterates the propagated mask results frame-by-frame, projects each mask's
    bounding box to pitch coordinates via the Kalman-smoothed homography, and
    stores observations.  Kalman prediction covers gaps up to 30 frames without
    a new keyframe estimate.
    """
    sess = predictor.start_session(video_path)
    session_id = sess["session_id"]

    mask_tracker = MaskIoUTracker(max_age=30, iou_threshold=0.5)
    rep_mask_interval = max(1, int(store.fps))

    for frame_idx, per_obj in predictor.propagate_in_video(session_id, start_frame_idx=0):
        progress.update(1)

        frame_masks = []
        for obj_id, outputs in per_obj.items():
            masks = outputs.get("masks")
            if masks is None:
                continue
            try:
                arr = np.array(masks)
            except Exception:
                arr = masks

            if hasattr(arr, "ndim"):
                if arr.ndim == 4:
                    m = arr[0, 0]
                elif arr.ndim == 3:
                    m = arr[0]
                else:
                    m = arr
            else:
                m = arr

            frame_masks.append((m > 0.5).astype(np.uint8))

        matches = mask_tracker.update(frame_masks)

        for track_id, mask in matches:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            box = [float(x1), float(y1), float(x2), float(y2)]

            pitch_pos = homography.project_player_to_pitch(box)
            rep_mask = mask if (frame_idx % rep_mask_interval == 0) else None

            store.add_observation(
                track_id=track_id,
                frame_idx=frame_idx,
                box=box,
                pitch_pos=pitch_pos,
                rep_mask=rep_mask,
            )

    predictor.close_session(session_id)


def _box_iou(a, b) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _attach_reid_embeddings(
    store: TrajectoryStore,
    frame_idx: int,
    active_tracks: list,
    detections: list,
    det_embeddings: np.ndarray,
    reid_embedder=None,
) -> None:
    """
    Match active track boxes to detection boxes via IoU and store ReID embeddings
    and team_probs (with temporal EMA) in the most recent observation.

    Team probabilities are updated via a 0.7/0.3 EMA stored on the TrackData to
    prevent per-frame label flipping.  The EMA value is written back to
    ``obs.team_probs`` so identity assignment always sees smoothed probabilities.
    """
    det_probs = None
    if reid_embedder is not None:
        _, det_probs = reid_embedder.classify_from_embeddings(det_embeddings)

    for track_id, track_box in active_tracks:
        best_iou = 0.0
        best_emb_idx = -1
        for det_idx, det_box in enumerate(detections):
            iou = _box_iou(track_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_emb_idx = det_idx

        if best_emb_idx >= 0 and best_iou > 0.2:
            track = store.get_track(track_id)
            if track and track.observations and track.observations[-1].frame_idx == frame_idx:
                obs = track.observations[-1]
                obs.reid_embedding = det_embeddings[best_emb_idx]
                if det_probs is not None:
                    probs_new = det_probs[best_emb_idx]
                    # EMA: blend new frame probs into running track estimate
                    if track.team_probs_ema is None:
                        track.team_probs_ema = probs_new.copy()
                    else:
                        track.team_probs_ema = 0.7 * track.team_probs_ema + 0.3 * probs_new
                    # Store smoothed probs so identity assignment is stable
                    obs.team_probs = track.team_probs_ema.copy()


def detect_and_project(
    video_path: str,
    detector,
    pitch_kp_detector,
    model_type: str,
    max_duration: Optional[float] = None,
    homography_interval: int = 1,
    device: Optional[torch.device] = None,
    reid_embedder=None,
    reid_interval: int = 5,
    siglip_embedder=None,
) -> TrajectoryStore:
    """
    Pass 1: Detection, tracking, and projection.

    Args:
        video_path: Input video path.
        detector: Detection model.
        pitch_kp_detector: Pitch keypoint detector (``ViTPoseKeypointDetector``
            or ``YOLOPoseKeypointDetector``). When None, homography estimation
            is skipped and only fallback projection is used.
        model_type: "yolo", "fcnn", or "rtdetr".
        max_duration: Maximum duration in seconds.
        homography_interval: Frames between homography updates.
        device: Torch device.
        reid_embedder: Optional ``DINOv2ReIDEmbedder`` instance. When provided,
            crops are extracted for active tracks every ``reid_interval`` frames
            and stored in ``obs.reid_embedding``. Also passed to
            ``ByteTracker.update_with_embeddings`` for appearance-guided MOT.
        reid_interval: Frames between ReID embedding updates (default 5).

    Returns:
        TrajectoryStore with all observations.
    """
    print("=" * 60)
    print("PASS 1: Detection + Tracking + Projection")
    print("=" * 60)

    homography = HomographyEstimator(
        min_correspondences=6,
        confidence_threshold=0.3,
        visibility_threshold=0.3,
    )

    # Non-YOLO models use ByteTracker (YOLO has built-in tracking via botsort)
    tracker = (
        ByteTracker(track_thresh=0.3, track_buffer=30, match_thresh=0.8) if model_type in ("fcnn", "rtdetr") else None
    )
    # Feet-projection capability: only ViTPoseKeypointDetector has detect_player_pose_batch
    _can_feet_project = pitch_kp_detector is not None and hasattr(pitch_kp_detector, "detect_player_pose_batch")

    # Special-case MLX Sam3 video predictor: it drives frame iteration itself
    if model_type == "sam3_mlx":
        with VideoReader(video_path, max_duration=max_duration) as reader:
            meta = reader.metadata
            store = TrajectoryStore(fps=meta.fps)
            progress = ProgressTracker(reader.max_frames, log_interval=100)
            # detector in this branch is actually Sam3VideoPredictorMLX
            _detect_and_project_sam3_mlx(
                video_path=video_path,
                predictor=detector,
                homography=homography,
                store=store,
                progress=progress,
            )
            return store

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata
        store = TrajectoryStore(fps=meta.fps)
        progress = ProgressTracker(reader.max_frames, log_interval=100)

        for frame_idx, frame_bgr in enumerate(reader):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Homography update — Kalman prediction covers up to 30-frame gaps automatically
            if frame_idx % homography_interval == 0 and pitch_kp_detector is not None:
                kps, conf = pitch_kp_detector.detect(frame_bgr)
                ok = homography.estimate(kps, conf, conf, frame_bgr.shape[:2])
                if ok and homography.H_inv is not None:
                    store.frame_homographies[frame_idx] = homography.H_inv.copy()

            # Per-frame feet keypoints cache (filled every reid_interval frames)
            _feet_cache: Dict[int, Tuple[float, float]] = {}

            # Run detection
            if model_type == "yolo":
                results = detector.track(
                    frame_bgr,
                    persist=True,
                    tracker="botsort.yaml",
                    verbose=False,
                    classes=[0],
                    conf=0.25,
                )

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    # Feet projection via ViTPose (batched, every reid_interval frames)
                    if _can_feet_project and frame_idx % reid_interval == 0 and len(boxes) > 0:
                        crops = [_crop_box(frame_rgb, b.tolist()) for b in boxes]
                        feet_raw = pitch_kp_detector.detect_player_pose_batch(crops)
                        for i, (box, tid) in enumerate(zip(boxes, track_ids)):
                            _feet_cache[tid] = (float(feet_raw[i, 0] + box[0]), float(feet_raw[i, 1] + box[1]))

                    for box, track_id in zip(boxes, track_ids):
                        pitch_pos = homography.project_player_to_pitch(box.tolist(), feet_uv=_feet_cache.get(track_id))
                        store.add_observation(
                            track_id=track_id,
                            frame_idx=frame_idx,
                            box=box,
                            pitch_pos=pitch_pos,
                        )

            elif model_type == "fcnn":
                import torchvision

                tensor = torchvision.transforms.functional.to_tensor(frame_rgb).to(device)
                with torch.no_grad():
                    predictions = detector([tensor])

                pred = predictions[0]
                boxes_np = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                valid_detections = [b for b, s in zip(boxes_np, scores) if s > 0.3]

                # Appearance-guided tracking: prefer ReID, fall back to SigLIP
                _active_embedder = reid_embedder if reid_embedder is not None else siglip_embedder
                if _active_embedder is not None and frame_idx % reid_interval == 0 and valid_detections:
                    det_crops = [_crop_box(frame_rgb, b) for b in valid_detections]
                    det_embeddings = _active_embedder.embed(det_crops)
                    active_tracks = tracker.update_with_embeddings(valid_detections, det_embeddings)
                    # Feet projection (same crops, shared compute)
                    if _can_feet_project:
                        feet_raw = pitch_kp_detector.detect_player_pose_batch(det_crops)
                        for i, det_box in enumerate(valid_detections):
                            _feet_cache[i] = (float(feet_raw[i, 0] + det_box[0]), float(feet_raw[i, 1] + det_box[1]))
                else:
                    det_embeddings = None
                    active_tracks = tracker.update(valid_detections)

                for track_id, box in active_tracks:
                    # Match detection index for feet lookup
                    best_idx, best_iou = -1, 0.0
                    for di, db in enumerate(valid_detections):
                        iou = _box_iou(box, db)
                        if iou > best_iou:
                            best_iou, best_idx = iou, di
                    feet_uv = _feet_cache.get(best_idx) if best_idx >= 0 else None
                    pitch_pos = homography.project_player_to_pitch(box.tolist(), feet_uv=feet_uv)
                    store.add_observation(track_id=track_id, frame_idx=frame_idx, box=box, pitch_pos=pitch_pos)

                if det_embeddings is not None and active_tracks:
                    _attach_reid_embeddings(
                        store, frame_idx, active_tracks, valid_detections, det_embeddings, reid_embedder
                    )

            elif model_type in ("rtdetr", "rfdetr"):
                valid_detections = [list(d.bbox) for d in detector.detect(frame_bgr)]

                _active_embedder = reid_embedder if reid_embedder is not None else siglip_embedder
                if _active_embedder is not None and frame_idx % reid_interval == 0 and valid_detections:
                    det_crops = [_crop_box(frame_rgb, b) for b in valid_detections]
                    det_embeddings = _active_embedder.embed(det_crops)
                    active_tracks = tracker.update_with_embeddings(valid_detections, det_embeddings)
                    if _can_feet_project:
                        feet_raw = pitch_kp_detector.detect_player_pose_batch(det_crops)
                        for i, det_box in enumerate(valid_detections):
                            _feet_cache[i] = (float(feet_raw[i, 0] + det_box[0]), float(feet_raw[i, 1] + det_box[1]))
                else:
                    det_embeddings = None
                    active_tracks = tracker.update(valid_detections)

                for track_id, box in active_tracks:
                    best_idx, best_iou = -1, 0.0
                    for di, db in enumerate(valid_detections):
                        iou = _box_iou(box, db)
                        if iou > best_iou:
                            best_iou, best_idx = iou, di
                    feet_uv = _feet_cache.get(best_idx) if best_idx >= 0 else None
                    pitch_pos = homography.project_player_to_pitch(box.tolist(), feet_uv=feet_uv)
                    store.add_observation(track_id=track_id, frame_idx=frame_idx, box=box, pitch_pos=pitch_pos)

                if det_embeddings is not None and active_tracks:
                    _attach_reid_embeddings(
                        store, frame_idx, active_tracks, valid_detections, det_embeddings, reid_embedder
                    )

            progress.update()
            if progress.should_log():
                print(progress.status())

        store.total_frames = frame_idx + 1

    print(f"Complete: {len(store.tracks)} tracks, {store.total_frames} frames")
    return store


def relink_tracks(
    store: TrajectoryStore,
    similarity_threshold: float = 0.7,
    max_gap_frames: int = 150,
    max_distance_m: float = 15.0,
) -> int:
    """
    Pass 1.5: Merge fragmented track IDs using ReID cosine similarity.

    Soccer tracks fragment frequently due to brief occlusions — the MOT tracker
    assigns a new ID on re-detection.  This post-hoc pass merges pairs (A, B)
    where B starts shortly after A ends, they are spatially close at the junction,
    and their mean ReID embeddings are similar.  Merges are applied greedily in
    temporal order and union-find handles transitive chains.

    Args:
        store: TrajectoryStore to modify in-place.
        similarity_threshold: Minimum cosine similarity to merge two tracks.
        max_gap_frames: Maximum frame gap between track end and next start.
        max_distance_m: Maximum pitch-space distance at the junction (meters).

    Returns:
        Number of track merges performed.
    """
    print("=" * 60)
    print("PASS 1.5: Track Re-linking")
    print("=" * 60)

    # Build per-track summary: (first_frame, last_frame, mean_embed, last_pos, first_pos)
    track_info: Dict[int, tuple] = {}
    for tid, track in store.tracks.items():
        if not track.observations:
            continue
        obs_sorted = sorted(track.observations, key=lambda o: o.frame_idx)

        embeddings = [o.reid_embedding for o in obs_sorted if o.reid_embedding is not None]
        if not embeddings:
            continue  # skip tracks without ReID data

        mean_embed = np.mean(embeddings, axis=0).astype(np.float64)
        norm = np.linalg.norm(mean_embed)
        mean_embed /= norm + 1e-8

        last_pos = next((o.pitch_pos for o in reversed(obs_sorted) if o.pitch_pos is not None), None)
        first_pos = next((o.pitch_pos for o in obs_sorted if o.pitch_pos is not None), None)

        track_info[tid] = (
            obs_sorted[0].frame_idx,  # first_frame
            obs_sorted[-1].frame_idx,  # last_frame
            mean_embed,
            last_pos,
            first_pos,
        )

    # Union-find
    parent: Dict[int, int] = {tid: tid for tid in track_info}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(b)] = find(a)

    # Sort by last_frame so we always attach B onto A in temporal order
    tids_sorted = sorted(track_info, key=lambda t: track_info[t][1])
    merges = 0

    for i, tid_a in enumerate(tids_sorted):
        first_a, last_a, embed_a, _, last_pos_a = track_info[tid_a]

        for tid_b in tids_sorted[i + 1 :]:
            first_b, last_b, embed_b, first_pos_b, _ = track_info[tid_b]

            # B must start after A ends
            gap = first_b - last_a
            if gap < 0 or gap > max_gap_frames:
                continue

            # Already in the same component
            if find(tid_a) == find(tid_b):
                continue

            # Spatial gate
            if last_pos_a is not None and first_pos_b is not None:
                dist = float(np.hypot(last_pos_a[0] - first_pos_b[0], last_pos_a[1] - first_pos_b[1]))
                if dist > max_distance_m:
                    continue

            # ReID gate
            cos_sim = float(np.dot(embed_a, embed_b))
            if cos_sim < similarity_threshold:
                continue

            union(tid_a, tid_b)
            merges += 1

    # Apply merges: re-map all observations to the root track_id
    if merges > 0:
        # Group tracks by root
        from collections import defaultdict as _dd

        root_groups: Dict[int, List[int]] = _dd(list)
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

    print(f"Complete: {merges} track merges performed ({len(store.tracks)} tracks remaining)")
    return merges


def smooth_trajectories(store: TrajectoryStore) -> int:
    """
    Pass 2: Trajectory smoothing.

    Args:
        store: TrajectoryStore with raw observations.

    Returns:
        Number of smoothed trajectories.
    """
    print("=" * 60)
    print("PASS 2: Trajectory Smoothing")
    print("=" * 60)

    smoother = TrajectorySmoother(
        fps=store.fps,
        max_speed_ms=12.0,
        smooth_sigma=7.0,
    )

    count = smoother.smooth_all(store, min_frames=10)

    print(f"Complete: Smoothed {count} trajectories")
    return count


def assign_identities(
    store: TrajectoryStore,
    reid_embedder=None,
    siglip_embedder=None,
) -> Tuple[Dict, PitchSlotManager]:
    """
    Pass 3: Identity assignment and team classification.

    Args:
        store: TrajectoryStore with smoothed trajectories.
        reid_embedder: Optional ``DINOv2ReIDEmbedder`` for embedding-based
            team clustering.
        siglip_embedder: Optional ``SigLIPTeamEmbedder`` used as zero-shot
            fallback when ``reid_embedder`` is None.

    Returns:
        (assignments, slot_manager) tuple.
    """
    print("=" * 60)
    print("PASS 3: Identity Assignment")
    print("=" * 60)

    assigner = IdentityAssigner(fps=store.fps, embedder=reid_embedder, siglip_embedder=siglip_embedder, debug=True)
    assignments = assigner.assign_roles(store)

    slot_manager = PitchSlotManager(fps=store.fps, debug=True)
    slot_manager.initialize_from_assignments(store, assignments)
    slot_manager.build_all_frame_positions(store, store.total_frames)

    print(f"Complete: Assigned {len(assignments)} identities")
    return assignments, slot_manager


def render_visualization(
    video_path: str,
    store: TrajectoryStore,
    assignments: Dict,
    slot_manager: PitchSlotManager,
    max_duration: Optional[float] = None,
    draw_overlay: bool = True,
    draw_dominance: bool = True,
) -> str:
    """
    Pass 4: Render output visualization.

    Args:
        video_path: Input video path.
        store: TrajectoryStore with all data.
        assignments: Identity assignments.
        slot_manager: Pitch slot manager.
        max_duration: Maximum duration in seconds.
        draw_overlay: Draw pitch lines on video.
        draw_dominance: Draw space control heatmap.

    Returns:
        Path to output video.
    """
    print("=" * 60)
    print("PASS 4: Visualization")
    print("=" * 60)

    pitch_viz = PitchVisualizer()

    # Build frame observations lookup
    frame_observations = defaultdict(list)
    for track_id, track in store.tracks.items():
        info = assignments.get(track_id, {'role': 'unknown', 'team': -1})
        slot_key = slot_manager.track_to_slot.get(track_id, None)

        for obs in track.observations:
            frame_observations[obs.frame_idx].append(
                {
                    'track_id': track_id,
                    'box': obs.box,
                    'role': info.get('role', 'unknown'),
                    'team': info.get('team', -1),
                    'slot_key': slot_key,
                }
            )

    output_path = generate_output_path(video_path, prefix="torchkick_analysis", duration=max_duration)

    current_H_inv = None
    position_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    smoothed_velocity: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

    # Physical constraints
    MAX_SPEED = 10.0
    VELOCITY_SMOOTHING = 0.3

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata

        pitch_h, pitch_w = pitch_viz.base_pitch.shape[:2]
        scale = meta.height / pitch_h
        output_w = meta.width + int(pitch_w * scale)

        progress = ProgressTracker(reader.max_frames, log_interval=100)

        with VideoWriter(output_path, meta.fps, (output_w, meta.height)) as writer:
            for frame_idx, frame_bgr in enumerate(reader):
                obs_list = frame_observations.get(frame_idx, [])

                # Update homography
                if draw_overlay:
                    for check_idx in range(frame_idx, -1, -1):
                        if check_idx in store.frame_homographies:
                            current_H_inv = store.frame_homographies[check_idx]
                            break

                # Draw pitch overlay
                if draw_overlay and current_H_inv is not None:
                    frame_bgr = _draw_pitch_overlay(frame_bgr, current_H_inv)

                # Get slot positions
                slot_positions = slot_manager.get_frame_positions(frame_idx)

                # Update position history
                for slot in slot_positions:
                    slot_key = slot['slot_key']
                    x, y = slot['position']
                    position_history[slot_key].append((x, y))
                    position_history[slot_key] = position_history[slot_key][-60:]

                # Draw bounding boxes
                for obs in obs_list:
                    box = obs['box']
                    role = obs['role']
                    team = obs['team']
                    slot_key = obs.get('slot_key')

                    x1, y1, x2, y2 = map(int, box)
                    display_id = slot_key if slot_key else f"?{obs['track_id']}"

                    if role == 'goalie':
                        color = (0, 255, 255) if team == 0 else (255, 255, 0)
                        label = f"GK:{display_id}"
                    elif role == 'referee':
                        color = (0, 0, 0)
                        label = "REF"
                    elif role == 'linesman':
                        color = (128, 128, 128)
                        label = "LN"
                    elif team == 0:
                        color = (0, 0, 255)
                        label = display_id
                    elif team == 1:
                        color = (255, 0, 0)
                        label = display_id
                    else:
                        color = (0, 255, 0)
                        label = display_id

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_bgr, str(label), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Build pitch positions with velocity
                pitch_positions = []
                dt = 1.0 / meta.fps

                for slot in slot_positions:
                    x, y = slot['position']
                    team = slot['team']
                    slot_id = slot['slot_id']
                    slot_key = slot['slot_key']

                    prev_vx, prev_vy = smoothed_velocity[slot_key]

                    # Compute velocity from history
                    vx, vy = 0.0, 0.0
                    if slot_key in position_history and len(position_history[slot_key]) >= 2:
                        hist = position_history[slot_key]
                        prev_x, prev_y = hist[-2] if len(hist) >= 2 else hist[-1]
                        dx = x - prev_x
                        dy = y - prev_y
                        vx = dx / dt
                        vy = dy / dt

                    # Smooth velocity
                    vx = prev_vx * (1 - VELOCITY_SMOOTHING) + vx * VELOCITY_SMOOTHING
                    vy = prev_vy * (1 - VELOCITY_SMOOTHING) + vy * VELOCITY_SMOOTHING

                    # Clamp speed
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed > MAX_SPEED:
                        vx *= MAX_SPEED / speed
                        vy *= MAX_SPEED / speed

                    smoothed_velocity[slot_key] = (vx, vy)

                    # Flip Y for visualization
                    pitch_positions.append((x, -y, vx, -vy, slot_id, team if team >= 0 else 2))

                # Draw pitch view
                if pitch_positions and draw_dominance:
                    trail_history = {}
                    for slot in slot_positions:
                        slot_key = slot['slot_key']
                        if slot_key in position_history:
                            trail_history[slot['slot_id']] = [(hx, -hy) for hx, hy in position_history[slot_key][-60:]]

                    pitch_img = pitch_viz.draw_with_trails(
                        pitch_positions,
                        trail_history,
                        trail_length=60,
                        draw_vectors=True,
                        draw_dominance=True,
                    )
                else:
                    pitch_img = (
                        pitch_viz.draw_players(
                            [
                                (x, -y, slot['slot_id'], slot['team'])
                                for slot, (x, y, _, _, _, _) in zip(slot_positions, pitch_positions)
                            ]
                        )
                        if pitch_positions
                        else pitch_viz.base_pitch.copy()
                    )

                # Combine frames
                pitch_scaled = cv2.resize(pitch_img, (int(pitch_w * scale), meta.height))
                combined = np.hstack([frame_bgr, pitch_scaled])
                writer.write(combined)

                progress.update()
                if progress.should_log():
                    print(f"Pass 4: {progress.status()}")

    print(f"Complete: Video saved to {output_path}")
    return output_path


def _draw_pitch_overlay(
    frame: np.ndarray,
    homography_H_inv: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw pitch lines on frame using homography."""
    frame_viz = frame.copy()
    h, w = frame.shape[:2]

    for class_name, points in PITCH_LINE_COORDINATES.items():
        pitch_pts = np.array([p.to_array() for _, p in points], dtype=np.float32)

        ones = np.ones((pitch_pts.shape[0], 1), dtype=np.float32)
        pts_h = np.hstack([pitch_pts, ones])

        projected = (homography_H_inv @ pts_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]

        valid_pts = []
        for pt in projected:
            x, y = int(pt[0]), int(pt[1])
            if -500 < x < w + 500 and -500 < y < h + 500:
                valid_pts.append((x, y))

        if len(valid_pts) < 2:
            continue

        if "Circle" in class_name and len(valid_pts) >= 3:
            pts = np.array(valid_pts, dtype=np.int32)
            cv2.polylines(frame_viz, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            for i in range(len(valid_pts) - 1):
                cv2.line(frame_viz, valid_pts[i], valid_pts[i + 1], color, thickness)

    return frame_viz


def run_analysis(
    video_path: str,
    model_path: Optional[str] = None,
    model_type: str = "rfdetr",
    duration: Optional[float] = None,
    homography_interval: int = 1,
    draw_overlay: bool = True,
    draw_dominance: bool = True,
    device: Optional[str] = None,
    reid_weights: Optional[str] = None,
    reid_interval: int = 5,
    pitch_weights: Optional[str] = None,
    pitch_detector_type: str = "yolo",
) -> str:
    """
    Run complete match analysis pipeline.

    This is the main entry point for the inference pipeline, combining
    detection, tracking, projection, identity assignment, and visualization.

    Args:
        video_path: Path to input video.
        model_path: Path to detection model weights.
            If None, uses pretrained weights for the selected model_type.
        model_type: Detection backend. Recommended: "rfdetr" (DINOv2 backbone,
            AP50 73.6, ~5ms/frame — default). Alternatives: "yolo" (fastest),
            "rtdetr" (~60 AP50), "sam3_mlx" (local dev/CPU).
        duration: Maximum duration in seconds.
        homography_interval: Frames between homography updates.
        draw_overlay: Draw pitch lines on video.
        draw_dominance: Draw space control heatmap.
        device: Device string ("cuda" or "cpu").
        reid_weights: Optional path to distilled ReID student checkpoint.
            Enables appearance-guided tracking and DINOv2-based team
            classification. Without this, SigLIP zero-shot clustering is used.
        reid_interval: Frames between ReID embedding extraction (default 5).

    Returns:
        Path to output video.

    Example:
        >>> output = run_analysis(
        ...     video_path="match.mp4",
        ...     model_type="yolo",
        ...     duration=60.0,
        ... )
        >>> print(f"Saved to: {output}")
    """
    # Set default model paths
    if model_path is None:
        if model_type == "yolo":
            model_path = "yolo11n.pt"
        elif model_type == "fcnn":
            model_path = "models/player/fcnn/fcnn_player_tracker.pth"
        elif model_type == "rtdetr":
            model_path = "models/player/rtdetr/rtdetr_player_tracker.pth"

    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {dev}")

    # Load models
    detector, pitch_kp_detector, reid_embedder = load_models(
        dev,
        model_path,
        model_type,
        reid_weights=reid_weights,
        pitch_weights=pitch_weights,
        pitch_detector_type=pitch_detector_type,
    )

    # Zero-shot SigLIP team embedder — used when no ReID weights are provided
    siglip_embedder = None
    if reid_embedder is None:
        try:
            from torchkick.models.reid import SigLIPTeamEmbedder

            siglip_embedder = SigLIPTeamEmbedder(device=str(dev))
            print("SigLIP zero-shot team embedder loaded (no --reid-weights provided)")
        except Exception as e:
            print(f"[info] SigLIP unavailable: {e}. Team classification disabled.")

    # Pass 1: Detection and projection
    store = detect_and_project(
        video_path,
        detector,
        pitch_kp_detector,
        model_type,
        max_duration=duration,
        homography_interval=homography_interval,
        device=dev,
        reid_embedder=reid_embedder,
        reid_interval=reid_interval,
        siglip_embedder=siglip_embedder,
    )

    # Pass 1.5: Re-link fragmented tracks via ReID similarity
    relink_tracks(store)

    # Pass 2: Smooth trajectories
    smooth_trajectories(store)

    # Pass 3: Identity assignment
    assignments, slot_manager = assign_identities(store, reid_embedder=reid_embedder, siglip_embedder=siglip_embedder)

    # Pass 4: Visualization
    output_path = render_visualization(
        video_path,
        store,
        assignments,
        slot_manager,
        max_duration=duration,
        draw_overlay=draw_overlay,
        draw_dominance=draw_dominance,
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run match analysis")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--model", type=str, default=None, help="Detection model path")
    parser.add_argument("--model-type", type=str, default="yolo", choices=["yolo", "fcnn"])
    parser.add_argument("--duration", type=float, default=None, help="Max duration in seconds")
    parser.add_argument("--homography-interval", type=int, default=1)
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--dominance", action="store_true")

    args = parser.parse_args()

    run_analysis(
        video_path=args.video,
        model_path=args.model,
        model_type=args.model_type,
        duration=args.duration,
        homography_interval=args.homography_interval,
        draw_overlay=args.overlay,
        draw_dominance=args.dominance,
    )
