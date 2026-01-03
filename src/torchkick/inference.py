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
        --no-dominance
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
    SimpleIoUTracker,
    IdentityAssigner,
    PitchSlotManager,
    FallbackProjector,
    HomographyEstimator,
    PitchVisualizer,
    PITCH_LINE_COORDINATES,
    get_jersey_color_feature,
)
from torchkick.utils import (
    VideoReader,
    VideoWriter,
    ProgressTracker,
    generate_output_path,
)


def load_models(
    device: torch.device,
    model_path: str,
    model_type: str,
) -> Tuple:
    """
    Load detection and homography models.

    Args:
        device: Torch device.
        model_path: Path to detection model weights.
        model_type: "yolo", "fcnn", or "rtdetr".

    Returns:
        (detector, pnl_calib) tuple.
    """
    # Load pitch line calibration model
    from torchkick.models import PitchCalibrator, PitchLineDetector

    base_path = Path(__file__).parent.parent.parent.parent / "models" / "pitch"
    if not base_path.exists():
        base_path = Path("models/pitch")

    weights_kp = base_path / "weights" / "SV_kp"
    weights_line = base_path / "weights" / "SV_lines"
    config_kp = base_path / "config" / "hrnetv2_w48.yaml"
    config_line = base_path / "config" / "hrnetv2_w48_l.yaml"

    pitch_detector = PitchLineDetector(
        weights_kp=str(weights_kp),
        weights_lines=str(weights_line),
        config_kp=str(config_kp),
        config_lines=str(config_line),
        device=str(device),
    )
    pnl_calib = PitchCalibrator(pitch_detector)

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
        from torchkick.training import get_rtdetr_model

        detector = get_rtdetr_model()
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            detector.load_state_dict(checkpoint['model_state_dict'])
        else:
            detector.load_state_dict(checkpoint)
        detector.to(device)
        detector.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'yolo', 'fcnn', or 'rtdetr'.")

    return detector, pnl_calib


def detect_and_project(
    video_path: str,
    detector,
    pnl_calib,
    model_type: str,
    max_duration: Optional[float] = None,
    homography_interval: int = 1,
    device: Optional[torch.device] = None,
) -> TrajectoryStore:
    """
    Pass 1: Detection, tracking, and projection.

    Args:
        video_path: Input video path.
        detector: Detection model.
        pnl_calib: Pitch calibration wrapper.
        model_type: "yolo", "fcnn", or "rtdetr".
        max_duration: Maximum duration in seconds.
        homography_interval: Frames between homography updates.
        device: Torch device.

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

    # Non-YOLO models need IoU tracker (YOLO has built-in tracking)
    tracker = SimpleIoUTracker(max_age=30, iou_threshold=0.3) if model_type in ("fcnn", "rtdetr") else None

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata
        store = TrajectoryStore(fps=meta.fps)
        progress = ProgressTracker(reader.max_frames, log_interval=100)

        fallback_proj = FallbackProjector(
            image_width=meta.width,
            image_height=meta.height,
        )

        homography_ok = False

        for frame_idx, frame_bgr in enumerate(reader):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Homography update
            if frame_idx % homography_interval == 0:
                P = pnl_calib.process_frame(frame_bgr)
                if P is not None:
                    H_inv = P[:, [0, 1, 3]]
                    if H_inv[2, 2] != 0:
                        H_inv /= H_inv[2, 2]
                    homography.H_inv = H_inv
                    store.frame_homographies[frame_idx] = H_inv.copy()
                    try:
                        homography.H = np.linalg.inv(H_inv)
                        homography_ok = True
                    except np.linalg.LinAlgError:
                        homography_ok = False
                else:
                    homography_ok = False

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

                    for box, track_id in zip(boxes, track_ids):
                        pitch_pos = None
                        if homography_ok:
                            pitch_pos = homography.project_player_to_pitch(box.tolist())
                            if pitch_pos is not None:
                                fallback_proj.add_reference(box, pitch_pos)
                        elif fallback_proj.is_calibrated():
                            pitch_pos = fallback_proj.project(box)

                        color_feat = get_jersey_color_feature(frame_rgb, box)
                        if np.linalg.norm(color_feat) == 0:
                            color_feat = None

                        store.add_observation(
                            track_id=track_id,
                            frame_idx=frame_idx,
                            box=box,
                            pitch_pos=pitch_pos,
                            color_feature=color_feat,
                        )

            elif model_type == "fcnn":
                import torchvision

                tensor = torchvision.transforms.functional.to_tensor(frame_rgb).to(device)
                with torch.no_grad():
                    predictions = detector([tensor])

                pred = predictions[0]
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()

                valid_detections = []
                for box, score in zip(boxes, scores):
                    if score > 0.3:
                        valid_detections.append(box)

                active_tracks = tracker.update(valid_detections)

                for track_id, box in active_tracks:
                    pitch_pos = None
                    if homography_ok:
                        pitch_pos = homography.project_player_to_pitch(box.tolist())
                        if pitch_pos is not None:
                            fallback_proj.add_reference(box, pitch_pos)
                    elif fallback_proj.is_calibrated():
                        pitch_pos = fallback_proj.project(box)

                    color_feat = get_jersey_color_feature(frame_rgb, box)
                    if np.linalg.norm(color_feat) == 0:
                        color_feat = None

                    store.add_observation(
                        track_id=track_id,
                        frame_idx=frame_idx,
                        box=box,
                        pitch_pos=pitch_pos,
                        color_feature=color_feat,
                    )

            elif model_type == "rtdetr":
                import torchvision
                from torchvision.transforms import functional as F

                # Prepare input for RT-DETR
                tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = detector(tensor)

                # RT-DETR outputs: logits and pred_boxes
                logits = outputs.logits[0]  # [num_queries, num_classes]
                pred_boxes = outputs.pred_boxes[0]  # [num_queries, 4] in cxcywh format

                # Convert logits to probabilities
                probs = torch.softmax(logits, dim=-1)
                scores, labels = probs.max(-1)

                # Filter by confidence and player class (class 0)
                valid_mask = (scores > 0.3) & (labels == 0)
                valid_boxes = pred_boxes[valid_mask].cpu().numpy()

                # Convert from cxcywh (normalized) to xyxy (pixel)
                h, w = frame_rgb.shape[:2]
                valid_detections = []
                for box in valid_boxes:
                    cx, cy, bw, bh = box
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    valid_detections.append([x1, y1, x2, y2])

                active_tracks = tracker.update(valid_detections)

                for track_id, box in active_tracks:
                    pitch_pos = None
                    if homography_ok:
                        pitch_pos = homography.project_player_to_pitch(box.tolist())
                        if pitch_pos is not None:
                            fallback_proj.add_reference(box, pitch_pos)
                    elif fallback_proj.is_calibrated():
                        pitch_pos = fallback_proj.project(box)

                    color_feat = get_jersey_color_feature(frame_rgb, box)
                    if np.linalg.norm(color_feat) == 0:
                        color_feat = None

                    store.add_observation(
                        track_id=track_id,
                        frame_idx=frame_idx,
                        box=box,
                        pitch_pos=pitch_pos,
                        color_feature=color_feat,
                    )

            progress.update()
            if progress.should_log():
                print(progress.status())

        store.total_frames = frame_idx + 1

    print(f"Complete: {len(store.tracks)} tracks, {store.total_frames} frames")
    return store


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


def assign_identities(store: TrajectoryStore) -> Tuple[Dict, PitchSlotManager]:
    """
    Pass 3: Identity assignment and team classification.

    Args:
        store: TrajectoryStore with smoothed trajectories.

    Returns:
        (assignments, slot_manager) tuple.
    """
    print("=" * 60)
    print("PASS 3: Identity Assignment")
    print("=" * 60)

    assigner = IdentityAssigner(fps=store.fps, debug=True)
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
    model_type: str = "fcnn",
    duration: Optional[float] = None,
    homography_interval: int = 1,
    draw_overlay: bool = True,
    draw_dominance: bool = True,
    device: Optional[str] = None,
) -> str:
    """
    Run complete match analysis pipeline.

    This is the main entry point for the inference pipeline, combining
    detection, tracking, projection, identity assignment, and visualization.

    Args:
        video_path: Path to input video.
        model_path: Path to detection model weights.
            If None, uses default for model_type.
        model_type: "yolo", "fcnn", or "rtdetr".
        duration: Maximum duration in seconds.
        homography_interval: Frames between homography updates.
        draw_overlay: Draw pitch lines on video.
        draw_dominance: Draw space control heatmap.
        device: Device string ("cuda" or "cpu").

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
    detector, pnl_calib = load_models(dev, model_path, model_type)

    # Pass 1: Detection and projection
    store = detect_and_project(
        video_path,
        detector,
        pnl_calib,
        model_type,
        max_duration=duration,
        homography_interval=homography_interval,
        device=dev,
    )

    # Pass 2: Smooth trajectories
    smooth_trajectories(store)

    # Pass 3: Identity assignment
    assignments, slot_manager = assign_identities(store)

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
    parser.add_argument("--model-type", type=str, default="fcnn", choices=["yolo", "fcnn"])
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
        draw_overlay=args.no_overlay,
        draw_dominance=args.no_dominance,
    )
