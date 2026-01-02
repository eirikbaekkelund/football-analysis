import argparse
import os
import cv2
import torch
import numpy as np
from collections import defaultdict

from match_state.offline_tracker import (
    TrajectoryStore,
    TrajectorySmootherVelocityConstrained,
    IdentityAssigner,
    FallbackProjector,
    PitchSlotManager,
)
from match_state.player_tracker import get_jersey_color_feature  # Use fast version, not K-Means
from match_state.pitch_homography import (
    HomographyEstimator,
    PitchVisualizer,
    PITCH_LINE_COORDINATES,
)
from models.pitch.wrapper import PnLCalibWrapper
from utils.video import VideoReader, VideoWriter, ProgressTracker, generate_output_path


def load_models(device: torch.device, model_path: str, model_type: str):
    """Load detection and homography models."""
    # Homography model
    base_path = os.path.join("models", "pitch")
    weights_kp = os.path.join(base_path, "weights", "SV_kp")
    weights_line = os.path.join(base_path, "weights", "SV_lines")
    config_kp = os.path.join(base_path, "config", "hrnetv2_w48.yaml")
    config_line = os.path.join(base_path, "config", "hrnetv2_w48_l.yaml")

    pnl_calib = PnLCalibWrapper(weights_kp, weights_line, config_kp, config_line, device=str(device))

    # Detection model
    if model_type == "yolo":
        from ultralytics import YOLO

        detector = YOLO(model_path)
        detector.to(device)
    elif model_type == "fcnn":
        import torchvision
        from models.player.fcnn.train_fcnn import get_player_detector_model

        detector = get_player_detector_model(num_classes=2)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        detector.load_state_dict(checkpoint)
        detector.to(device)
        detector.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return detector, pnl_calib


def run_pass_1(
    video_path: str,
    detector,
    pnl_calib,
    model_type: str,
    max_duration: float = None,
    homography_interval: int = 1,  # Run every frame for proper 2D smoothing
    device: torch.device = None,
) -> TrajectoryStore:
    """
    Pass 1: Detection + Tracking + Homography projection.
    Stores all observations in TrajectoryStore.

    Homography is run every frame so that 2D smoothing can be done
    properly in projection space (after homography).
    """
    print("=" * 60)
    print("PASS 1: Detection + Tracking + Projection (every frame)")
    print("=" * 60)

    homography = HomographyEstimator(
        min_correspondences=6,
        confidence_threshold=0.3,
        visibility_threshold=0.3,
    )

    # For FCNN, we need a tracker
    if model_type == "fcnn":
        from match_state.player_tracker import PlayerTracker

        tracker = PlayerTracker(max_age=30, iou_threshold=0.3)

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata
        store = TrajectoryStore(fps=meta.fps)
        progress = ProgressTracker(reader.max_frames, log_interval=100)

        # Fallback projector for when homography fails
        fallback_proj = FallbackProjector(
            image_width=meta.width,
            image_height=meta.height,
        )

        homography_ok = False

        for frame_idx, frame_bgr in enumerate(reader):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Update homography periodically
            if frame_idx % homography_interval == 0:
                P = pnl_calib.process_frame(frame_bgr)
                if P is not None:
                    H_inv = P[:, [0, 1, 3]]
                    if H_inv[2, 2] != 0:
                        H_inv /= H_inv[2, 2]
                    homography.H_inv = H_inv
                    try:
                        homography.H = np.linalg.inv(H_inv)
                        homography_ok = True
                    except np.linalg.LinAlgError:
                        homography_ok = False
                else:
                    homography_ok = False

            # Detection + Tracking
            if model_type == "yolo":
                results = detector.track(
                    frame_bgr,
                    persist=True,
                    tracker="botsort.yaml",
                    verbose=False,
                    classes=[0],
                )

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        # Project to pitch - try homography first, then fallback
                        pitch_pos = None
                        if homography_ok:
                            pitch_pos = homography.project_player_to_pitch(box.tolist())
                            # Add reference for fallback calibration
                            if pitch_pos is not None:
                                fallback_proj.add_reference(box, pitch_pos)
                        elif fallback_proj.is_calibrated():
                            # Use bbox-based fallback when homography fails
                            pitch_pos = fallback_proj.project(box)

                        # Extract color
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
                    if score > 0.7:
                        valid_detections.append(np.append(box, score))

                active_tracks = tracker.update(valid_detections, frame_rgb)

                for track in active_tracks:
                    track_id = track['id']
                    box = np.array(track['box'])

                    # Project to pitch - try homography first, then fallback
                    pitch_pos = None
                    if homography_ok:
                        pitch_pos = homography.project_player_to_pitch(box.tolist())
                        # Add reference for fallback calibration
                        if pitch_pos is not None:
                            fallback_proj.add_reference(box, pitch_pos)
                    elif fallback_proj.is_calibrated():
                        # Use bbox-based fallback when homography fails
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
                print(f"Pass 1: {progress.status()}")

        store.total_frames = frame_idx + 1

    print(f"Pass 1 Complete: {len(store.tracks)} tracks, {store.total_frames} frames")
    return store


def run_pass_2(store: TrajectoryStore) -> int:
    """
    Pass 2: Smooth trajectories in 2D projection space.

    Now that we have per-frame homography projections, we smooth
    the 2D pitch positions so movement is smooth relative to pitch lines.
    """
    print("=" * 60)
    print("PASS 2: 2D Trajectory Smoothing (in projection space)")
    print("=" * 60)

    smoother = TrajectorySmootherVelocityConstrained(
        fps=store.fps,
        max_speed_ms=12.0,  # Generous for sprints
        smooth_sigma=7.0,  # Increased for more smoothing
    )

    count = smoother.smooth_all(store, min_frames=10)

    print(f"Pass 2 Complete: Smoothed {count} trajectories")
    return count


def run_pass_3(store: TrajectoryStore) -> dict:
    """
    Pass 3: Identity assignment with spatial priors and majority voting.

    Uses per-frame color voting with majority across track lifetime.
    Enforces 22 players (11v11) + 1 referee constraint for 2D pitch view.
    """
    print("=" * 60)
    print("PASS 3: Identity Assignment + Fixed Slot Mapping")
    print("=" * 60)

    assigner = IdentityAssigner(fps=store.fps, debug=True)
    assignments = assigner.assign_roles(store)

    # Create PitchSlotManager for fixed 11v11 + referee
    slot_manager = PitchSlotManager(fps=store.fps, debug=True)
    slot_manager.initialize_from_assignments(store, assignments)

    # Build frame-by-frame positions for all slots
    # Slots stay at last known position until re-observed
    slot_manager.build_all_frame_positions(store, store.total_frames)

    print(f"Pass 3 Complete: Assigned {len(assignments)} identities")
    return assignments, slot_manager


def draw_pitch_overlay_on_frame(
    frame: np.ndarray,
    homography_H_inv: np.ndarray,
    color: tuple = (0, 255, 255),  # Yellow
    thickness: int = 2,
) -> np.ndarray:
    """Project pitch lines back onto the video frame."""
    if homography_H_inv is None:
        return frame

    frame_viz = frame.copy()
    h, w = frame.shape[:2]

    for class_name, points in PITCH_LINE_COORDINATES.items():
        pitch_pts = np.array([p.to_array() for _, p in points], dtype=np.float32)

        # Add homogeneous coordinate
        ones = np.ones((pitch_pts.shape[0], 1), dtype=np.float32)
        pts_h = np.hstack([pitch_pts, ones])

        # Project to image
        projected = (homography_H_inv @ pts_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]

        # Filter valid points
        valid_pts = []
        for pt in projected:
            x, y = int(pt[0]), int(pt[1])
            if -500 < x < w + 500 and -500 < y < h + 500:
                valid_pts.append((x, y))

        if len(valid_pts) < 2:
            continue

        # Draw
        if "Circle" in class_name and len(valid_pts) >= 3:
            pts = np.array(valid_pts, dtype=np.int32)
            cv2.polylines(frame_viz, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            for i in range(len(valid_pts) - 1):
                cv2.line(frame_viz, valid_pts[i], valid_pts[i + 1], color, thickness)

    return frame_viz


def run_pass_4(
    video_path: str,
    store: TrajectoryStore,
    assignments: dict,
    slot_manager: PitchSlotManager,
    max_duration: float = None,
    draw_overlay: bool = True,
    draw_dominance: bool = True,
    homography_interval: int = 5,
) -> str:
    """
    Pass 4: Visualization with fixed 11v11 slots, pitch overlay, and space control.

    Uses slot_manager for consistent player positions - slots stay at last
    known position until re-observed, ensuring we always show 11v11.
    """
    print("=" * 60)
    print("PASS 4: Visualization (fixed 11v11 slots)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load homography model for pitch overlay
    base_path = os.path.join("models", "pitch")
    weights_kp = os.path.join(base_path, "weights", "SV_kp")
    weights_line = os.path.join(base_path, "weights", "SV_lines")
    config_kp = os.path.join(base_path, "config", "hrnetv2_w48.yaml")
    config_line = os.path.join(base_path, "config", "hrnetv2_w48_l.yaml")
    pnl_calib = PnLCalibWrapper(weights_kp, weights_line, config_kp, config_line, device=str(device))

    pitch_viz = PitchVisualizer()

    # Build frame -> raw observations lookup (for video bounding boxes)
    # Include slot info so we can show merged IDs
    frame_observations = defaultdict(list)
    total_obs_count = 0
    for track_id, track in store.tracks.items():
        info = assignments.get(track_id, {'role': 'unknown', 'team': -1})

        # Get slot key if this track is assigned to a slot
        slot_key = slot_manager.track_to_slot.get(track_id, None)

        for obs in track.observations:
            total_obs_count += 1
            frame_observations[obs.frame_idx].append(
                {
                    'track_id': track_id,
                    'box': obs.box,
                    'role': info.get('role', 'unknown'),
                    'team': info.get('team', -1),
                    'slot_key': slot_key,  # Will be None if not assigned to a slot
                }
            )

    # Diagnostic: count observations
    frames_with_obs = len(frame_observations)
    avg_obs_per_frame = total_obs_count / frames_with_obs if frames_with_obs > 0 else 0
    print(
        f"[Diagnostic] Total observations: {total_obs_count}, Frames with obs: {frames_with_obs}, Avg per frame: {avg_obs_per_frame:.1f}"
    )

    output_path = generate_output_path(video_path, prefix="offline_match", duration=max_duration)

    # Current homography for overlay
    current_H_inv = None

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata

        # Output: video + pitch side by side
        pitch_h, pitch_w = pitch_viz.base_pitch.shape[:2]
        scale = meta.height / pitch_h
        output_w = meta.width + int(pitch_w * scale)

        progress = ProgressTracker(reader.max_frames, log_interval=100)

        with VideoWriter(output_path, meta.fps, (output_w, meta.height)) as writer:
            for frame_idx, frame_bgr in enumerate(reader):
                obs_list = frame_observations.get(frame_idx, [])

                # Update homography for overlay
                if draw_overlay and frame_idx % homography_interval == 0:
                    P = pnl_calib.process_frame(frame_bgr)
                    if P is not None:
                        H_inv = P[:, [0, 1, 3]]
                        if H_inv[2, 2] != 0:
                            H_inv /= H_inv[2, 2]
                        current_H_inv = H_inv

                # Draw pitch overlay on frame
                if draw_overlay and current_H_inv is not None:
                    frame_bgr = draw_pitch_overlay_on_frame(frame_bgr, current_H_inv)

                # Draw on video frame
                for obs in obs_list:
                    box = obs['box']
                    role = obs['role']
                    team = obs['team']
                    tid = obs['track_id']
                    slot_key = obs.get('slot_key')  # May be None

                    x1, y1, x2, y2 = map(int, box)

                    # Use slot_key for label if available (shows merged identity)
                    display_id = slot_key if slot_key else f"?{tid}"

                    # Color by role/team
                    if role == 'goalie':
                        color = (0, 255, 255) if team == 0 else (255, 255, 0)  # Yellow variants
                        label = f"GK:{display_id}"
                    elif role == 'referee':
                        color = (0, 0, 0)  # Black
                        label = "REF"
                    elif role == 'linesman':
                        color = (128, 128, 128)  # Gray
                        label = "LN"
                    elif team == 0:
                        color = (0, 0, 255)  # Red (BGR)
                        label = display_id
                    elif team == 1:
                        color = (255, 0, 0)  # Blue (BGR)
                        label = display_id
                    else:
                        color = (0, 255, 0)  # Green for unknown
                        label = display_id

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Get slot positions for this frame (fixed 11v11 + referee)
                slot_positions = slot_manager.get_frame_positions(frame_idx)

                # Build pitch positions for space control from slots
                pitch_positions = []
                for slot in slot_positions:
                    x, y = slot['position']
                    team = slot['team']
                    slot_id = slot['slot_id']
                    # Format: (x, y, vx, vy, track_id, team)
                    # Flip Y for proper orientation
                    pitch_positions.append((x, -y, 0.0, 0.0, slot_id, team if team >= 0 else 2))  # 2 = referee

                # Draw pitch visualization with space control
                if pitch_positions and draw_dominance:
                    pitch_img = pitch_viz.draw_with_trails(
                        pitch_positions,
                        {},  # No history for now
                        draw_vectors=False,
                        draw_dominance=True,
                    )
                else:
                    pitch_img = pitch_viz.base_pitch.copy()

                    # Draw players manually from slots
                    for slot in slot_positions:
                        x, y = slot['position']
                        team = slot['team']
                        is_goalie = slot['is_goalie']
                        slot_id = slot['slot_id']

                        # Flip Y for proper orientation
                        px, py = pitch_viz._pitch_to_pixel(x, -y)

                        if is_goalie:
                            color = (0, 255, 255) if team == 0 else (255, 255, 0)
                        elif team == -1:  # Referee
                            color = (0, 0, 0)
                        elif team == 0:
                            color = (0, 0, 255)
                        elif team == 1:
                            color = (255, 0, 0)
                        else:
                            color = (0, 255, 0)

                        cv2.circle(pitch_img, (px, py), 8, color, -1)
                        cv2.circle(pitch_img, (px, py), 8, (0, 0, 0), 1)
                        # Show slot ID
                        cv2.putText(
                            pitch_img,
                            str(slot_id),
                            (px - 4, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 255, 255),
                            1,
                        )

                # Scale pitch to match video height
                pitch_scaled = cv2.resize(pitch_img, (int(pitch_w * scale), meta.height))

                # Combine
                combined = np.hstack([frame_bgr, pitch_scaled])
                writer.write(combined)

                progress.update()
                if progress.should_log():
                    print(f"Pass 4: {progress.status()}")

    print(f"Pass 4 Complete: Video saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Unified Offline Match Analysis")
    parser.add_argument("--video", type=str, default="videos/go_baerum.mp4", help="Input video path")
    parser.add_argument("--model", type=str, default=None, help="Detection model path (auto-detected if not provided)")
    parser.add_argument("--model_type", type=str, default="yolo", choices=["yolo", "fcnn"])
    parser.add_argument("--duration", type=float, default=None, help="Max duration in seconds")
    parser.add_argument(
        "--homography_interval",
        type=int,
        default=1,
        help="Frames between homography updates (1=every frame for smoothest 2D)",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Disable pitch line overlay on video")
    parser.add_argument("--no-dominance", action="store_true", help="Disable space control heatmap")
    args = parser.parse_args()

    # Set default model path based on model type
    if args.model is None:
        if args.model_type == "yolo":
            args.model = "yolo11n.pt"
        elif args.model_type == "fcnn":
            args.model = "models/player/fcnn/fcnn_player_tracker.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    detector, pnl_calib = load_models(device, args.model, args.model_type)

    # Pass 1: Detection + Tracking + Projection
    store = run_pass_1(
        args.video,
        detector,
        pnl_calib,
        args.model_type,
        max_duration=args.duration,
        homography_interval=args.homography_interval,
        device=device,
    )

    # Pass 2: Trajectory Smoothing (light smoothing)
    run_pass_2(store)

    # Pass 3: Identity Assignment + Fixed Slot Mapping
    assignments, slot_manager = run_pass_3(store)

    # Pass 4: Visualization with fixed 11v11 slots
    output_path = run_pass_4(
        args.video,
        store,
        assignments,
        slot_manager,
        max_duration=args.duration,
        draw_overlay=not args.no_overlay,
        draw_dominance=not args.no_dominance,
        homography_interval=args.homography_interval,
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
