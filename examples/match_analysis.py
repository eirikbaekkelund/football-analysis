import argparse
import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict

from match_state.tracker import (
    TrajectoryStore,
    TrajectorySmootherVelocityConstrained,
    IdentityAssigner,
    FallbackProjector,
    PitchSlotManager,
    SimpleIoUTracker,
)
from match_state.bbox_features import get_jersey_color_feature
from match_state.pitch_homography import (
    HomographyEstimator,
    PitchVisualizer,
    PITCH_LINE_COORDINATES,
)
from models.pitch.wrapper import PnLCalibWrapper
from utils.video import VideoReader, VideoWriter, ProgressTracker, generate_output_path

# TODO: evaluate semantic segmentation model for field + player heatmap/homography
# as line detection can be noisy in some cases and may fail entirely
# TODO: optical flow for smoothing + short-term position interpolation
# TODO: bytetrack for better ID persistence through occlusions
# TODO: use bottom of bbox for projection instead of center
# TODO: re-id using spatial + color features for fragmented track merging
def load_models(device: torch.device, model_path: str, model_type: str):
    """Load detection and homography models."""
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
        from models.player.fcnn.train_fcnn import get_player_detector_model

        detector = get_player_detector_model(num_classes=2)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        detector.load_state_dict(checkpoint)
        detector.to(device)
        detector.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return detector, pnl_calib


def detect_and_project(
    video_path: str,
    detector,
    pnl_calib,
    model_type: str,
    max_duration: float = None,
    homography_interval: int = 1,
    device: torch.device = None,
) -> TrajectoryStore:
    print("=" * 60)
    print("Detection + Tracking + Projection")
    print("=" * 60)

    homography = HomographyEstimator(
        min_correspondences=6,
        confidence_threshold=0.3,
        visibility_threshold=0.3,
    )

    # For FCNN, we need a simple IoU tracker (no GMM, just association)
    if model_type == "fcnn":
        tracker = SimpleIoUTracker(max_age=30, iou_threshold=0.3)

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

            # homography periodic update
            if frame_idx % homography_interval == 0:
                P = pnl_calib.process_frame(frame_bgr)
                if P is not None:
                    H_inv = P[:, [0, 1, 3]]
                    if H_inv[2, 2] != 0:
                        H_inv /= H_inv[2, 2]
                    homography.H_inv = H_inv
                    store.frame_homographies[frame_idx] = H_inv.copy()  # cache for visualization
                    try:
                        homography.H = np.linalg.inv(H_inv)
                        homography_ok = True
                    except np.linalg.LinAlgError:
                        homography_ok = False
                else:
                    homography_ok = False

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

            progress.update()
            if progress.should_log():
                print({progress.status()})

        store.total_frames = frame_idx + 1

    print(f"Complete: {len(store.tracks)} tracks, {store.total_frames} frames")
    return store


def smooth_trajectories(store: TrajectoryStore) -> int:
    print("=" * 60)
    print("PASS 2: Trajectory Smoothing")
    print("=" * 60)

    smoother = TrajectorySmootherVelocityConstrained(
        fps=store.fps,
        max_speed_ms=12.0,
        smooth_sigma=7.0,
    )

    count = smoother.smooth_all(store, min_frames=10)

    print(f"Pass 2 Complete: Smoothed {count} trajectories")
    return count


def assign_identities(store: TrajectoryStore) -> dict:
    print("=" * 60)
    print("PASS 3: Identity Assignment")
    print("=" * 60)

    assigner = IdentityAssigner(fps=store.fps, debug=True)
    assignments = assigner.assign_roles(store)

    slot_manager = PitchSlotManager(fps=store.fps, debug=True)
    slot_manager.initialize_from_assignments(store, assignments)
    slot_manager.build_all_frame_positions(store, store.total_frames)

    print(f"Pass 3 Complete: Assigned {len(assignments)} identities")
    return assignments, slot_manager


def compute_dominance_heatmap_image_space(
    frame_shape: Tuple[int, int],
    pitch_positions: List[Tuple[float, float, float, float, int]],  # (x, y, vx, vy, team)
    homography_H_inv: np.ndarray,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    resolution: int = 50,
    base_sigma: float = 8.0,
) -> np.ndarray:
    """
    Compute space dominance heatmap with velocity-aware anisotropic influence.

    Returns an RGBA overlay image (H, W, 4) where:
    - Red/Orange = Team 0 dominates
    - Cyan/Blue = Team 1 dominates
    - Alpha = strength of dominance

    Influence is shifted in velocity direction and stretched along movement axis.
    Players moving fast have less influence behind them (harder to turn 180).
    """
    h_img, w_img = frame_shape[:2]

    if homography_H_inv is None or len(pitch_positions) < 2:
        return np.zeros((h_img, w_img, 4), dtype=np.uint8)

    # Create pitch grid - note: row 0 = +half_wid (top), row -1 = -half_wid (bottom)
    half_len = pitch_length / 2
    half_wid = pitch_width / 2

    x_grid = np.linspace(-half_len, half_len, resolution)
    y_grid = np.linspace(half_wid, -half_wid, resolution)  # Flipped: top to bottom
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Compute influence for each team with velocity-aware anisotropic Gaussians
    inf_0 = np.zeros_like(xx)
    inf_1 = np.zeros_like(xx)

    for px, py, vx, vy, team in pitch_positions:
        speed = np.sqrt(vx**2 + vy**2)
        angle = np.arctan2(vy, vx)

        # Momentum-shifted center (influence extends ahead of player)
        mu_x = px + vx * 10.0
        mu_y = py + vy * 10.0

        # Anisotropic sigma: stretched along movement, compressed perpendicular
        # At high speed, influence extends far ahead but not behind
        sigma_along = base_sigma * (1 + speed * 0.4)  # Stretch along velocity
        sigma_perp = base_sigma / (1 + speed * 0.15)  # Compress perpendicular

        # Rotate grid to align with velocity direction
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = xx - mu_x
        dy = yy - mu_y

        # Rotated coordinates
        dx_rot = dx * cos_a + dy * sin_a
        dy_rot = -dx * sin_a + dy * cos_a

        # Anisotropic Gaussian
        influence = np.exp(-(dx_rot**2 / (2 * sigma_along**2) + dy_rot**2 / (2 * sigma_perp**2)))

        if team == 0:
            inf_0 += influence
        elif team == 1:
            inf_1 += influence

    # Normalize to [0, 1]
    total = inf_0 + inf_1 + 1e-6
    dominance = (inf_0 - inf_1) / total  # -1 to 1

    # Create colored pitch heatmap with bright, visible colors
    # Using BGR format: Team 0 = Orange/Red, Team 1 = Cyan/Blue
    heatmap_pitch = np.zeros((resolution, resolution, 4), dtype=np.uint8)

    for i in range(resolution):
        for j in range(resolution):
            d = dominance[i, j]
            strength = min(abs(d), 1.0)
            # Softer alpha for better blending
            alpha = int(40 + strength * 100)  # Range 40-140

            if d > 0.02:
                # Team 0 - Orange/Red (BGR: some blue, no green, full red)
                heatmap_pitch[i, j] = [0, int(80 * strength), 220, alpha]
            elif d < -0.02:
                # Team 1 - Cyan (BGR: full blue, some green, no red)
                heatmap_pitch[i, j] = [220, int(140 * strength), 0, alpha]

    pitch_corners = np.array(
        [
            [-half_len, half_wid],  # top-left of heatmap = (-half_len, +half_wid)
            [half_len, half_wid],  # top-right
            [half_len, -half_wid],  # bottom-right
            [-half_len, -half_wid],  # bottom-left
        ],
        dtype=np.float32,
    )

    ones = np.ones((4, 1), dtype=np.float32)
    pts_h = np.hstack([pitch_corners, ones])
    projected = (homography_H_inv @ pts_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    src_pts = np.array(
        [[0, 0], [resolution - 1, 0], [resolution - 1, resolution - 1], [0, resolution - 1]], dtype=np.float32
    )
    dst_pts = projected.astype(np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(heatmap_pitch, M, (w_img, h_img), flags=cv2.INTER_LINEAR)

    return warped


def overlay_dominance_on_frame(frame: np.ndarray, dominance_overlay: np.ndarray) -> np.ndarray:
    """Blend dominance heatmap onto frame using alpha channel."""
    if dominance_overlay is None or dominance_overlay.shape[2] != 4:
        return frame

    frame_out = frame.copy()

    overlay_rgb = dominance_overlay[:, :, :3]
    alpha = dominance_overlay[:, :, 3:4].astype(np.float32) / 255.0
    frame_out = (frame_out * (1 - alpha) + overlay_rgb * alpha).astype(np.uint8)

    return frame_out


def draw_pitch_overlay_on_frame(
    frame: np.ndarray,
    homography_H_inv: np.ndarray,
    color: tuple = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Project pitch lines back onto the video frame."""
    if homography_H_inv is None:
        return frame

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


def render_visualization(
    video_path: str,
    store: TrajectoryStore,
    assignments: dict,
    slot_manager: PitchSlotManager,
    max_duration: float = None,
    draw_overlay: bool = True,
    draw_dominance: bool = True,
    homography_interval: int = 5,
) -> str:
    print("=" * 60)
    print("PASS 4: Visualization")
    print("=" * 60)

    pitch_viz = PitchVisualizer()

    # NOTE: slot info is included s.t. we can show merged IDs
    frame_observations = defaultdict(list)
    total_obs_count = 0
    for track_id, track in store.tracks.items():
        info = assignments.get(track_id, {'role': 'unknown', 'team': -1})

        slot_key = slot_manager.track_to_slot.get(track_id, None)

        for obs in track.observations:
            total_obs_count += 1
            frame_observations[obs.frame_idx].append(
                {
                    'track_id': track_id,
                    'box': obs.box,
                    'role': info.get('role', 'unknown'),
                    'team': info.get('team', -1),
                    'slot_key': slot_key,
                }
            )

    frames_with_obs = len(frame_observations)
    avg_obs_per_frame = total_obs_count / frames_with_obs if frames_with_obs > 0 else 0
    obs_counts = [len(frame_observations.get(i, [])) for i in range(store.total_frames)]
    min_obs = min(obs_counts) if obs_counts else 0
    max_obs = max(obs_counts) if obs_counts else 0
    print(f"[Diagnostic] Total observations: {total_obs_count}, Frames: {frames_with_obs}")
    print(f"[Diagnostic] Obs per frame: min={min_obs}, max={max_obs}, avg={avg_obs_per_frame:.1f}")

    output_path = generate_output_path(video_path, prefix="offline_match", duration=max_duration)

    current_H_inv = None

    position_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    smoothed_velocity: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

    MAX_SPEED = 10.0  # m/s - realistic max sprint speed
    MAX_ACCELERATION = 4.0  # m/s² - max acceleration from rest
    MAX_DECELERATION = 6.0  # m/s² - max braking deceleration
    VELOCITY_SMOOTHING = 0.3  # EMA alpha (lower = more smoothing)
    JITTER_THRESHOLD = 2.0  # m - max plausible single-frame displacement at 30fps

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata

        pitch_h, pitch_w = pitch_viz.base_pitch.shape[:2]
        scale = meta.height / pitch_h
        output_w = meta.width + int(pitch_w * scale)

        progress = ProgressTracker(reader.max_frames, log_interval=100)

        with VideoWriter(output_path, meta.fps, (output_w, meta.height)) as writer:
            for frame_idx, frame_bgr in enumerate(reader):
                obs_list = frame_observations.get(frame_idx, [])

                if draw_overlay:
                    for check_idx in range(frame_idx, -1, -1):
                        if check_idx in store.frame_homographies:
                            current_H_inv = store.frame_homographies[check_idx]
                            break

                if draw_overlay and current_H_inv is not None:
                    frame_bgr = draw_pitch_overlay_on_frame(frame_bgr, current_H_inv)

                slot_positions = slot_manager.get_frame_positions(frame_idx)

                for slot in slot_positions:
                    slot_key = slot['slot_key']
                    x, y = slot['position']
                    position_history[slot_key].append((x, y))
                    position_history[slot_key] = position_history[slot_key][-60:]

                for obs in obs_list:
                    box = obs['box']
                    role = obs['role']
                    team = obs['team']
                    tid = obs['track_id']
                    slot_key = obs.get('slot_key')

                    x1, y1, x2, y2 = map(int, box)

                    display_id = slot_key if slot_key else f"?{tid}"

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
                    cv2.putText(frame_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                pitch_positions = []
                dt = 1.0 / meta.fps  # Time between frames in seconds

                for slot in slot_positions:
                    x, y = slot['position']
                    team = slot['team']
                    slot_id = slot['slot_id']
                    slot_key = slot['slot_key']

                    # Get previous smoothed velocity
                    prev_vx, prev_vy = smoothed_velocity[slot_key]
                    prev_speed = np.sqrt(prev_vx**2 + prev_vy**2)

                    # Compute raw velocity from position history
                    raw_vx, raw_vy = 0.0, 0.0
                    if slot_key in position_history and len(position_history[slot_key]) >= 1:
                        hist = position_history[slot_key]
                        prev_x, prev_y = hist[-1]

                        # Single-frame displacement
                        dx = x - prev_x
                        dy = y - prev_y
                        displacement = np.sqrt(dx**2 + dy**2)

                        # Jitter detection: if displacement is physically impossible, ignore it
                        # At 30fps, max plausible displacement is ~0.33m (10 m/s * 1/30s)
                        max_frame_displacement = MAX_SPEED * dt * 1.5  # Allow 50% margin

                        if displacement > JITTER_THRESHOLD:
                            # Likely tracking jitter - use previous velocity
                            raw_vx, raw_vy = prev_vx, prev_vy
                        elif displacement > max_frame_displacement:
                            # Clamp displacement to physical limit, preserve direction
                            if displacement > 0:
                                scale_factor = max_frame_displacement / displacement
                                dx *= scale_factor
                                dy *= scale_factor
                            raw_vx = dx / dt
                            raw_vy = dy / dt
                        else:
                            raw_vx = dx / dt
                            raw_vy = dy / dt

                    # Apply acceleration constraints
                    target_vx, target_vy = raw_vx, raw_vy
                    dvx = target_vx - prev_vx
                    dvy = target_vy - prev_vy
                    dv_mag = np.sqrt(dvx**2 + dvy**2)

                    if dv_mag > 0:
                        # Check if accelerating or decelerating
                        target_speed = np.sqrt(target_vx**2 + target_vy**2)
                        max_dv = MAX_ACCELERATION * dt if target_speed > prev_speed else MAX_DECELERATION * dt

                        if dv_mag > max_dv:
                            # Clamp acceleration to physical limit
                            accel_scale = max_dv / dv_mag
                            dvx *= accel_scale
                            dvy *= accel_scale

                        constrained_vx = prev_vx + dvx
                        constrained_vy = prev_vy + dvy
                    else:
                        constrained_vx, constrained_vy = prev_vx, prev_vy

                    # Apply exponential moving average for smooth transitions
                    vx = prev_vx * (1 - VELOCITY_SMOOTHING) + constrained_vx * VELOCITY_SMOOTHING
                    vy = prev_vy * (1 - VELOCITY_SMOOTHING) + constrained_vy * VELOCITY_SMOOTHING

                    # Final speed clamp
                    final_speed = np.sqrt(vx**2 + vy**2)
                    if final_speed > MAX_SPEED:
                        speed_scale = MAX_SPEED / final_speed
                        vx *= speed_scale
                        vy *= speed_scale

                    # Store smoothed velocity for next frame
                    smoothed_velocity[slot_key] = (vx, vy)

                    # Flip Y to match heatmap and align with 3D view
                    pitch_positions.append((x, -y, vx, -vy, slot_id, team if team >= 0 else 2))

                # Build history dict for trails (with flipped Y)
                trail_history = {}
                for slot in slot_positions:
                    slot_key = slot['slot_key']
                    if slot_key in position_history:
                        trail_history[slot['slot_id']] = [(hx, -hy) for hx, hy in position_history[slot_key][-60:]]

                if pitch_positions and draw_dominance:
                    pitch_img = pitch_viz.draw_with_trails(
                        pitch_positions,
                        trail_history,
                        trail_length=60,
                        draw_vectors=True,
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

                        # Flip Y to match heatmap and align with 3D view
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
                        cv2.putText(
                            pitch_img,
                            str(slot_id),
                            (px - 4, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 255, 255),
                            1,
                        )

                pitch_scaled = cv2.resize(pitch_img, (int(pitch_w * scale), meta.height))
                combined = np.hstack([frame_bgr, pitch_scaled])
                writer.write(combined)

                progress.update()
                if progress.should_log():
                    print(f"Pass 4: {progress.status()}")

    print(f"Pass 4 Complete: Video saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Unified Offline Match Analysis")
    parser.add_argument("--video", type=str, default="videos/croatia_czechia.mp4", help="Input video path")
    parser.add_argument("--model", type=str, default=None, help="Detection model path (auto-detected if not provided)")
    parser.add_argument("--model_type", type=str, default="fcnn", choices=["yolo", "fcnn"])
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

    if args.model is None:
        if args.model_type == "yolo":
            args.model = "yolo11n.pt"
        elif args.model_type == "fcnn":
            args.model = "models/player/fcnn/fcnn_player_tracker.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    detector, pnl_calib = load_models(device, args.model, args.model_type)

    store = detect_and_project(
        args.video,
        detector,
        pnl_calib,
        args.model_type,
        max_duration=args.duration,
        homography_interval=args.homography_interval,
        device=device,
    )

    smooth_trajectories(store)

    assignments, slot_manager = assign_identities(store)

    output_path = render_visualization(
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
