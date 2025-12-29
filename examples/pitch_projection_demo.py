"""
Full pipeline example: Player tracking with 2D pitch projection.

This script demonstrates:
1. Player detection (YOLO)
2. Player tracking with team classification
3. Line detection for camera calibration
4. Homography estimation
5. 2D pitch visualization with player positions

Usage:
    python examples/pitch_projection_demo.py --video input.mp4 --output output.mp4
"""

import argparse
import cv2
import torch
import numpy as np
import os
from typing import Dict, List, Tuple
from collections import defaultdict

from match_state.player_tracker import TeamClassifier
from match_state.pitch_homography import HomographyEstimator, PitchVisualizer, PITCH_LINE_COORDINATES
from models.pnl_calib.wrapper import PnLCalibWrapper


def load_pnl_calib(device: torch.device):
    """Load the PnLCalib model."""
    base_path = os.path.join("models", "pnl_calib")
    weights_kp = os.path.join(base_path, "weights", "SV_kp")
    weights_line = os.path.join(base_path, "weights", "SV_lines")
    config_kp = os.path.join(base_path, "config", "hrnetv2_w48.yaml")
    config_line = os.path.join(base_path, "config", "hrnetv2_w48_l.yaml")

    if not os.path.exists(weights_kp) or not os.path.exists(weights_line):
        print(f"Warning: Weights not found at {weights_kp} or {weights_line}")
        print("Please download them from https://github.com/mguti97/PnLCalib/releases/tag/v1.0.0")

    return PnLCalibWrapper(weights_kp, weights_line, config_kp, config_line, device=str(device))


def load_player_detector(device: torch.device):
    """Load YOLO model for player detection and tracking."""
    from ultralytics import YOLO

    # Using YOLO11n as requested for better tracking support
    try:
        model = YOLO("yolo11n.pt")
    except Exception as e:
        print(f"Could not load yolo11n.pt, falling back to yolov8m.pt. Error: {e}")
        model = YOLO("yolov8m.pt")

    return model


def detect_players(
    model,
    frame: np.ndarray,
    conf_threshold: float = 0.3,
) -> List[List[float]]:
    """
    Detect players using YOLO.

    Returns:
        List of [x1, y1, x2, y2, score] detections
    """
    results = model(frame, verbose=False, conf=conf_threshold)

    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            # COCO class 0 = person
            if cls == 0:
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                score = float(boxes.conf[i])
                detections.append([x1, y1, x2, y2, score])

    return detections


def create_combined_visualization(
    frame: np.ndarray,
    pitch_img: np.ndarray,
    tracks: List[dict],
    homography_ok: bool,
    num_inliers: int = 0,
) -> np.ndarray:
    """
    Create side-by-side visualization of video frame and pitch map.
    """
    h, w = frame.shape[:2]
    pitch_h, pitch_w = pitch_img.shape[:2]

    # Scale pitch to match frame height
    scale = h / pitch_h
    new_pitch_w = int(pitch_w * scale)
    pitch_scaled = cv2.resize(pitch_img, (new_pitch_w, h))

    # Draw tracking boxes on frame
    frame_viz = frame.copy()
    for track in tracks:
        box = track["box"]
        track_id = track["id"]
        team_id = track.get("team")

        x1, y1, x2, y2 = map(int, box)

        # Team colors
        if team_id == 0:
            color = (0, 0, 255)  # Red
        elif team_id == 1:
            color = (255, 0, 0)  # Blue
        else:
            color = (128, 128, 128)  # Gray

        cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_viz, f"#{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw foot position
        foot_x, foot_y = int((x1 + x2) / 2), int(y2)
        cv2.circle(frame_viz, (foot_x, foot_y), 4, (0, 255, 0), -1)

    # Add homography status
    status_color = (0, 255, 0) if homography_ok else (0, 0, 255)
    status_text = f"Homography: {'OK' if homography_ok else 'Failed'} ({num_inliers} inliers)"
    cv2.putText(frame_viz, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Combine horizontally
    combined = np.hstack([frame_viz, pitch_scaled])

    return combined


def draw_pitch_overlay(
    frame: np.ndarray,
    homography: HomographyEstimator,
    color: Tuple[int, int, int] = (255, 255, 0),  # Cyan
    thickness: int = 2,
) -> np.ndarray:
    """
    Project pitch lines back onto the frame to verify homography.
    """
    if homography.H_inv is None:
        return frame

    frame_viz = frame.copy()
    h, w = frame.shape[:2]

    # Draw all defined pitch lines
    for class_name, points in PITCH_LINE_COORDINATES.items():
        # Get pitch points for this line
        pitch_pts = np.array([p.to_array() for _, p in points])

        # Project to image
        img_pts = homography.project_to_image(pitch_pts)

        if img_pts is None:
            continue

        # Filter points outside image
        valid_pts = []
        for pt in img_pts:
            x, y = int(pt[0]), int(pt[1])
            # Allow some margin outside image
            if -1000 < x < w + 1000 and -1000 < y < h + 1000:
                valid_pts.append((x, y))

        if len(valid_pts) < 2:
            continue

        # Draw lines
        if "Circle" in class_name:
            # Draw as closed loop if enough points
            if len(valid_pts) >= 3:
                pts = np.array(valid_pts, dtype=np.int32)
                cv2.polylines(frame_viz, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            # Draw as connected segments
            for i in range(len(valid_pts) - 1):
                cv2.line(frame_viz, valid_pts[i], valid_pts[i + 1], color, thickness)

    return frame_viz


def main():
    parser = argparse.ArgumentParser(description="Player tracking with pitch projection")
    parser.add_argument("--video", type=str, default="videos/croatia_czechia.mp4", help="Input video path")
    parser.add_argument("--output", type=str, default="output_2d_test.mp4", help="Output video path")
    parser.add_argument("--homography-interval", type=int, default=5, help="Frames between homography updates")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    parser.add_argument("--duration", type=int, default=60, help="Output video duration in seconds")
    parser.add_argument(
        "--debug-overlay", action="store_true", default=True, help="Draw reprojected pitch lines on video"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("Loading PnLCalib model...")
    pnl_calib = load_pnl_calib(device)

    print("Loading player detector...")
    player_detector = load_player_detector(device)

    # Initialize tracker and homography
    # tracker = PlayerTracker() # Deprecated in favor of YOLO11 tracking
    team_classifier = TeamClassifier()

    homography = HomographyEstimator(
        min_correspondences=6,
        confidence_threshold=0.3,
        visibility_threshold=0.3,
    )
    pitch_viz = PitchVisualizer()

    # Position history for trails
    position_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Limit output to requested duration
    max_frames = min(fps * args.duration, total_frames)

    print(f"Video: {frame_w}x{frame_h} @ {fps}fps, {total_frames} frames")
    print(f"Output limited to {args.duration}s ({max_frames} frames)")

    # Setup output video
    # Combined width = frame + scaled pitch
    pitch_h, pitch_w = pitch_viz.base_pitch.shape[:2]
    scale = frame_h / pitch_h
    output_w = frame_w + int(pitch_w * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (output_w, frame_h))

    frame_idx = 0
    homography_ok = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= max_frames:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update homography periodically
            if frame_idx % args.homography_interval == 0:
                P = pnl_calib.process_frame(frame)
                if P is not None:
                    # P maps (x, y, 0, 1) to (u, v, w)
                    # H maps (x, y, 1) to (u, v, w)
                    # So H is columns 0, 1, 3 of P
                    H_inv = P[:, [0, 1, 3]]

                    # Normalize H_inv
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

            # Detect and track players
            # Run YOLO11 tracking with BoT-SORT
            # persist=True maintains tracks between frames
            # classes=[0] filters for persons only
            results = player_detector.track(frame, persist=True, tracker="botsort.yaml", verbose=False, classes=[0])

            tracks = []
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()

                raw_tracks = []
                for box, track_id in zip(boxes, track_ids):
                    raw_tracks.append({'box': box.tolist(), 'id': int(track_id)})

                # Update team classification
                tracks = team_classifier.update(raw_tracks, frame_rgb)

            # Project to pitch coordinates
            pitch_positions = []
            for track in tracks:
                if homography_ok:
                    pitch_pos = homography.project_player_to_pitch(track["box"])
                    if pitch_pos is not None:
                        x, y = pitch_pos
                        pitch_positions.append((x, y, track["id"], track.get("team")))

                        # Update history
                        position_history[track["id"]].append((x, y))
                        # Limit history length
                        if len(position_history[track["id"]]) > 100:
                            position_history[track["id"]] = position_history[track["id"]][-100:]

            # Draw pitch visualization
            if pitch_positions:
                pitch_img = pitch_viz.draw_with_trails(pitch_positions, position_history)
            else:
                pitch_img = pitch_viz.base_pitch.copy()

            # Draw reprojected pitch lines (debug)
            if args.debug_overlay and homography_ok:
                frame = draw_pitch_overlay(frame, homography)

            # Create combined visualization
            combined = create_combined_visualization(
                frame, pitch_img, tracks, homography_ok, num_inliers=0  # PnLCalib doesn't return inliers count easily
            )

            # Write output
            out.write(combined)

            if args.show:
                cv2.imshow("Pitch Projection", cv2.resize(combined, (1600, 600)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\nSaved output to {args.output}")
    print(f"Processed {frame_idx} frames")


if __name__ == "__main__":
    main()
