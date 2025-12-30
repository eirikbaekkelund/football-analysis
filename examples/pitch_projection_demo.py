import argparse
import cv2
import torch
import numpy as np
import os
from typing import Dict, List, Tuple

from match_state.player_tracker import TeamClassifier
from match_state.pitch_homography import HomographyEstimator, PitchVisualizer, PITCH_LINE_COORDINATES
from models.pnl_calib.wrapper import PnLCalibWrapper


class PitchTrackManager:
    """
    Manages player tracks on the pitch plane with smoothing and persistence.
    """

    def __init__(self, max_missing_frames: int = 15, smoothing_alpha: float = 0.2):
        self.tracks = {}  # id -> {pos: (x,y), velocity: (vx,vy), last_seen: frame_idx, team: int, history: []}
        self.max_missing_frames = max_missing_frames
        self.alpha = smoothing_alpha

    def update(
        self, observations: List[Tuple[float, float, int, int]], frame_idx: int
    ) -> List[Tuple[float, float, float, float, int, int]]:
        """
        Update tracks with new observations.

        Args:
            observations: List of (x, y, track_id, team_id)
            frame_idx: Current frame number

        Returns:
            List of active tracks (x, y, vx, vy, track_id, team_id)
        """
        observed_ids = set()

        for x, y, tid, team in observations:
            observed_ids.add(tid)

            if tid not in self.tracks:
                self.tracks[tid] = {
                    "pos": (x, y),
                    "velocity": (0.0, 0.0),
                    "last_seen": frame_idx,
                    "team": team,
                    "history": [(x, y)],
                }
            else:
                # smooth position
                prev_x, prev_y = self.tracks[tid]["pos"]

                # exponential smoothing for potential jitter reduction of positions
                smooth_x = prev_x * (1 - self.alpha) + x * self.alpha
                smooth_y = prev_y * (1 - self.alpha) + y * self.alpha

                # velocity (pixels/frame) - calculated from smoothed positions
                inst_vx = smooth_x - prev_x
                inst_vy = smooth_y - prev_y

                # smooth velocity vectors as well
                prev_vx, prev_vy = self.tracks[tid]["velocity"]
                smooth_vx = prev_vx * (1 - self.alpha) + inst_vx * self.alpha
                smooth_vy = prev_vy * (1 - self.alpha) + inst_vy * self.alpha

                self.tracks[tid]["pos"] = (smooth_x, smooth_y)
                self.tracks[tid]["velocity"] = (smooth_vx, smooth_vy)
                self.tracks[tid]["last_seen"] = frame_idx
                if team is not None:
                    self.tracks[tid]["team"] = team

                self.tracks[tid]["history"].append((smooth_x, smooth_y))
                if len(self.tracks[tid]["history"]) > 50:
                    self.tracks[tid]["history"].pop(0)

        active_tracks = []
        to_remove = []

        for tid, track in self.tracks.items():
            if tid not in observed_ids:
                # decay velocity for stationary ghosts
                vx, vy = track["velocity"]
                track["velocity"] = (vx * 0.9, vy * 0.9)

            if frame_idx - track["last_seen"] <= self.max_missing_frames:
                x, y = track["pos"]
                vx, vy = track["velocity"]
                active_tracks.append((x, y, vx, vy, tid, track["team"]))
            else:
                to_remove.append(tid)

        for tid in to_remove:
            del self.tracks[tid]

        return active_tracks

    def get_history(self) -> Dict[int, List[Tuple[float, float]]]:
        return {tid: track["history"] for tid, track in self.tracks.items()}


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


def load_player_detector(model_path: str, device: torch.device):
    """Load YOLO model for player detection and tracking."""
    from ultralytics import YOLO

    print(f"Loading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Could not load {model_path}. Error: {e}")
        raise e

    return model


def create_combined_visualization(
    frame: np.ndarray,
    pitch_img: np.ndarray,
    tracks: List[dict],
    homography_ok: bool,
    num_inliers: int = 0,
    pitch_viz: PitchVisualizer = None,
) -> np.ndarray:
    """
    Create side-by-side visualization of video frame and pitch map.
    """
    h, w = frame.shape[:2]
    pitch_h, pitch_w = pitch_img.shape[:2]

    # scale pitch to match frame height
    scale = h / pitch_h
    new_pitch_w = int(pitch_w * scale)
    pitch_scaled = cv2.resize(pitch_img, (new_pitch_w, h))

    frame_viz = frame.copy()
    for track in tracks:
        box = track["box"]
        track_id = track["id"]
        team_id = track.get("team")

        x1, y1, x2, y2 = map(int, box)

        if team_id == 0:
            color = pitch_viz.team1_color if pitch_viz else (0, 0, 255)
        elif team_id == 1:
            color = pitch_viz.team2_color if pitch_viz else (255, 0, 0)
        elif team_id == 2:  # officials (referee + linesmen)
            color = pitch_viz.referee_color if pitch_viz else (0, 255, 255)
        else:
            color = pitch_viz.unknown_color if pitch_viz else (128, 128, 128)

        cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_viz, f"#{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        foot_x, foot_y = int((x1 + x2) / 2), int(y2)
        cv2.circle(frame_viz, (foot_x, foot_y), 4, (0, 255, 0), -1)

    status_color = (0, 255, 0) if homography_ok else (0, 0, 255)
    status_text = f"Homography: {'OK' if homography_ok else 'Failed'} ({num_inliers} inliers)"
    cv2.putText(frame_viz, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    combined = np.hstack([frame_viz, pitch_scaled])

    return combined


def draw_pitch_overlay(
    frame: np.ndarray,
    homography: HomographyEstimator,
    color: Tuple[int, int, int] = (255, 255, 0),  # cyan
    thickness: int = 2,
) -> np.ndarray:
    """
    Project pitch lines back onto the frame to verify homography.
    """
    if homography.H_inv is None:
        return frame

    frame_viz = frame.copy()
    h, w = frame.shape[:2]

    # draw all defined pitch lines
    for class_name, points in PITCH_LINE_COORDINATES.items():
        pitch_pts = np.array([p.to_array() for _, p in points])

        img_pts = homography.project_to_image(pitch_pts)

        if img_pts is None:
            continue

        # filter out points outside image
        valid_pts = []
        for pt in img_pts:
            x, y = int(pt[0]), int(pt[1])
            # allow some margin outside image
            if -1000 < x < w + 1000 and -1000 < y < h + 1000:
                valid_pts.append((x, y))

        if len(valid_pts) < 2:
            continue

        # drawing lines
        if "Circle" in class_name:
            # closed loop if enough points
            if len(valid_pts) >= 3:
                pts = np.array(valid_pts, dtype=np.int32)
                cv2.polylines(frame_viz, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            for i in range(len(valid_pts) - 1):
                cv2.line(frame_viz, valid_pts[i], valid_pts[i + 1], color, thickness)

    return frame_viz


def main():
    parser = argparse.ArgumentParser(description="Player tracking with pitch projection")
    parser.add_argument("--video", type=str, default="videos/croatia_czechia.mp4", help="Input video path")
    parser.add_argument("--output", type=str, default="output_2d_test.mp4", help="Output video path")
    parser.add_argument(
        "--model", type=str, default="models/tracking/yolo/yolov11_player_tracker.pt", help="Path to YOLO model"
    )
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="Classes to track (default: auto)")
    parser.add_argument("--homography-interval", type=int, default=5, help="Frames between homography updates")
    parser.add_argument("--ghost-frames", type=int, default=15, help="Frames to keep lost tracks alive")
    parser.add_argument("--smooth-alpha", type=float, default=0.2, help="Smoothing factor (0.0-1.0)")
    parser.add_argument("--no-vectors", action="store_true", help="Disable velocity vectors")
    parser.add_argument("--no-dominance", action="store_true", help="Disable spatial dominance heatmap")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    parser.add_argument("--duration", type=int, default=60, help="Output video duration in seconds")
    parser.add_argument(
        "--debug-overlay", action="store_true", default=True, help="Draw reprojected pitch lines on video"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pnl_calib = load_pnl_calib(device)
    player_detector = load_player_detector(args.model, device)

    track_classes = args.classes
    if track_classes is None:
        names = player_detector.names
        if 0 in names and names[0] == 'person':
            track_classes = [0]
        else:
            track_classes = list(names.keys())
    print(f"Tracking classes: {track_classes}")

    team_classifier = TeamClassifier()
    track_manager = PitchTrackManager(max_missing_frames=args.ghost_frames, smoothing_alpha=args.smooth_alpha)

    homography = HomographyEstimator(
        min_correspondences=6,
        confidence_threshold=0.3,
        visibility_threshold=0.3,
    )
    pitch_viz = PitchVisualizer()

    cap = cv2.VideoCapture(args.video)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = min(fps * args.duration, total_frames)

    print(f"Video: {frame_w}x{frame_h} @ {fps}fps, {total_frames} frames")
    print(f"Output limited to {args.duration}s ({max_frames} frames)")

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

            if frame_idx % args.homography_interval == 0:
                P = pnl_calib.process_frame(frame)
                if P is not None:
                    # P maps (x, y, 0, 1) to (u, v, w)
                    # H maps (x, y, 1) to (u, v, w)
                    # s.t. H is columns 0, 1, 3 of P
                    H_inv = P[:, [0, 1, 3]]

                    # H_inv normalization
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

            results = player_detector.track(
                frame, persist=True, tracker="botsort.yaml", verbose=False, classes=track_classes
            )

            tracks = []
            pitch_positions = {}  # track_id -> (x, y) in meters

            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()

                raw_tracks = []
                for box, track_id in zip(boxes, track_ids):
                    raw_tracks.append({'box': box.tolist(), 'id': int(track_id)})

                    # pitch position for linesman detection
                    if homography_ok:
                        pitch_pos = homography.project_player_to_pitch(box.tolist())
                        if pitch_pos is not None:
                            pitch_positions[int(track_id)] = pitch_pos

                tracks = team_classifier.update(raw_tracks, frame_rgb, pitch_positions)

            current_observations = []
            for track in tracks:
                if homography_ok:
                    pitch_pos = homography.project_player_to_pitch(track["box"])
                    if pitch_pos is not None:
                        x, y = pitch_pos
                        current_observations.append((x, y, track["id"], track.get("team")))

            active_pitch_positions = track_manager.update(current_observations, frame_idx)

            if active_pitch_positions:
                pitch_img = pitch_viz.draw_with_trails(
                    active_pitch_positions,
                    track_manager.get_history(),
                    draw_vectors=not args.no_vectors,
                    draw_dominance=not args.no_dominance,
                )
            else:
                pitch_img = pitch_viz.base_pitch.copy()

            if args.debug_overlay and homography_ok:
                frame = draw_pitch_overlay(frame, homography)

            combined = create_combined_visualization(
                frame, pitch_img, tracks, homography_ok, num_inliers=0, pitch_viz=pitch_viz
            )

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
