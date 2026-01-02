"""
YOLO Player Detection and Tracking Demo

Uses Ultralytics YOLO for fast player detection and tracking.
NOTE: AGPL-3.0 license - requires commercial license for closed-source use.

Expected ~60+ FPS on RTX 2000 Ada.

Usage:
    python examples/yolo_player_tracker.py --input videos/match.mp4 --duration 15
"""

import argparse
import time
import cv2
import torch
from ultralytics import YOLO

from match_state.player_tracker import TeamClassifier
from utils.video import VideoReader, VideoWriter, ProgressTracker, generate_output_path
from utils.visualization import TrackVisualizer


def process_video(
    video_path: str,
    model_path: str,
    max_duration: float = None,
    tracker: str = "botsort",
    show_fps: bool = True,
) -> None:
    """Process video with YOLO detection and tracking."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Model: YOLO (AGPL-3.0 - commercial license required)")
    print(f"Device: {device}")
    print(f"Loading model from {model_path}...")

    model = YOLO(model_path)
    model.to(device)

    team_classifier = TeamClassifier()
    visualizer = TrackVisualizer()

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata
        output_path = generate_output_path(video_path, prefix="output_yolo", duration=max_duration)

        print(f"Input: {meta.width}x{meta.height} @ {meta.fps}fps")
        print(f"Processing {reader.max_frames} frames...")

        progress = ProgressTracker(reader.max_frames, log_interval=30)

        with VideoWriter(output_path, meta.fps, (meta.width, meta.height)) as writer:
            for frame_bgr in reader:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                start = time.perf_counter()

                results = model.track(
                    frame_bgr,
                    persist=True,
                    tracker=f"{tracker}.yaml",
                    verbose=False,
                )

                tracks_list = []
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        tracks_list.append({'id': track_id, 'box': box, 'team': 0})

                active_tracks = team_classifier.update(tracks_list, frame_rgb)

                end = time.perf_counter()

                for track in active_tracks:
                    visualizer.draw_track(frame_bgr, track, label_format="T{team}:{id}")

                if show_fps:
                    current_fps = 1.0 / (end - start) if (end - start) > 0 else 0
                    visualizer.draw_fps(frame_bgr, current_fps)

                writer.write(frame_bgr)
                progress.update()

                if progress.should_log():
                    print(progress.status())

        print(f"\n{progress.summary()}")
        print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO player tracking demo")
    parser.add_argument("--video", type=str, default="videos/croatia_czechia.mp4", help="Input video path")
    parser.add_argument(
        "--model", type=str, default="models/player/yolo/yolov11_player_tracker.pt", help="YOLO model path"
    )
    parser.add_argument("--duration", type=float, default=15, help="Max duration in seconds")
    parser.add_argument("--tracker", type=str, default="botsort", help="YOLO tracker config (botsort/bytetrack)")

    args = parser.parse_args()
    process_video(args.video, args.model, args.duration, args.tracker)
