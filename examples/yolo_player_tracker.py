import cv2
import argparse
import time
import torch
from ultralytics import YOLO
from match_state.player_tracker import TeamClassifier


def process_video(
    video_path: str, model_path: str, max_duration: float = None, show_fps: bool = True, tracker: str = None
) -> None:
    print(f"Loading YOLO model from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    print(f"Running on: {device}")

    team_classifier = TeamClassifier()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    max_frames = int(max_duration * fps) if max_duration else float('inf')
    if max_duration:
        print(f"Limiting processing to {max_duration} seconds ({max_frames} frames)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    n_sec = max_duration if max_duration else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    video_folder = '/'.join(video_path.split('/')[:-1])
    model_name = model_path.split('/')[-1].split('.')[0]
    output_path = f"{video_folder}/{model_name}_{int(n_sec)}s_tracked.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_count >= max_frames:
            print(f"Reached max duration of {max_duration} seconds. Stopping.")
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()

        results = model.track(frame_bgr, persist=True, tracker=f"{tracker}.yaml", verbose=False)

        tracks_list = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                tracks_list.append({'id': track_id, 'box': box, 'team': 0})  # TODO: refs/outliers tbd

        active_tracks = team_classifier.update(tracks_list, frame_rgb)
        end = time.perf_counter()

        for track in active_tracks:
            x1, y1, x2, y2 = map(int, track['box'])
            team_id = track.get('team', '?')

            if team_id == 0:
                color = (0, 0, 255)  # team 0 = red
            elif team_id == 1:
                color = (255, 0, 0)  # team 1 = blue
            else:
                color = (0, 255, 0)  # green = unknown

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            label = f"T{team_id}:{track['id']}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if show_fps and frame_count > 1:
            current_fps = 1.0 / (end - start)
            cv2.putText(frame_bgr, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame_bgr)

    cap.release()
    out.release()
    print(f"\nOutput saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="videos/croatia_czechia.mp4", help="Path to input video file")
    parser.add_argument(
        "--model", type=str, default="models/tracking/yolo/yolo11_player_tracker.pt", help="Path to YOLO model"
    )
    parser.add_argument("--duration", type=float, default=15, help="Max duration in seconds to process")
    parser.add_argument("--tracker", type=str, default="botsort", help="Optional YOLO tracker yaml file")
    args = parser.parse_args()

    process_video(args.input, args.model, args.duration, tracker=args.tracker)
