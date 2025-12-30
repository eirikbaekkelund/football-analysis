import torch
import torchvision
import cv2
import numpy as np
import argparse
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from match_state.player_tracker import PlayerTracker


def get_player_detector_model(num_classes: int = 2) -> torchvision.models.detection.FasterRCNN:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def process_video(video_path: str, checkpoint_path: str, max_duration: float = None) -> None:
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Processing {video_path} on {DEVICE}...")

    model = get_player_detector_model(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    tracker = PlayerTracker(max_age=30, iou_threshold=0.3)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    max_frames = int(max_duration * fps) if max_duration else float('inf')
    if max_duration:
        print(f"Limiting processing to {max_duration} seconds ({max_frames} frames)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    n_sec = max_duration if max_duration else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    output_path = f"output_{video_path.split('/')[-1]}_{int(n_sec)}s_tracked.mp4"
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
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}...")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # converts [0, 255] -> [0.0, 1.0] and HWC -> CHW
        input_tensor = torchvision.transforms.functional.to_tensor(frame_rgb).to(DEVICE)
        input_tensor = input_tensor.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        with torch.no_grad():
            prediction = model(input_tensor)

        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()

        valid_detections = []
        for box, score in zip(boxes, scores):
            if score > 0.8:
                valid_detections.append(np.append(box, score))

        active_tracks = tracker.update(valid_detections, frame_rgb)

        for track in active_tracks:
            x1, y1, x2, y2 = map(int, track['box'])

            team_id = track.get('team', '?')

            if team_id == 0:
                color = (0, 0, 255)  # team 0 = red
            elif team_id == 1:
                color = (255, 0, 0)  # team 1 = blue
            else:
                color = (0, 255, 0)  # ?/unknown = green

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track['id']} T:{team_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame_bgr)

    cap.release()
    out.release()
    print(f"Done! Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="videos/croatia_czechia.mp4", help="Path to input video file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tracking_models/fcnn/fcnn_player_tracker.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--duration", type=float, default=15, help="Max duration in seconds to process")

    args = parser.parse_args()

    process_video(args.input, args.checkpoint, args.duration)
