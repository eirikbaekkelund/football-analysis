import argparse
import torch
import torchvision
import cv2
import numpy as np
from torch.amp import autocast
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from match_state.player_tracker import PlayerTracker
from utils.video import VideoReader, VideoWriter, ProgressTracker, generate_output_path
from utils.visualization import TrackVisualizer


def get_player_detector_model(num_classes: int = 2) -> torchvision.models.detection.FasterRCNN:
    """Load Faster-RCNN with custom head for player detection."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def process_video(
    video_path: str,
    checkpoint_path: str,
    max_duration: float = None,
    confidence_threshold: float = 0.8,
) -> None:
    """
    Process video with Faster-RCNN detection and tracking.

    NOTE: Batching does NOT speed up Faster-RCNN inference. The RPN and ROI pooling
    operations process each image independently, so batch_size>1 just increases memory
    usage without improving throughput. Tested: batch=4 gave 3.3 FPS vs 3.7 FPS at batch=1.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    print(f"Model: Faster-RCNN (BSD licensed)")
    print(f"Device: {device} (AMP={use_amp})")

    # Load model
    model = get_player_detector_model(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Tracker and visualizer
    tracker = PlayerTracker(max_age=30, iou_threshold=0.3)
    visualizer = TrackVisualizer()

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata
        output_path = generate_output_path(video_path, prefix="output_fcnn", duration=max_duration)

        print(f"Input: {meta.width}x{meta.height} @ {meta.fps}fps")
        print(f"Processing {reader.max_frames} frames...")

        progress = ProgressTracker(reader.max_frames, log_interval=10)

        with VideoWriter(output_path, meta.fps, (meta.width, meta.height)) as writer:
            for frame_bgr in reader:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tensor = torchvision.transforms.functional.to_tensor(frame_rgb).to(device)

                # Inference
                with torch.no_grad():
                    if use_amp:
                        with autocast(device_type='cuda', dtype=torch.float16):
                            predictions = model([tensor])
                    else:
                        predictions = model([tensor])

                pred = predictions[0]
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()

                # Filter by confidence
                valid_detections = []
                for box, score in zip(boxes, scores):
                    if score > confidence_threshold:
                        valid_detections.append(np.append(box, score))

                # Update tracker
                active_tracks = tracker.update(valid_detections, frame_rgb)

                # Draw tracks
                for track in active_tracks:
                    visualizer.draw_track(frame_bgr, track)

                writer.write(frame_bgr)
                progress.update()

                if progress.should_log():
                    print(progress.status())

        print(f"\n{progress.summary()}")
        print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faster-RCNN player tracking demo")
    parser.add_argument("--video", type=str, default="videos/croatia_czechia.mp4", help="Input video path")
    parser.add_argument(
        "--checkpoint", type=str, default="models/player/fcnn/fcnn_player_tracker.pth", help="Model checkpoint"
    )
    parser.add_argument("--duration", type=float, default=15, help="Max duration in seconds")
    parser.add_argument("--threshold", type=float, default=0.8, help="Detection confidence threshold")

    args = parser.parse_args()
    process_video(args.video, args.checkpoint, args.duration, args.threshold)
