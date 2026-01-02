"""
RT-DETR Player Detection and Tracking Demo

RT-DETR is Apache 2.0 licensed - safe for commercial use.
Expected ~6 FPS on RTX 2000 Ada with FP16 (vs ~3.7 FPS with Faster-RCNN).

Usage:
    python examples/rtdetr_player_tracker.py --input videos/match.mp4 --duration 15
"""

import argparse
import torch
import cv2
import numpy as np
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from match_state.player_tracker import PlayerTracker
from utils.video import VideoReader, VideoWriter, ProgressTracker, generate_output_path
from utils.visualization import TrackVisualizer


def load_rtdetr_model(checkpoint_path: str, device: torch.device, use_fp16: bool = True):
    """Load fine-tuned RT-DETR model."""
    model = RTDetrForObjectDetection.from_pretrained(checkpoint_path)
    processor = RTDetrImageProcessor.from_pretrained(checkpoint_path)

    model.to(device)
    if use_fp16 and device.type == 'cuda':
        model = model.half()
    model.eval()

    return model, processor


def load_pretrained_rtdetr(device: torch.device, use_fp16: bool = True):
    """Load pretrained RT-DETR (COCO). Class 0 = person."""
    model_name = "PekingU/rtdetr_r50vd"

    model = RTDetrForObjectDetection.from_pretrained(model_name)
    processor = RTDetrImageProcessor.from_pretrained(model_name)

    model.to(device)
    if use_fp16 and device.type == 'cuda':
        model = model.half()
    model.eval()

    return model, processor


def process_video(
    video_path: str,
    checkpoint_path: str = None,
    max_duration: float = None,
    confidence_threshold: float = 0.7,
) -> None:
    """Process video with RT-DETR detection and tracking."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = device.type == 'cuda'

    print(f"Model: RT-DETR (Apache 2.0 - commercial safe)")
    print(f"Device: {device} (FP16={use_fp16})")

    # Load model
    if checkpoint_path and checkpoint_path != "pretrained":
        print(f"Loading fine-tuned model from {checkpoint_path}")
        model, processor = load_rtdetr_model(checkpoint_path, device, use_fp16)
        use_coco = False
    else:
        print("Using pretrained COCO model (class 0 = person)")
        model, processor = load_pretrained_rtdetr(device, use_fp16)
        use_coco = True

    tracker = PlayerTracker(max_age=30, iou_threshold=0.3)
    visualizer = TrackVisualizer()

    with VideoReader(video_path, max_duration=max_duration) as reader:
        meta = reader.metadata
        output_path = generate_output_path(video_path, prefix="output_rtdetr", duration=max_duration)

        print(f"Input: {meta.width}x{meta.height} @ {meta.fps}fps")
        print(f"Processing {reader.max_frames} frames...")

        progress = ProgressTracker(reader.max_frames, log_interval=30)

        with VideoWriter(output_path, meta.fps, (meta.width, meta.height)) as writer:
            for frame_bgr in reader:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                inputs = processor(images=frame_rgb, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                if use_fp16:
                    pixel_values = pixel_values.half()

                with torch.no_grad():
                    outputs = model(pixel_values)

                target_sizes = torch.tensor([[meta.height, meta.width]], device=device)
                results = processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=confidence_threshold,
                )[0]

                boxes = results['boxes'].cpu().numpy()
                scores = results['scores'].cpu().numpy()
                labels = results['labels'].cpu().numpy()

                valid_detections = []
                for box, score, label in zip(boxes, scores, labels):
                    if use_coco and label != 0:
                        continue
                    if score > confidence_threshold:
                        valid_detections.append(np.append(box, score))

                active_tracks = tracker.update(valid_detections, frame_rgb)

                for track in active_tracks:
                    visualizer.draw_track(frame_bgr, track)

                writer.write(frame_bgr)
                progress.update()

                if progress.should_log():
                    print(progress.status())

        print(f"\n{progress.summary()}")
        print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-DETR player tracking demo")
    parser.add_argument("--video", type=str, default="videos/croatia_czechia.mp4", help="Input video path")
    parser.add_argument("--checkpoint", type=str, default="pretrained", help="Checkpoint path or 'pretrained'")
    parser.add_argument("--duration", type=float, default=15, help="Max duration in seconds")
    parser.add_argument("--threshold", type=float, default=0.7, help="Detection confidence threshold")

    args = parser.parse_args()
    process_video(args.video, args.checkpoint, args.duration, args.threshold)
