import cv2
import numpy as np
import argparse
import time
import torch
import os
from models.line_detection.model import KeyPointLineDetector
from soccernet.calibration_data import LINE_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_video(
    video_path: str, model_path: str, max_duration: float = None, show_fps: bool = True, threshold: float = 0.5
) -> None:
    print(f"Loading Line Detection model from {model_path}...")

    # Initialize model
    model = KeyPointLineDetector(num_classes=len(LINE_CLASSES), max_points=12)

    # Load weights
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(DEVICE)
    model.eval()
    print(f"Running on: {DEVICE}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Model input resolution
    MODEL_W, MODEL_H = 960, 540

    max_frames = int(max_duration * fps) if max_duration else float('inf')
    if max_duration:
        print(f"Limiting processing to {max_duration} seconds ({max_frames} frames)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_folder = os.path.dirname(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(video_folder, f"{video_name}_lines.mp4")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    # Colors for visualization (random but consistent per class)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(LINE_CLASSES), 3), dtype=np.uint8)

    print("Starting inference...")
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_count >= max_frames:
            print(f"Reached max duration of {max_duration} seconds. Stopping.")
            break

        frame_count += 1

        # Preprocess
        # Resize to model input size
        frame_resized = cv2.resize(frame_bgr, (MODEL_W, MODEL_H))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(img_tensor)
            pred_coords = outputs['keypoints'][0]  # [num_classes, max_points, 2]
            pred_logits = outputs['visibility_logits'][0]  # [num_classes, max_points]

        pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
        pred_coords = pred_coords.cpu().numpy()

        end = time.perf_counter()

        # Draw predictions on original frame
        # We need to scale coordinates from [0, 1] to [width, height]

        for cls_idx, (coords, probs) in enumerate(zip(pred_coords, pred_probs)):
            # Filter points by threshold
            valid_indices = probs > threshold
            if not np.any(valid_indices):
                continue

            valid_coords = coords[valid_indices]
            valid_probs = probs[valid_indices]

            # Scale to original video resolution
            points = []
            for pt in valid_coords:
                px = int(pt[0] * width)
                py = int(pt[1] * height)
                points.append((px, py))

            color = (int(colors[cls_idx][0]), int(colors[cls_idx][1]), int(colors[cls_idx][2]))

            # Draw points
            for pt in points:
                cv2.circle(frame_bgr, pt, 5, color, -1)

            # Draw lines connecting points
            if len(points) > 1:
                # For circles, we might want to close the loop if it's a circle class
                # But generally just connecting sequential points is fine for visualization
                for i in range(len(points) - 1):
                    cv2.line(frame_bgr, points[i], points[i + 1], color, 3)

            # Draw label near the first point
            if len(points) > 0:
                label = f"{LINE_CLASSES[cls_idx]}"
                cv2.putText(
                    frame_bgr, label, (points[0][0] + 10, points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        if show_fps:
            current_fps = 1.0 / (end - start)
            cv2.putText(frame_bgr, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame_bgr)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"\nOutput saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Line Detection on Video")
    parser.add_argument("--video", type=str, default="videos/croatia_czechia.mp4", help="Path to input video file")
    parser.add_argument(
        "--model", type=str, default="models/line_detection/keypoint_detector.pth", help="Path to model checkpoint"
    )
    parser.add_argument("--duration", type=float, default=15, help="Max duration in seconds to process")
    parser.add_argument("--threshold", type=float, default=0.9, help="Confidence threshold for visualization")

    args = parser.parse_args()

    process_video(args.video, args.model, args.duration, threshold=args.threshold)
