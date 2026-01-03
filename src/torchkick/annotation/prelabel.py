"""
Pre-labeling pipeline for automated annotation.

This module provides tools for running detection models on videos,
generating track annotations, and uploading pre-labels to CVAT for
human review and correction.

Example:
    >>> from torchkick.annotation import prelabel_video
    >>> 
    >>> annotations = prelabel_video(
    ...     video_path="match.mp4",
    ...     detector=yolo_model,
    ...     tracker=tracker,
    ... )
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2

from torchkick.annotation.labels import LINE_INDEX_TO_NAME
from torchkick.annotation.models import Annotation, TrackAnnotation


def process_video_for_cvat(
    video_path: str,
    detector: Any,
    tracker: Optional[Any] = None,
    frame_step: int = 1,
    max_frames: Optional[int] = None,
    confidence_threshold: float = 0.3,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[TrackAnnotation]:
    """
    Process video with detector and tracker to generate pre-labels.

    Runs detection on each frame, optionally links detections into
    tracks, and returns annotations ready for CVAT upload.

    Args:
        video_path: Path to input video.
        detector: Detection model with predict() method.
        tracker: Optional tracker for linking detections.
        frame_step: Process every Nth frame.
        max_frames: Maximum frames to process.
        confidence_threshold: Minimum detection confidence.
        progress_callback: Optional callback(current, total) for progress.

    Returns:
        List of TrackAnnotation objects.

    Example:
        >>> from ultralytics import YOLO
        >>> detector = YOLO("yolov8n.pt")
        >>> tracks = process_video_for_cvat("match.mp4", detector)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracks: Dict[int, TrackAnnotation] = {}
    next_track_id = 1
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx % frame_step == 0:
                if progress_callback:
                    progress_callback(frame_idx, min(total_frames, max_frames or total_frames))

                # Run detection
                results = detector.predict(frame, verbose=False)
                detections = _parse_detections(results, confidence_threshold)

                if tracker:
                    # Link to existing tracks
                    track_ids = tracker.update(detections, frame)
                    for det, track_id in zip(detections, track_ids):
                        if track_id not in tracks:
                            tracks[track_id] = TrackAnnotation(
                                track_id=track_id,
                                label=det["label"],
                                frames=[],
                                boxes=[],
                            )
                        tracks[track_id].frames.append(frame_idx)
                        tracks[track_id].boxes.append(det["box"])
                else:
                    # Create single-frame "tracks"
                    for det in detections:
                        tracks[next_track_id] = TrackAnnotation(
                            track_id=next_track_id,
                            label=det["label"],
                            frames=[frame_idx],
                            boxes=[det["box"]],
                        )
                        next_track_id += 1

            frame_idx += 1

    finally:
        cap.release()

    return list(tracks.values())


def _parse_detections(
    results: Any,
    confidence_threshold: float,
) -> List[Dict[str, Any]]:
    """Parse detection results from YOLO or similar model."""
    detections = []

    # Handle YOLO results
    if hasattr(results, "__iter__"):
        for result in results:
            if hasattr(result, "boxes"):
                boxes = result.boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    if conf < confidence_threshold:
                        continue

                    xyxy = boxes.xyxy[i].cpu().numpy()
                    cls = int(boxes.cls[i])

                    # Map class index to label name
                    label = result.names.get(cls, f"class_{cls}")

                    detections.append(
                        {
                            "box": [float(x) for x in xyxy],  # [x1, y1, x2, y2]
                            "label": label,
                            "confidence": conf,
                            "class_id": cls,
                        }
                    )

    return detections


def upload_player_tracks(
    client: Any,
    task_id: int,
    tracks: List[TrackAnnotation],
    label_name: str = "Player",
) -> None:
    """
    Upload player track annotations to CVAT task.

    Converts TrackAnnotation objects to CVAT annotation format
    and uploads them for review.

    Args:
        client: Authenticated CVATClient.
        task_id: Target CVAT task ID.
        tracks: List of player track annotations.
        label_name: CVAT label name for players.
    """
    annotations = []

    for track in tracks:
        if track.label.lower() not in ("player", "person"):
            continue

        for frame, box in zip(track.frames, track.boxes):
            annotations.append(
                Annotation(
                    frame=frame,
                    label=label_name,
                    xtl=box[0],
                    ytl=box[1],
                    xbr=box[2],
                    ybr=box[3],
                    attributes={"track_id": str(track.track_id)},
                )
            )

    if annotations:
        client.upload_annotations(task_id, annotations)
        print(f"Uploaded {len(annotations)} player annotations to task {task_id}")


def upload_line_tracks(
    client: Any,
    task_id: int,
    line_detections: List[Dict[str, Any]],
) -> None:
    """
    Upload pitch line annotations to CVAT task.

    Args:
        client: Authenticated CVATClient.
        task_id: Target CVAT task ID.
        line_detections: List of line detection dicts with
            'frame', 'line_idx', 'points' keys.
    """
    annotations = []

    for det in line_detections:
        frame = det["frame"]
        line_idx = det["line_idx"]
        points = det.get("points", [])

        label = LINE_INDEX_TO_NAME.get(line_idx, f"line_{line_idx}")

        if not points:
            continue

        # Convert points to bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        annotations.append(
            Annotation(
                frame=frame,
                label=label,
                xtl=min(xs),
                ytl=min(ys),
                xbr=max(xs),
                ybr=max(ys),
                attributes={
                    "line_idx": str(line_idx),
                    "num_points": str(len(points)),
                },
            )
        )

    if annotations:
        client.upload_annotations(task_id, annotations)
        print(f"Uploaded {len(annotations)} line annotations to task {task_id}")


def prelabel_task(
    client: Any,
    task_id: int,
    detector: Any,
    tracker: Optional[Any] = None,
    frame_step: int = 1,
) -> Dict[str, Any]:
    """
    Download frames from CVAT task, run detection, and upload pre-labels.

    Complete workflow for model-assisted labeling on an existing task.

    Args:
        client: Authenticated CVATClient.
        task_id: CVAT task ID with uploaded images/video.
        detector: Detection model.
        tracker: Optional tracker.
        frame_step: Process every Nth frame.

    Returns:
        Dict with annotation statistics.
    """
    # Get task info
    task = client.get_task(task_id)

    # Download frames
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        frames_dir.mkdir()

        # Download frame data
        response = client._request("GET", f"tasks/{task_id}/data", params={"type": "chunk"})

        # Process frames
        tracks = []
        frame_idx = 0

        for frame_path in sorted(frames_dir.glob("*")):
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            results = detector.predict(frame, verbose=False)
            detections = _parse_detections(results, confidence_threshold=0.3)

            for det in detections:
                tracks.append(
                    TrackAnnotation(
                        track_id=len(tracks) + 1,
                        label=det["label"],
                        frames=[frame_idx],
                        boxes=[det["box"]],
                    )
                )

            frame_idx += 1

        # Upload annotations
        if tracks:
            upload_player_tracks(client, task_id, tracks)

    return {
        "task_id": task_id,
        "tracks_uploaded": len(tracks),
    }


def batch_process_videos(
    video_paths: List[str],
    detector: Any,
    tracker: Optional[Any] = None,
    output_dir: str = "./prelabels",
    **kwargs: Any,
) -> Dict[str, List[TrackAnnotation]]:
    """
    Process multiple videos and save pre-labels.

    Args:
        video_paths: List of video file paths.
        detector: Detection model.
        tracker: Optional tracker.
        output_dir: Directory for output JSON files.
        **kwargs: Additional arguments for process_video_for_cvat.

    Returns:
        Dict mapping video names to their track annotations.
    """
    import json

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for video_path in video_paths:
        video_name = Path(video_path).stem
        print(f"Processing {video_name}...")

        tracks = process_video_for_cvat(video_path, detector, tracker, **kwargs)
        results[video_name] = tracks

        # Save to JSON
        out_file = out_dir / f"{video_name}_tracks.json"
        with open(out_file, "w") as f:
            json.dump(
                [t.model_dump() for t in tracks],
                f,
                indent=2,
            )

        print(f"  Saved {len(tracks)} tracks to {out_file}")

    return results


__all__ = [
    "process_video_for_cvat",
    "upload_player_tracks",
    "upload_line_tracks",
    "prelabel_task",
    "batch_process_videos",
]
