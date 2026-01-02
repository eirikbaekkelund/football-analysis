import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from examples.match_analysis import (
    load_models,
    detect_and_project,
    smooth_trajectories,
    assign_identities,
)
from cvat.cvat_integration import Client


class TrackAnnotation(BaseModel):
    track_id: int
    label: str
    frames: List[int]
    boxes: List[Tuple[float, float, float, float]]
    attributes: Dict[str, str]


def process_video_for_cvat(
    video_path: str,
    max_duration: Optional[float] = None,
    model_type: str = "fcnn",
    model_path: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[List[TrackAnnotation], Dict]:
    """
    Process a video using the match_analysis pipeline and extract annotations.

    Args:
        video_path: Path to input video
        max_duration: Maximum duration in seconds
        model_type: "yolo" or "fcnn"
        model_path: Path to detection model (auto-detected if None)
        device: "cuda" or "cpu"

    Returns:
        Tuple of (annotations list, metadata dict)
    """
    import torch

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Set default model path
    if model_path is None:
        if model_type == "yolo":
            model_path = "yolo11n.pt"
        elif model_type == "fcnn":
            model_path = "models/player/fcnn/fcnn_player_tracker.pth"

    print(f"[PreLabeler] Processing {video_path}")
    print(f"[PreLabeler] Model: {model_type}, Device: {device}")

    # Load models
    detector, pnl_calib = load_models(device, model_path, model_type)

    store = detect_and_project(
        video_path,
        detector,
        pnl_calib,
        model_type,
        max_duration=max_duration,
        homography_interval=1,
        device=device,
    )

    smooth_trajectories(store)

    assignments, slot_manager = assign_identities(store)

    # Convert to TrackAnnotation objects
    annotations = []

    for slot_key, slot in slot_manager.slots.items():
        if not slot.assigned_track_ids:
            continue

        # Gather all observations for this slot
        frames = []
        boxes = []

        for tid in slot.assigned_track_ids:
            track = store.get_track(tid)
            if track:
                for obs in track.observations:
                    frames.append(obs.frame_idx)
                    boxes.append(tuple(obs.box))

        if not frames:
            continue

        # Sort by frame
        sorted_data = sorted(zip(frames, boxes), key=lambda x: x[0])
        frames = [f for f, _ in sorted_data]
        boxes = [b for _, b in sorted_data]

        # Determine label and attributes
        if slot.team == -1:
            label = "referee"
            attributes = {"role": "main" if "REF" in slot_key else "assistant"}
        elif slot.is_goalie:
            label = "goalkeeper"
            attributes = {"team": f"team_{'a' if slot.team == 0 else 'b'}"}
        else:
            label = "player"
            attributes = {
                "team": f"team_{'a' if slot.team == 0 else 'b'}",
                "role": "outfield",
            }

        annotations.append(
            TrackAnnotation(
                track_id=slot.slot_id,
                label=label,
                frames=frames,
                boxes=boxes,
                attributes=attributes,
            )
        )

    # Metadata
    metadata = {
        "fps": store.fps,
        "total_frames": store.total_frames,
        "source": video_path,
        "model_type": model_type,
        "num_tracks": len(store.tracks),
        "num_slots": len(annotations),
    }

    print(f"[PreLabeler] Generated {len(annotations)} constrained tracks (22+1 slots)")
    return annotations, metadata


def upload_to_cvat(
    task_id: int,
    annotations: List[TrackAnnotation],
    host: str = "https://app.cvat.ai",
    token: str = "",
) -> bool:
    """
    Upload pre-labeled annotations to a CVAT task.

    Args:
        task_id: CVAT task ID
        annotations: List of TrackAnnotation objects
        host: CVAT host URL
        token: API token

    Returns:
        True if upload successful
    """
    headers = {"Authorization": f"Bearer {token}"}

    # Get label mapping
    labels_response = requests.get(f"{host}/api/labels?task_id={task_id}&page_size=100", headers=headers)
    label_map = {l["name"]: l["id"] for l in labels_response.json().get("results", [])}

    tracks = []

    for ann in annotations:
        label_id = label_map.get(ann.label)
        if label_id is None:
            print(f"[Warning] Label '{ann.label}' not found in CVAT task")
            continue

        shapes = []
        for frame, box in zip(ann.frames, ann.boxes):
            shapes.append(
                {
                    "type": "rectangle",
                    "frame": frame,
                    "points": list(box),
                    "occluded": False,
                    "outside": False,
                    "attributes": [],
                }
            )

        tracks.append(
            {
                "label_id": label_id,
                "frame": min(ann.frames),
                "group": 0,
                "source": "auto",
                "shapes": shapes,
                "attributes": [],
            }
        )

    print(f"[Upload] Uploading {len(tracks)} tracks to task {task_id}")

    upload_response = requests.patch(
        f"{host}/api/tasks/{task_id}/annotations?action=update",
        headers={**headers, "Content-Type": "application/json"},
        json={"shapes": [], "tags": [], "tracks": tracks},
    )

    success = upload_response.status_code in [200, 201]
    if success:
        print(f"[Upload] Successfully uploaded annotations")
    else:
        print(f"[Upload] Failed: {upload_response.status_code} - {upload_response.text}")

    return success


def export_to_mot_format(
    annotations: List[TrackAnnotation],
    output_path: str,
) -> None:
    """
    Export annotations to MOT format.
    Format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ann in annotations:
            class_id = {"player": 1, "goalkeeper": 2, "referee": 3}.get(ann.label, 0)
            for frame, box in zip(ann.frames, ann.boxes):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                f.write(f"{frame},{ann.track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,{class_id},1\n")

    print(f"[Export] Saved MOT format to {output_path}")


def to_cvat_annotations(annotations: List[TrackAnnotation]) -> List:
    """
    Convert TrackAnnotation objects to CVAT Annotation objects.

    Returns list of cvat.cvat_integration.Annotation objects ready for upload.
    """
    from cvat.cvat_integration import Annotation

    cvat_annotations = []
    for ann in annotations:
        for frame, box in zip(ann.frames, ann.boxes):
            x1, y1, x2, y2 = box
            cvat_annotations.append(
                Annotation(
                    frame=frame,
                    label=ann.label,
                    xtl=float(x1),
                    ytl=float(y1),
                    xbr=float(x2),
                    ybr=float(y2),
                    occluded=False,
                    attributes=ann.attributes,
                    track_id=ann.track_id,
                )
            )

    return cvat_annotations


def upload_with_client(
    client: "Client",
    task_id: int,
    annotations: List[TrackAnnotation],
) -> None:
    """
    Upload annotations using the existing CVAT Client.

    Args:
        client: Authenticated CVAT Client instance
        task_id: Target task ID
        annotations: List of TrackAnnotation objects
    """
    cvat_annotations = to_cvat_annotations(annotations)
    client.upload_annotations(task_id, cvat_annotations)
    print(f"[Upload] Uploaded {len(cvat_annotations)} annotations to task {task_id}")


def upload_tracks_with_client(
    client: "Client",
    task_id: int,
    annotations: List[TrackAnnotation],
    keyframe_interval: int = 15,
    total_frames: Optional[int] = None,
) -> bool:
    """
    Upload as tracking annotations (maintains track IDs across frames).

    Uses the CVAT tracks API for proper multi-frame object tracking.
    All tracks span the full video duration with keyframes at regular intervals.

    Args:
        client: Authenticated CVAT Client instance
        task_id: Target task ID
        annotations: List of TrackAnnotation objects
        keyframe_interval: Set keyframes every N frames (default: 15 = 0.5s at 30fps)
        total_frames: Total frames in video (auto-detected from task if None)

    Returns:
        True if upload successful
    """
    # Get label mapping and task info
    labels_response = client._request("GET", f"labels?task_id={task_id}&page_size=100")
    label_map = {l["name"]: l["id"] for l in labels_response.json().get("results", [])}

    # Get total frames from task if not provided
    if total_frames is None:
        task_response = client._request("GET", f"tasks/{task_id}")
        total_frames = task_response.json().get("size", 0)

    print(f"[Upload] Label mapping: {label_map}")
    print(f"[Upload] Total frames: {total_frames}, keyframe interval: {keyframe_interval}")

    tracks = []
    for ann in annotations:
        label_id = label_map.get(ann.label)
        if label_id is None:
            print(f"[Warning] Label '{ann.label}' not found in CVAT task, skipping")
            continue

        # Build frame->box lookup for this track
        frame_to_box = {int(f): box for f, box in zip(ann.frames, ann.boxes)}

        # Get the range of frames this track covers
        min_frame = min(ann.frames)
        max_frame = max(ann.frames)

        # Get first and last known boxes for extrapolation
        first_known_box = ann.boxes[0]  # Box at min_frame
        last_known_box = ann.boxes[-1]  # Box at max_frame

        # Extend to full video duration (all tracks start at 0)
        track_start = 0
        track_end = total_frames - 1

        shapes = []
        current_box = first_known_box

        for frame in range(track_start, track_end + 1):
            # include keyframes (every N-th frame) plus first/last
            is_keyframe = (frame % keyframe_interval == 0) or frame == track_start or frame == track_end

            if not is_keyframe:
                continue

            if frame in frame_to_box:
                box = frame_to_box[frame]
                current_box = box
                outside = False
            elif frame < min_frame:
                box = first_known_box
                outside = True
            elif frame > max_frame:
                box = last_known_box
                outside = True
            else:
                box = current_box
                outside = True

            shapes.append(
                {
                    "type": "rectangle",
                    "frame": int(frame),
                    "points": [float(v) for v in box],
                    "occluded": False,
                    "outside": outside,
                    "keyframe": True,
                    "attributes": [],
                }
            )

        if not shapes:
            continue

        tracks.append(
            {
                "label_id": label_id,
                "frame": int(shapes[0]["frame"]),
                "group": 0,
                "source": "auto",
                "shapes": shapes,
                "attributes": [],
            }
        )

    print(f"[Upload] Uploading {len(tracks)} tracks to task {task_id}")

    # Use session.patch directly to get better error info
    response = client.session.patch(
        f"{client.credentials.base_url}/tasks/{task_id}/annotations?action=update",
        json={"shapes": [], "tags": [], "tracks": tracks},
    )

    if response.status_code not in [200, 201]:
        print(f"[Upload] Error {response.status_code}: {response.text[:500]}")
        response.raise_for_status()

    print(f"[Upload] Successfully uploaded {len(tracks)} tracks")
    return True


def prelabel_task(
    client: "Client",
    task_id: int,
    video_path: str,
    max_duration: Optional[float] = None,
    model_type: str = "fcnn",
    use_tracks: bool = True,
) -> Tuple[List[TrackAnnotation], Dict]:
    """
    Complete pipeline: process video and upload pre-labels to CVAT task.

    Args:
        client: Authenticated CVAT Client
        task_id: Target CVAT task ID
        video_path: Path to video file
        max_duration: Maximum duration to process (seconds)
        model_type: "yolo" or "fcnn"
        use_tracks: If True, upload as tracks (recommended). If False, upload as shapes.

    Returns:
        Tuple of (annotations, metadata)

    Example:
        from cvat import Client, Credentials

        creds = Credentials(host="https://app.cvat.ai", username="", password="token", use_token=True)
        client = Client(creds)

        annotations, meta = prelabel_task(client, task_id=123, video_path="match.mp4")
    """
    annotations, metadata = process_video_for_cvat(
        video_path,
        max_duration=max_duration,
        model_type=model_type,
    )

    upload_tracks_with_client(client, task_id, annotations)

    return annotations, metadata


def batch_process_videos(
    video_paths: List[str],
    output_dir: str,
    max_duration: Optional[float] = None,
    model_type: str = "fcnn",
    upload_to_cvat_task: Optional[int] = None,
    cvat_host: str = "https://app.cvat.ai",
    cvat_token: str = "",
) -> Dict[str, List[TrackAnnotation]]:
    """
    Process multiple videos and generate pre-labeled annotations.

    Args:
        video_paths: List of video paths
        output_dir: Output directory for annotations
        max_duration: Max duration per video
        model_type: "yolo" or "fcnn"
        upload_to_cvat_task: If provided, upload to this CVAT task
        cvat_host: CVAT host URL
        cvat_token: CVAT API token

    Returns:
        Dict mapping video names to annotation lists
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for video_path in video_paths:
        video_name = Path(video_path).stem
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        print("=" * 60)

        try:
            annotations, metadata = process_video_for_cvat(
                video_path,
                max_duration=max_duration,
                model_type=model_type,
            )
            upload_to_cvat(
                upload_to_cvat_task,
                annotations,
                host=cvat_host,
                token=cvat_token,
            )

            all_results[video_name] = annotations

            # Summary
            n_players = len([a for a in annotations if a.label == "player"])
            n_goalies = len([a for a in annotations if a.label == "goalkeeper"])
            n_refs = len([a for a in annotations if a.label == "referee"])

            print(f"\nSummary for {video_name}:")
            print(f"  Players: {n_players}")
            print(f"  Goalkeepers: {n_goalies}")
            print(f"  Referees: {n_refs}")

        except Exception as e:
            print(f"[Error] Failed to process {video_path}: {e}")
            import traceback

            traceback.print_exc()

    return all_results
