import requests
import cv2
from pathlib import Path
from collections import defaultdict
from typing import List


def upload_player_tracks(
    task_id: int,
    frame_paths: List[Path],
    detector,
    host: str = "https://app.cvat.ai",
    token: str = "",
    conf_threshold: float = 0.5,
) -> bool:
    headers = {"Authorization": f"Bearer {token}"}

    labels_response = requests.get(f"{host}/api/labels?task_id={task_id}&page_size=100", headers=headers)
    label_map = {l["name"]: l["id"] for l in labels_response.json().get("results", [])}

    tracks_data = defaultdict(list)
    class_names = detector.names

    for frame_idx, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path))
        results = detector.track(img, persist=True, conf=conf_threshold, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i].item())
                cls_id = int(boxes.cls[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                label = class_names.get(cls_id, "player")

                tracks_data[track_id].append(
                    {
                        "frame": frame_idx,
                        "label": label,
                        "points": [x1, y1, x2, y2],
                    }
                )

    tracks = []
    for track_id, detections in tracks_data.items():
        if not detections:
            continue

        label = detections[0]["label"]
        label_id = label_map.get(label)
        if label_id is None:
            continue

        shapes = [
            {
                "type": "rectangle",
                "frame": det["frame"],
                "points": det["points"],
                "occluded": False,
                "outside": False,
                "attributes": [],
            }
            for det in detections
        ]

        frames = [d["frame"] for d in detections]
        tracks.append(
            {
                "label_id": label_id,
                "frame": min(frames),
                "group": 0,
                "source": "auto",
                "shapes": shapes,
                "attributes": [],
            }
        )

    upload_response = requests.patch(
        f"{host}/api/tasks/{task_id}/annotations?action=update",
        headers={**headers, "Content-Type": "application/json"},
        json={"shapes": [], "tags": [], "tracks": tracks},
    )

    return upload_response.status_code in [200, 201]
