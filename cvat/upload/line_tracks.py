import requests
import cv2
from pathlib import Path
from collections import defaultdict
from typing import List


LINE_INDEX_TO_NAME = {
    1: "Big rect. left bottom",
    2: "Big rect. left main",
    3: "Big rect. left top",
    4: "Big rect. right bottom",
    5: "Big rect. right main",
    6: "Big rect. right top",
    7: "Goal left crossbar",
    8: "Goal left post left",
    9: "Goal left post right",
    10: "Goal right crossbar",
    11: "Goal right post left",
    12: "Goal right post right",
    13: "Middle line",
    14: "Side line bottom",
    15: "Side line left",
    16: "Side line right",
    17: "Side line top",
    18: "Small rect. left bottom",
    19: "Small rect. left main",
    20: "Small rect. left top",
    21: "Small rect. right bottom",
    22: "Small rect. right main",
    23: "Small rect. right top",
}


def upload_line_tracks(
    task_id: int,
    frame_paths: List[Path],
    calibrator,
    host: str = "https://app.cvat.ai",
    token: str = "",
) -> bool:
    headers = {"Authorization": f"Bearer {token}"}

    labels_response = requests.get(f"{host}/api/labels?task_id={task_id}&page_size=100", headers=headers)
    label_map = {l["name"]: l["id"] for l in labels_response.json().get("results", [])}

    line_tracks = defaultdict(list)

    for frame_idx, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        if calibrator.cam is None:
            from models.pitch.utils.calib import FramebyFrameCalib

            calibrator.cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)

        calibrator.inference(img)
        lines_dict = calibrator.cam.lines_dict

        for line_idx, line_data in lines_dict.items():
            line_name = LINE_INDEX_TO_NAME.get(line_idx)
            if line_name is None:
                continue

            if "x_1" in line_data and "x_2" in line_data:
                line_tracks[line_name].append(
                    {
                        "frame": frame_idx,
                        "points": [line_data["x_1"], line_data["y_1"], line_data["x_2"], line_data["y_2"]],
                    }
                )

    tracks = []
    for line_name, detections in line_tracks.items():
        if not detections:
            continue

        label_id = label_map.get(line_name)
        if label_id is None:
            continue

        shapes = [
            {
                "type": "polyline",
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
