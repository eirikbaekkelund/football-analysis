import cv2
import numpy as np
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple


class ColorScheme(BaseModel):
    team_0: Tuple[int, int, int] = (0, 0, 255)
    team_1: Tuple[int, int, int] = (255, 0, 0)
    referee: Tuple[int, int, int] = (0, 255, 255)
    unknown: Tuple[int, int, int] = (0, 255, 0)

    def get_team_color(self, team_id) -> Tuple[int, int, int]:
        if team_id == 0:
            return self.team_0
        elif team_id == 1:
            return self.team_1
        elif team_id == 2:
            return self.referee
        else:
            return self.unknown


class TrackVisualizer:
    def __init__(
        self,
        colors: Optional[ColorScheme] = None,
        box_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        label_height: int = 20,
    ):
        """
        Args:
            colors: Color scheme for teams
            box_thickness: Bounding box line thickness
            font_scale: Label font scale
            font_thickness: Label font thickness
            label_height: Height of label background
        """
        self.colors = colors or ColorScheme()
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.label_height = label_height
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_track(
        self,
        frame: np.ndarray,
        track: Dict,
        label_format: str = "ID:{id} T:{team}",
    ) -> None:
        """
        Draw a single track on the frame (in-place).

        Args:
            frame: BGR image
            track: Dict with 'box', 'id', and optionally 'team'
            label_format: Format string for label (supports {id}, {team})
        """
        box = track['box']
        track_id = track['id']
        team_id = track.get('team', '?')

        x1, y1, x2, y2 = map(int, box)
        color = self.colors.get_team_color(team_id)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

        # Draw label
        label = label_format.format(id=track_id, team=team_id)
        (text_w, text_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)

        # Label background
        cv2.rectangle(
            frame,
            (x1, y1 - self.label_height),
            (x1 + text_w, y1),
            color,
            -1,
        )

        # Label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness,
        )

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        label_format: str = "ID:{id} T:{team}",
    ) -> np.ndarray:
        """
        Draw all tracks on a copy of the frame.

        Args:
            frame: BGR image
            tracks: List of track dicts
            label_format: Format string for labels

        Returns:
            Frame with tracks drawn
        """
        result = frame.copy()
        for track in tracks:
            self.draw_track(result, track, label_format)
        return result

    def draw_fps(
        self,
        frame: np.ndarray,
        fps: float,
        position: Tuple[int, int] = (10, 30),
    ) -> None:
        """Draw FPS counter on frame (in-place)."""
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            position,
            self.font,
            1.0,
            (0, 255, 0),
            2,
        )

    def draw_status(
        self,
        frame: np.ndarray,
        text: str,
        ok: bool = True,
        position: Tuple[int, int] = (10, 30),
    ) -> None:
        """Draw status text with color indicating success/failure."""
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.putText(frame, text, position, self.font, 0.7, color, 2)


def draw_detection_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection boxes without tracking info.

    Args:
        frame: BGR image
        boxes: Nx4 array of [x1, y1, x2, y2]
        scores: N array of confidence scores
        threshold: Minimum score to draw
        color: Box color (BGR)
        thickness: Line thickness

    Returns:
        Frame with boxes drawn
    """
    result = frame.copy()
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            result,
            f"{score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return result
