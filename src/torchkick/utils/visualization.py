"""
Visualization utilities for drawing tracks, detections, and overlays.

This module provides classes and functions for rendering bounding boxes,
track labels, and status information on video frames.

Example:
    >>> from torchkick.utils import TrackVisualizer, ColorScheme
    >>> 
    >>> viz = TrackVisualizer(colors=ColorScheme(team_0=(255, 0, 0)))
    >>> for frame, tracks in video_with_tracks:
    ...     annotated = viz.draw_tracks(frame, tracks)
    ...     viz.draw_fps(annotated, current_fps)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel


class ColorScheme(BaseModel):
    """
    Color scheme for team and role visualization.

    Colors are in BGR format for OpenCV compatibility.

    Attributes:
        team_0: Color for team 0 (default: red).
        team_1: Color for team 1 (default: blue).
        referee: Color for referees (default: yellow).
        unknown: Color for unassigned players (default: green).
    """

    team_0: Tuple[int, int, int] = (0, 0, 255)  # Red in BGR
    team_1: Tuple[int, int, int] = (255, 0, 0)  # Blue in BGR
    referee: Tuple[int, int, int] = (0, 255, 255)  # Yellow in BGR
    unknown: Tuple[int, int, int] = (0, 255, 0)  # Green in BGR

    def get_team_color(self, team_id: int | str | None) -> Tuple[int, int, int]:
        """
        Get color for a team ID.

        Args:
            team_id: Team identifier (0, 1, 2 for referee, or other for unknown).

        Returns:
            BGR color tuple for the specified team.
        """
        if team_id == 0:
            return self.team_0
        elif team_id == 1:
            return self.team_1
        elif team_id == 2:
            return self.referee
        else:
            return self.unknown


class TrackVisualizer:
    """
    Visualizer for drawing player tracks on video frames.

    Draws bounding boxes with labels showing track ID and team assignment.
    Supports customizable colors, fonts, and label formatting.

    Args:
        colors: Color scheme for teams. Uses default if not provided.
        box_thickness: Bounding box line thickness in pixels.
        font_scale: Font size multiplier for labels.
        font_thickness: Font stroke thickness.
        label_height: Height of label background rectangle.

    Example:
        >>> viz = TrackVisualizer(box_thickness=3)
        >>> track = {"box": [100, 100, 200, 300], "id": 5, "team": 0}
        >>> viz.draw_track(frame, track)
    """

    def __init__(
        self,
        colors: Optional[ColorScheme] = None,
        box_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        label_height: int = 20,
    ) -> None:
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
        Draw a single track on the frame (in-place modification).

        Args:
            frame: BGR image array to draw on.
            track: Track dictionary with keys:
                - "box": [x1, y1, x2, y2] bounding box coordinates
                - "id": Track identifier
                - "team": (optional) Team identifier for coloring
            label_format: Format string for label. Supports {id} and {team} placeholders.
        """
        box = track["box"]
        track_id = track["id"]
        team_id = track.get("team", "?")

        x1, y1, x2, y2 = map(int, box)
        color = self.colors.get_team_color(team_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

        # Prepare label text
        label = label_format.format(id=track_id, team=team_id)
        (text_w, text_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)

        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - self.label_height),
            (x1 + text_w, y1),
            color,
            -1,
        )

        # Draw label text
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
            frame: BGR image array.
            tracks: List of track dictionaries (see draw_track for format).
            label_format: Format string for labels.

        Returns:
            New frame with tracks drawn (original unchanged).
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
        """
        Draw FPS counter on frame (in-place).

        Args:
            frame: BGR image array to draw on.
            fps: Frames per second value to display.
            position: (x, y) position for text.
        """
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
        """
        Draw status text with success/failure color.

        Args:
            frame: BGR image array to draw on.
            text: Status message to display.
            ok: If True, uses green; if False, uses red.
            position: (x, y) position for text.
        """
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
    Draw detection boxes without tracking information.

    Useful for visualizing raw detector output before tracking is applied.

    Args:
        frame: BGR image array.
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2] coordinates.
        scores: Array of shape (N,) with confidence scores.
        threshold: Minimum score to draw a box.
        color: Box color in BGR format.
        thickness: Line thickness in pixels.

    Returns:
        New frame with detection boxes drawn (original unchanged).

    Example:
        >>> boxes = np.array([[100, 100, 200, 300], [400, 200, 500, 400]])
        >>> scores = np.array([0.95, 0.72])
        >>> result = draw_detection_boxes(frame, boxes, scores, threshold=0.8)
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


__all__ = [
    "ColorScheme",
    "TrackVisualizer",
    "draw_detection_boxes",
]
