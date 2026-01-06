"""
IoU-based detection tracker.

This module provides a lightweight IoU-based tracker for offline
analysis. It handles detection-to-track association without team
assignment or Kalman filtering.

Example:
    >>> from torchkick.tracking import SimpleIoUTracker
    >>> 
    >>> tracker = SimpleIoUTracker(max_age=30, iou_threshold=0.3)
    >>> results = tracker.update(detections)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class SimpleIoUTracker:
    """
    Lightweight IoU-based tracker for offline analysis.

    Performs detection-to-track association using IoU matching
    with Hungarian algorithm. Team and identity assignment
    happens later in the pipeline.

    Args:
        max_age: Frames to keep unmatched tracks alive.
        iou_threshold: Minimum IoU for valid match.

    Example:
        >>> tracker = SimpleIoUTracker()
        >>> for frame_detections in video_detections:
        ...     tracks = tracker.update(frame_detections)
        ...     for track_id, box in tracks:
        ...         process_track(track_id, box)
    """

    def __init__(
        self,
        max_age: int = 30,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1

    def _iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """
        Calculate IoU between two boxes.

        Args:
            box_a: [x1, y1, x2, y2] format.
            box_b: [x1, y1, x2, y2] format.

        Returns:
            IoU value in [0, 1].
        """
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return inter / (area_a + area_b - inter + 1e-6)

    def update(
        self,
        detections: List[np.ndarray],
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Update tracks with new detections.

        Args:
            detections: List of [x1, y1, x2, y2] or [x1, y1, x2, y2, score] boxes.

        Returns:
            List of (track_id, box) tuples for active tracks.
        """
        # Convert to arrays, strip score if present
        det_boxes = [np.array(det[:4]) for det in detections]

        # Age all tracks
        for tid in self.tracks:
            self.tracks[tid]["age"] += 1

        if not det_boxes:
            self._prune_old_tracks()
            return [(tid, t["box"]) for tid, t in self.tracks.items() if t["age"] <= 1]

        if not self.tracks:
            # Create new tracks for all detections
            results = []
            for box in det_boxes:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": box, "age": 0}
                results.append((tid, box))
            return results

        # Build cost matrix
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(det_boxes)))

        for i, tid in enumerate(track_ids):
            for j, det_box in enumerate(det_boxes):
                cost_matrix[i, j] = -self._iou(self.tracks[tid]["box"], det_box)

        # Hungarian matching
        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_inds, col_inds):
            if -cost_matrix[r, c] >= self.iou_threshold:
                tid = track_ids[r]
                self.tracks[tid]["box"] = det_boxes[c]
                self.tracks[tid]["age"] = 0
                matched_tracks.add(tid)
                matched_dets.add(c)

        # Create new tracks for unmatched detections
        for j, det_box in enumerate(det_boxes):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": det_box, "age": 0}

        self._prune_old_tracks()

        return [(tid, t["box"]) for tid, t in self.tracks.items() if t["age"] <= 1]

    def _prune_old_tracks(self) -> None:
        """Remove tracks not seen for max_age frames."""
        to_remove = [tid for tid, t in self.tracks.items() if t["age"] > self.max_age]
        for tid in to_remove:
            del self.tracks[tid]

    def reset(self) -> None:
        """Clear all tracks and reset state."""
        self.tracks.clear()
        self.next_id = 1


__all__ = [
    "SimpleIoUTracker",
]
