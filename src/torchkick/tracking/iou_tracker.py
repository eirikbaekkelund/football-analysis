"""
Mask IoU tracker for SAM3-based segmentation tracking.

Used exclusively with SAM3 video predictor output where detections are
binary segmentation masks rather than bounding boxes.  For box-based
player tracking use ByteTracker (BoT-SORT backend).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class MaskIoUTracker:
    """IoU-based tracker that matches binary masks between frames.

    Detections should be provided as a list of binary 2D numpy arrays
    (H, W). Tracks store the most recent mask for IoU computation.
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.5) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1

    def _iou_mask(self, a: np.ndarray, b: np.ndarray) -> float:
        a_bool = (a > 0).astype(np.uint8)
        b_bool = (b > 0).astype(np.uint8)
        inter = (a_bool & b_bool).sum()
        union = (a_bool | b_bool).sum()
        if union == 0:
            return 0.0
        return float(inter) / float(union)

    def update(self, detections: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        """Update tracks with a list of binary mask detections.

        Returns list of (track_id, mask) for active tracks this frame.
        """
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1

        if not detections:
            self._prune_old_tracks()
            return [(tid, t["mask"]) for tid, t in self.tracks.items() if t["age"] <= 1]

        if not self.tracks:
            results = []
            for mask in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"mask": mask, "age": 0}
                results.append((tid, mask))
            return results

        from scipy.optimize import linear_sum_assignment

        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, tid in enumerate(track_ids):
            for j, det_mask in enumerate(detections):
                cost_matrix[i, j] = -self._iou_mask(self.tracks[tid]["mask"], det_mask)

        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        matched_dets = set()
        for r, c in zip(row_inds, col_inds):
            if -cost_matrix[r, c] >= self.iou_threshold:
                tid = track_ids[r]
                self.tracks[tid]["mask"] = detections[c]
                self.tracks[tid]["age"] = 0
                matched_dets.add(c)

        for j, det_mask in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"mask": det_mask, "age": 0}

        self._prune_old_tracks()
        return [(tid, t["mask"]) for tid, t in self.tracks.items() if t["age"] <= 1]

    def _prune_old_tracks(self) -> None:
        to_remove = [tid for tid, t in self.tracks.items() if t["age"] > self.max_age]
        for tid in to_remove:
            del self.tracks[tid]

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1


__all__ = ["MaskIoUTracker"]
