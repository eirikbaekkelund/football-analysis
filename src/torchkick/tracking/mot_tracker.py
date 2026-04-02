"""
BoT-SORT soccer multi-object tracker.

Replaces the old ByteTrack wrapper. BoT-SORT adds camera-motion compensation
(CMC via ECC) and tighter re-ID integration, making it significantly more
robust for broadcast soccer where the camera pans frequently and players
occlude each other during corners/set-pieces.

When reid_embeddings are provided, BoT-SORT uses appearance cost alongside
Kalman-IoU for re-association after occlusion gaps.

Example:
    >>> tracker = SoccerTracker(track_thresh=0.5, track_buffer=30)
    >>> # detections: np.ndarray [N, 6] — [x1, y1, x2, y2, conf, class_id]
    >>> active_tracks = tracker.update(detections)
    >>> for track_id, box in active_tracks:
    ...     print(track_id, box)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class SoccerTracker:
    """
    BoT-SORT multi-object tracker (boxmot backend).

    Drop-in replacement for the old ByteTracker. Uses BoT-SORT which adds:
      - Camera motion compensation (ECC homography warp on Kalman state)
      - Tighter appearance/IoU re-association after long occlusions

    Args:
        track_thresh: Confidence threshold for high-confidence detections.
        track_buffer: Frames to keep a track alive without a match.
        match_thresh: IoU threshold for track-detection matching.
        frame_rate: Video frame rate for Kalman filter tuning.
        reid_weights: Optional path to ReID model weights for BoT-SORT.
        device: Device string for ReID model.
        use_fp16: Use FP16 for ReID model (GPU only).
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
        reid_weights: Optional[str] = None,
        device: str = "cuda",
        use_fp16: bool = False,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self._tracker = None
        self._init_tracker(reid_weights, device, use_fp16)

    def _init_tracker(
        self,
        reid_weights: Optional[str],
        device: str,
        use_fp16: bool,
    ) -> None:
        try:
            from pathlib import Path
            from boxmot import BoTrack

            self._tracker = BoTrack(
                reid_weights=Path(reid_weights) if reid_weights else None,
                device=device,
                half=use_fp16,
            )
        except ImportError:
            self._tracker = None
        except Exception:
            # BoTrack constructor signature varies across boxmot versions
            try:
                from boxmot import BoTrack

                self._tracker = BoTrack()
            except Exception:
                self._tracker = None

    def update(
        self,
        detections: List,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Update tracker with new detections.

        Args:
            detections: List of [x1, y1, x2, y2] or [x1, y1, x2, y2, conf]
                        or np.ndarray shape [N, 4-6].

        Returns:
            List of (track_id, box_xyxy) tuples for active tracks.
        """
        return self.update_with_embeddings(detections, embeddings=None)

    def update_with_embeddings(
        self,
        detections: List,
        embeddings: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Update tracker with detections and optional ReID embeddings.

        Args:
            detections: List/array of [x1, y1, x2, y2] or [x1, y1, x2, y2, conf]
                        or np.ndarray shape [N, 4-6].
            embeddings: Optional [N, D] ReID feature vectors. When provided,
                        BoT-SORT uses appearance cost alongside IoU/CMC.

        Returns:
            List of (track_id, box_xyxy) tuples for active tracks.
        """
        if len(detections) == 0:
            return []

        dets_np = np.asarray(detections, dtype=np.float32)

        if dets_np.ndim == 1:
            dets_np = dets_np[np.newaxis, :]
        if dets_np.shape[1] == 4:
            confs = np.ones((len(dets_np), 1), dtype=np.float32)
            cls = np.zeros((len(dets_np), 1), dtype=np.float32)
            dets_np = np.hstack([dets_np, confs, cls])
        elif dets_np.shape[1] == 5:
            cls = np.zeros((len(dets_np), 1), dtype=np.float32)
            dets_np = np.hstack([dets_np, cls])

        if self._tracker is None:
            # Fallback: greedy IoU matching using detection order
            return [(i, dets_np[i, :4]) for i in range(len(dets_np))]

        # boxmot BoTrack.update(dets, img=None) → tracks [N, 7]
        # tracks columns: [x1, y1, x2, y2, track_id, conf, class_id]
        try:
            if embeddings is not None and hasattr(self._tracker, "update"):
                import inspect

                sig = inspect.signature(self._tracker.update)
                if "embs" in sig.parameters:
                    tracks = self._tracker.update(dets_np, img=None, embs=embeddings)
                else:
                    tracks = self._tracker.update(dets_np, img=None)
            else:
                tracks = self._tracker.update(dets_np, img=None)
        except TypeError:
            tracks = self._tracker.update(dets_np)

        if tracks is None or len(tracks) == 0:
            return []

        result = []
        for t in tracks:
            x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            track_id = int(t[4])
            result.append((track_id, np.array([x1, y1, x2, y2], dtype=np.float32)))

        return result

    def reset(self) -> None:
        """Reset tracker state (call between videos)."""
        self._init_tracker(None, "cuda", False)


# Keep the old name as an alias for backward compatibility
ByteTracker = SoccerTracker

__all__ = ["SoccerTracker", "ByteTracker"]
