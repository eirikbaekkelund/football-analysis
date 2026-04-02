# Copyright (c) Meta Platforms, Inc. and affiliates.
# MLX port of sam3/model/sam3_tracking_predictor.py
#
# The "tracking predictor" wraps both the SAM 3 detector (for initialising
# tracks from text/exemplar prompts) and the SAM 2-style tracker (for
# propagating masks through time).  It exposes the handle_request() session
# API used by the SAM 3 video demo.

import uuid
from typing import Dict
import numpy as np

from torchkick.sam_3._mlx.tracker.video_predictor import Sam3VideoPredictor


# ------------------------------------------------------------------ #
# Session state                                                        #
# ------------------------------------------------------------------ #


class TrackingSession:
    """Holds all state for one video tracking session."""

    def __init__(self, session_id: str, inference_state: dict):
        self.session_id = session_id
        self.inference_state = inference_state
        self.next_obj_id = 1
        self.detector_outputs: Dict[int, dict] = {}  # frame -> detector out
        self.is_running = False

    def new_obj_id(self) -> int:
        oid = self.next_obj_id
        self.next_obj_id += 1
        return oid


# ------------------------------------------------------------------ #
# Main tracking predictor                                              #
# ------------------------------------------------------------------ #


class Sam3TrackerPredictor:
    """
    High-level SAM 3 tracking predictor.

    Wraps a Sam3VideoPredictor (tracker) and a detector model and exposes a
    request/response API compatible with the original PyTorch implementation:

        response = predictor.handle_request(request=dict(type="start_session", ...))

    Supported request types:
      - start_session    : load a video, return session_id
      - reset_session    : clear all objects but keep the video loaded
      - add_prompt       : add point/box/mask/text prompt for an object
      - propagate        : run forward (or backward) propagation
      - get_masks        : retrieve cached mask for a specific frame
      - close_session    : free session resources
    """

    def __init__(
        self,
        tracker: Sam3VideoPredictor,
        detector=None,  # optional Sam3 detector (not ported here)
        max_concurrent_sessions: int = 4,
    ):
        self.tracker = tracker
        self.detector = detector
        self.sessions: Dict[str, TrackingSession] = {}
        self.max_concurrent_sessions = max_concurrent_sessions

    # ---------------------------------------------------------------- #
    # Request dispatcher                                                 #
    # ---------------------------------------------------------------- #

    def handle_request(self, request: dict) -> dict:
        """
        Dispatch a request dict to the appropriate handler.
        Returns a response dict.
        """
        req_type = request.get("type")
        dispatch = {
            "start_session": self._handle_start_session,
            "reset_session": self._handle_reset_session,
            "add_prompt": self._handle_add_prompt,
            "propagate": self._handle_propagate,
            "get_masks": self._handle_get_masks,
            "close_session": self._handle_close_session,
        }
        handler = dispatch.get(req_type)
        if handler is None:
            return {"error": f"Unknown request type: {req_type}"}
        try:
            return handler(request)
        except Exception as exc:
            return {"error": str(exc), "request_type": req_type}

    # ---------------------------------------------------------------- #
    # Session lifecycle                                                  #
    # ---------------------------------------------------------------- #

    def _handle_start_session(self, request: dict) -> dict:
        if len(self.sessions) >= self.max_concurrent_sessions:
            # Evict the oldest session
            oldest = next(iter(self.sessions))
            del self.sessions[oldest]

        resource_path = request.get("resource_path")
        if resource_path is None:
            raise ValueError("start_session requires 'resource_path'")

        inference_state = self.tracker.init_state(
            video_path=resource_path,
            offload_video_to_cpu=request.get("offload_video_to_cpu", False),
        )
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = TrackingSession(session_id, inference_state)

        return {
            "session_id": session_id,
            "num_frames": inference_state["num_frames"],
        }

    def _handle_reset_session(self, request: dict) -> dict:
        session = self._get_session(request)
        self.tracker.reset_state(session.inference_state)
        session.next_obj_id = 1
        session.detector_outputs.clear()
        session.is_running = False
        return {"status": "ok", "session_id": session.session_id}

    def _handle_close_session(self, request: dict) -> dict:
        session_id = request.get("session_id", "")
        self.sessions.pop(session_id, None)
        return {"status": "ok"}

    # ---------------------------------------------------------------- #
    # Prompting                                                          #
    # ---------------------------------------------------------------- #

    def _handle_add_prompt(self, request: dict) -> dict:
        """
        Add a prompt for one object at one frame.

        Accepted prompt keys (all optional, at least one required):
          points (list of [x,y]),  labels (list of int 0/1),
          box    ([x1,y1,x2,y2]),
          mask   (H×W float/bool array),
          text   (str, requires self.detector to be set)
        """
        session = self._get_session(request)
        state = session.inference_state
        frame_idx = int(request.get("frame_index", 0))

        # Resolve / allocate object id
        obj_id = request.get("obj_id")
        if obj_id is None:
            obj_id = session.new_obj_id()

        # ---- text prompt → run detector to get box/mask init ----
        text = request.get("text")
        if text is not None:
            if self.detector is None:
                raise RuntimeError(
                    "Text prompts require a detector model. " "Pass detector= to Sam3TrackerPredictor.__init__."
                )
            det_out = self._run_detector(session, frame_idx, text, request)
            # Store detector output
            session.detector_outputs.setdefault(frame_idx, {})[obj_id] = det_out

            # Convert detector boxes/masks to tracker prompts
            boxes = det_out.get("boxes")  # (K, 4) np float
            det_masks = det_out.get("masks")  # (K, H, W) bool
            outputs = []
            for k in range(len(boxes) if boxes is not None else 0):
                oid_k = obj_id if k == 0 else session.new_obj_id()
                self.tracker.add_new_points_or_box(
                    state,
                    frame_idx=frame_idx,
                    obj_id=oid_k,
                    box=boxes[k] if boxes is not None else None,
                )
                if det_masks is not None:
                    self.tracker.add_new_mask(state, frame_idx, oid_k, det_masks[k])
                outputs.append(oid_k)

            return {
                "session_id": session.session_id,
                "obj_ids": outputs,
                "outputs": det_out,
            }

        # ---- geometric prompts ----
        raw_points = request.get("points")
        raw_labels = request.get("labels")
        box = request.get("box")
        mask = request.get("mask")

        points_np = np.array(raw_points, dtype=np.float32) if raw_points else None
        labels_np = np.array(raw_labels, dtype=np.int32) if raw_labels else None

        if mask is not None:
            frame_idx_out, obj_ids, masks_out = self.tracker.add_new_mask(state, frame_idx, obj_id, np.array(mask))
        else:
            frame_idx_out, obj_ids, masks_out = self.tracker.add_new_points_or_box(
                state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points_np,
                labels=labels_np,
                box=np.array(box, dtype=np.float32) if box is not None else None,
                clear_old_points=request.get("clear_old_points", True),
                normalize_coords=request.get("normalize_coords", True),
            )

        # Convert masks to numpy for the response
        masks_np = np.array(masks_out)  # (num_obj, 1, H, W)
        boxes_xywh = _masks_to_boxes(masks_np[:, 0])

        return {
            "session_id": session.session_id,
            "frame_index": frame_idx_out,
            "obj_ids": obj_ids,
            "outputs": {
                "masks": (masks_np[:, 0] > 0.0).tolist(),
                "out_boxes_xywh": boxes_xywh.tolist(),
            },
        }

    # ---------------------------------------------------------------- #
    # Propagation                                                        #
    # ---------------------------------------------------------------- #

    def _handle_propagate(self, request: dict) -> dict:
        """
        Propagate all tracks through (part of) the video.
        Returns a list of per-frame result dicts.
        """
        session = self._get_session(request)
        state = session.inference_state

        start = request.get("start_frame_index")
        max_frames = request.get("max_frame_num_to_track")
        reverse = request.get("reverse", False)

        session.is_running = True
        frame_results = []

        for frame_idx, obj_ids, masks in self.tracker.propagate_in_video(
            state,
            start_frame_idx=start,
            max_frame_num_to_track=max_frames,
            reverse=reverse,
        ):
            masks_np = np.array(masks)  # (num_obj, 1, H, W)
            boxes = _masks_to_boxes(masks_np[:, 0])
            frame_results.append(
                {
                    "frame_index": frame_idx,
                    "obj_ids": list(obj_ids),
                    "masks": (masks_np[:, 0] > 0.0).tolist(),
                    "out_boxes_xywh": boxes.tolist(),
                }
            )

        session.is_running = False
        return {
            "session_id": session.session_id,
            "frames": frame_results,
        }

    # ---------------------------------------------------------------- #
    # Mask retrieval                                                     #
    # ---------------------------------------------------------------- #

    def _handle_get_masks(self, request: dict) -> dict:
        session = self._get_session(request)
        frame_idx = int(request.get("frame_index", 0))
        try:
            obj_ids = session.inference_state["obj_ids"]
            all_masks = []
            for oid in obj_ids:
                m = self.tracker.get_mask(session.inference_state, frame_idx, oid)
                all_masks.append(m)
            masks_np = np.stack(all_masks, axis=0) if all_masks else np.array([])
            return {
                "session_id": session.session_id,
                "frame_index": frame_idx,
                "obj_ids": obj_ids,
                "masks": masks_np.tolist(),
            }
        except RuntimeError as exc:
            return {"error": str(exc)}

    # ---------------------------------------------------------------- #
    # Detector integration                                               #
    # ---------------------------------------------------------------- #

    def _run_detector(
        self,
        session: TrackingSession,
        frame_idx: int,
        text: str,
        request: dict,
    ) -> dict:
        """Run the detector on *frame_idx* conditioned on *text*."""
        image = session.inference_state["images"][frame_idx]  # (1,3,H,W)
        det_out = self.detector.detect(
            image,
            text=text,
            threshold=request.get("det_threshold", 0.3),
        )
        return det_out

    # ---------------------------------------------------------------- #
    # Internal helpers                                                   #
    # ---------------------------------------------------------------- #

    def _get_session(self, request: dict) -> TrackingSession:
        session_id = request.get("session_id", "")
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found. " "Call handle_request(type='start_session') first.")
        return session


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Convert a batch of binary (N, H, W) masks to (N, 4) cx,cy,w,h boxes.
    Objects with no positive pixels get all-zero boxes.
    """
    N, H, W = masks.shape
    boxes = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        ys, xs = np.where(masks[i] > 0)
        if len(xs) == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        boxes[i] = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return boxes
