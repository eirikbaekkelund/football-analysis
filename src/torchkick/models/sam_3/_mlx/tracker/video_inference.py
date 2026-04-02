# Copyright (c) Meta Platforms, Inc. and affiliates.
# MLX port of sam3/model/sam3_video_inference.py
#
# Batch / offline inference helpers that run SAM 3 on a whole video and
# return structured results without requiring a generator loop.

from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np

from torchkick.sam_3._mlx.tracker.video_predictor import Sam3VideoPredictor


# ------------------------------------------------------------------ #
# Result containers                                                    #
# ------------------------------------------------------------------ #


@dataclass
class ObjectResult:
    """Segmentation result for a single object across all tracked frames."""

    obj_id: int
    # frame_idx -> binary (H, W) mask
    masks: Dict[int, np.ndarray] = field(default_factory=dict)
    # frame_idx -> presence probability
    presence: Dict[int, float] = field(default_factory=dict)
    # frame_idx -> IoU score
    iou: Dict[int, float] = field(default_factory=dict)


@dataclass
class VideoInferenceResult:
    """All object results for one video."""

    num_frames: int
    objects: Dict[int, ObjectResult] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """
        Return a dense (num_obj, num_frames, H, W) bool array.
        Objects/frames with no mask are filled with False.
        """
        if not self.objects:
            return np.zeros((0, self.num_frames, 0, 0), dtype=bool)
        obj_ids = sorted(self.objects)
        first_obj = self.objects[obj_ids[0]]
        sample_frame = next(iter(first_obj.masks.values()))
        H, W = sample_frame.shape
        out = np.zeros((len(obj_ids), self.num_frames, H, W), dtype=bool)
        for oi, oid in enumerate(obj_ids):
            for fi, m in self.objects[oid].masks.items():
                out[oi, fi] = m
        return out


# ------------------------------------------------------------------ #
# Main inference function                                              #
# ------------------------------------------------------------------ #


def run_video_inference(
    predictor: Sam3VideoPredictor,
    video_path: Union[str, List[np.ndarray]],
    prompts: List[dict],
    reverse: bool = False,
    return_logits: bool = False,
) -> VideoInferenceResult:
    """
    Run full-video SAM 3 tracking from a list of prompts.

    Args:
        predictor   : initialised Sam3VideoPredictor (MLX)
        video_path  : JPEG folder, mp4 path, or list of (H,W,3) uint8 arrays
        prompts     : list of prompt dicts, each with keys:
                        - "frame_idx"  (int)
                        - "obj_id"     (int)
                        - "points"     (np.ndarray, shape (N,2), optional)
                        - "labels"     (np.ndarray, shape (N,),  optional)
                        - "box"        (np.ndarray, shape (4,),   optional)
                        - "mask"       (np.ndarray, shape (H,W),  optional)
        reverse     : propagate backwards in time
        return_logits: store raw float logits instead of binary masks

    Returns:
        VideoInferenceResult
    """
    state = predictor.init_state(video_path)

    # ---- register prompts ----
    for p in prompts:
        frame_idx = p["frame_idx"]
        obj_id = p["obj_id"]
        if "mask" in p:
            predictor.add_new_mask(state, frame_idx, obj_id, p["mask"])
        else:
            predictor.add_new_points_or_box(
                state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=p.get("points"),
                labels=p.get("labels"),
                box=p.get("box"),
                clear_old_points=p.get("clear_old_points", True),
                normalize_coords=p.get("normalize_coords", True),
            )

    # ---- build result containers ----
    result = VideoInferenceResult(num_frames=state["num_frames"])
    for oid in state["obj_ids"]:
        result.objects[oid] = ObjectResult(obj_id=oid)

    # ---- propagate ----
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state, reverse=reverse):
        masks_np = np.array(masks)  # (num_obj, 1, H, W)
        for i, oid in enumerate(obj_ids):
            m = masks_np[i, 0]
            if return_logits:
                result.objects[oid].masks[frame_idx] = m
            else:
                result.objects[oid].masks[frame_idx] = m > 0.0

    return result


# ------------------------------------------------------------------ #
# Streaming variant (memory-efficient)                                #
# ------------------------------------------------------------------ #


def stream_video_inference(
    predictor: Sam3VideoPredictor,
    video_path: Union[str, List[np.ndarray]],
    prompts: List[dict],
    callback,
    reverse: bool = False,
):
    """
    Like run_video_inference but calls *callback(frame_idx, obj_ids, masks)*
    for each frame instead of accumulating results in RAM.

    *masks* shape: (num_obj, 1, H, W) float logits.
    """
    state = predictor.init_state(video_path)

    for p in prompts:
        frame_idx = p["frame_idx"]
        obj_id = p["obj_id"]
        if "mask" in p:
            predictor.add_new_mask(state, frame_idx, obj_id, p["mask"])
        else:
            predictor.add_new_points_or_box(
                state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=p.get("points"),
                labels=p.get("labels"),
                box=p.get("box"),
                clear_old_points=p.get("clear_old_points", True),
                normalize_coords=p.get("normalize_coords", True),
            )

    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state, reverse=reverse):
        callback(frame_idx, obj_ids, masks)
