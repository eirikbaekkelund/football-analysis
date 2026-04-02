# Copyright (c) Meta Platforms, Inc. and affiliates.
# MLX port of sam3/model/sam3_video_base.py
#
# Manages per-session inference state (frame loading, memory bank layout,
# object tracking bookkeeping) for video-object segmentation.

from typing import List, Optional, Tuple, Union
from pathlib import Path

import mlx.core as mx
import numpy as np

from torchkick.sam_3._mlx.tracker.video_base import Sam3TrackerBase, _bilinear_resize_np
from torchkick.sam_3._mlx.tracker.utils import apply_non_overlapping_constraints, fill_holes_in_mask_scores

# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #
_JPEG_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}


class Sam3VideoBase(Sam3TrackerBase):
    """
    Extends Sam3TrackerBase with:
      - Session-level inference state initialisation
      - Frame loading (JPEG folder or MP4 via cv2)
      - Object CRUD (add / reset)
      - propagate_in_video  (forward OR backward pass)
    """

    # -------------------------------------------------------------- #
    # Session management                                               #
    # -------------------------------------------------------------- #

    def init_state(
        self,
        video_path: Union[str, List[np.ndarray]],
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> dict:
        """
        Load a video and return an inference_state dict.

        *video_path* can be:
          - a directory of JPEG frames (named <idx>.jpg / <idx>.jpeg)
          - a path to an mp4 file
          - a list of (H, W, 3) uint8 numpy arrays
        """
        compute_device = mx.default_device()
        inference_state = {
            "images": None,  # loaded lazily below
            "num_frames": 0,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            # Object bookkeeping
            "obj_id_to_idx": {},
            "obj_idx_to_id": {},
            "obj_ids": [],
            # Per-frame caches
            "cached_features": {},
            # Per-object, per-frame outputs
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            # Per-object point/mask prompts
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            # Whether each object has received its initial prompt
            "obj_idx_to_is_init": {},
            # last propagation direction
            "propagation_started": False,
            "track_in_reverse": False,
            "frames_tracked_per_obj": {},
            "storage_device": compute_device,
        }

        # ---- load frames ----
        if isinstance(video_path, list):
            frames = video_path
        elif isinstance(video_path, (str, Path)) and str(video_path).endswith(".mp4"):
            frames = _load_mp4_frames(str(video_path))
        else:
            frames = _load_jpeg_frames(str(video_path))

        inference_state["num_frames"] = len(frames)
        inference_state["images"] = _frames_to_tensors(frames, self.image_size)
        return inference_state

    def reset_state(self, inference_state: dict):
        """Clear all prompts, memory, and object state."""
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["cached_features"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["obj_idx_to_is_init"].clear()
        inference_state["frames_tracked_per_obj"].clear()
        inference_state["propagation_started"] = False

    # -------------------------------------------------------------- #
    # Object / prompt management                                       #
    # -------------------------------------------------------------- #

    def _obj_id_to_idx(self, inference_state: dict, obj_id: int) -> int:
        """Map a user-facing object id to internal slot index."""
        idx = inference_state["obj_id_to_idx"].get(obj_id)
        if idx is not None:
            return idx
        idx = len(inference_state["obj_id_to_idx"])
        inference_state["obj_id_to_idx"][obj_id] = idx
        inference_state["obj_idx_to_id"][idx] = obj_id
        inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
        inference_state["point_inputs_per_obj"][idx] = {}
        inference_state["mask_inputs_per_obj"][idx] = {}
        inference_state["obj_idx_to_is_init"][idx] = False
        inference_state["frames_tracked_per_obj"][idx] = set()
        return idx

    def add_new_points_or_box(
        self,
        inference_state: dict,
        frame_idx: int,
        obj_id: int,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[int, List[int], mx.array]:
        """
        Add click-point or box prompts for *obj_id* at *frame_idx*.
        Returns (frame_idx, object_ids, mask_logits).
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        H, W = self.image_size, self.image_size

        # Convert box to two points if provided
        if box is not None:
            if not isinstance(box, np.ndarray):
                box = np.array(box)
            box = box.reshape(2, 2)
            box_pts = box[None]  # (1, 2, 2)
            box_labels = np.array([[2, 3]])  # left-top=2, right-bot=3
            if points is not None:
                points = np.concatenate([box_pts[0], points], axis=0)[None]
                labels = np.concatenate([box_labels[0], labels], axis=0)[None]
            else:
                points, labels = box_pts, box_labels

        if points is None:
            points = np.zeros((1, 0, 2), dtype=np.float32)
            labels = np.zeros((1, 0), dtype=np.int32)

        if normalize_coords:
            points = points.astype(np.float32) / np.array([W, H], dtype=np.float32)

        pt_tensor = mx.array(points.astype(np.float32))  # (1, N, 2)
        lbl_tensor = mx.array(labels.astype(np.int32))  # (1, N)

        if clear_old_points:
            inference_state["point_inputs_per_obj"][obj_idx][frame_idx] = {
                "point_coords": pt_tensor,
                "point_labels": lbl_tensor,
            }
        else:
            old = inference_state["point_inputs_per_obj"][obj_idx].get(frame_idx)
            from torchkick.sam_3._mlx.tracker.utils import concat_points

            inference_state["point_inputs_per_obj"][obj_idx][frame_idx] = concat_points(old, pt_tensor, lbl_tensor)

        inference_state["obj_idx_to_is_init"][obj_idx] = True
        return self._run_single_frame_inference(inference_state, frame_idx, obj_idx, is_init=True)

    def add_new_mask(
        self,
        inference_state: dict,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray,
    ) -> Tuple[int, List[int], mx.array]:
        """Provide a binary mask as prompt for *obj_id* at *frame_idx*."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        # Resize mask to low-res decoder input size
        mask_lr = _bilinear_resize_np(
            mask[None, None].astype(np.float32),
            self.image_size // 4,
            self.image_size // 4,
        )[0]
        inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = mx.array(mask_lr)
        inference_state["obj_idx_to_is_init"][obj_idx] = True
        return self._run_single_frame_inference(inference_state, frame_idx, obj_idx, is_init=True)

    def clear_all_prompts_in_frame(self, inference_state: dict, frame_idx: int, obj_id: int):
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        # Remove cached outputs for this frame
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            inference_state["output_dict"][key].pop(frame_idx, None)

    # -------------------------------------------------------------- #
    # Single-frame inference (prompt → mask)                          #
    # -------------------------------------------------------------- #

    def _run_single_frame_inference(
        self,
        inference_state: dict,
        frame_idx: int,
        obj_idx: int,
        is_init: bool,
    ) -> Tuple[int, List[int], mx.array]:
        """Run tracker on a single frame and return (frame_idx, obj_ids, masks)."""
        num_obj = len(inference_state["obj_id_to_idx"])
        backbone_feats, vision_pos = self._get_image_feature(inference_state, frame_idx, batch_size=1)
        feat_sizes = self._bb_feat_sizes

        point_inputs = inference_state["point_inputs_per_obj"][obj_idx].get(frame_idx)
        mask_inputs = inference_state["mask_inputs_per_obj"][obj_idx].get(frame_idx)

        out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init,
            current_vision_feats=backbone_feats,
            current_vision_pos_embeds=vision_pos,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=inference_state["output_dict"],
            num_frames=inference_state["num_frames"],
        )
        masks_out = out["pred_masks_high_res"]  # (1, 1, H, W)
        obj_ids = inference_state["obj_ids"]
        return frame_idx, obj_ids, masks_out

    # -------------------------------------------------------------- #
    # Video propagation                                                #
    # -------------------------------------------------------------- #

    def propagate_in_video_preflight(self, inference_state: dict):
        """Validate that at least one object has been initialised."""
        if not any(inference_state["obj_idx_to_is_init"].values()):
            raise RuntimeError("No objects have been prompted. Call add_new_points_or_box() first.")

    def propagate_in_video(
        self,
        inference_state: dict,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
    ):
        """
        Generator: propagate all object masks through the video.

        Yields (frame_idx, object_ids, masks) for every tracked frame.
        *masks* has shape (num_obj, 1, H, W) as float logits.
        """
        self.propagate_in_video_preflight(inference_state)

        num_frames = inference_state["num_frames"]
        obj_ids = inference_state["obj_ids"]
        num_obj = len(obj_ids)

        if start_frame_idx is None:
            if reverse:
                start_frame_idx = num_frames - 1
            else:
                # Start from the earliest prompted frame
                all_cond = inference_state["output_dict"]["cond_frame_outputs"]
                start_frame_idx = min(all_cond) if all_cond else 0

        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames

        if reverse:
            frame_range = range(start_frame_idx, -1, -1)
        else:
            frame_range = range(start_frame_idx, min(start_frame_idx + max_frame_num_to_track, num_frames))

        inference_state["propagation_started"] = True
        inference_state["track_in_reverse"] = reverse

        for frame_idx in frame_range:
            # Collect per-object outputs
            all_masks = []
            for obj_id in obj_ids:
                obj_idx = inference_state["obj_id_to_idx"][obj_id]
                is_cond = frame_idx in inference_state["output_dict"]["cond_frame_outputs"]

                backbone_feats, vision_pos = self._get_image_feature(inference_state, frame_idx, batch_size=1)
                out = self.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_cond,
                    current_vision_feats=backbone_feats,
                    current_vision_pos_embeds=vision_pos,
                    feat_sizes=self._bb_feat_sizes,
                    point_inputs=inference_state["point_inputs_per_obj"][obj_idx].get(frame_idx),
                    mask_inputs=inference_state["mask_inputs_per_obj"][obj_idx].get(frame_idx),
                    output_dict=inference_state["output_dict"],
                    num_frames=num_frames,
                    track_in_reverse=reverse,
                )
                mask = out["pred_masks_high_res"]  # (1, 1, H, W)
                all_masks.append(mask)
                inference_state["frames_tracked_per_obj"][obj_idx].add(frame_idx)

            # Stack across objects: (num_obj, 1, H, W)
            masks_stacked = mx.concatenate(all_masks, axis=0)

            if self.non_overlap_masks:
                masks_stacked = apply_non_overlapping_constraints(masks_stacked[:, 0])[:, None]

            yield frame_idx, obj_ids, masks_stacked

    # -------------------------------------------------------------- #
    # Post-processing helper                                           #
    # -------------------------------------------------------------- #

    def postprocess_masks(
        self,
        masks: mx.array,
        orig_hw: Tuple[int, int],
    ) -> mx.array:
        """
        Resize mask logits back to *orig_hw* and optionally fill small holes.
        """
        H_orig, W_orig = orig_hw
        masks_np = np.array(masks)
        masks_resized = _bilinear_resize_np(masks_np, H_orig, W_orig)
        masks_mx = mx.array(masks_resized)
        if self.fill_hole_area > 0:
            # Apply per-object
            filled = []
            for i in range(masks_mx.shape[0]):
                filled.append(fill_holes_in_mask_scores(masks_mx[i, 0], self.fill_hole_area)[None])
            masks_mx = mx.stack(filled, axis=0)
        return masks_mx


# ---------------------------------------------------------------------------
# Frame-loading utilities
# ---------------------------------------------------------------------------


def _load_jpeg_frames(folder: str) -> List[np.ndarray]:
    """Load all JPEG frames from *folder*, sorted numerically by filename stem."""
    paths = sorted(
        [p for p in Path(folder).iterdir() if p.suffix in _JPEG_EXTENSIONS],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not paths:
        raise FileNotFoundError(f"No JPEG frames found in {folder}")
    try:
        from PIL import Image as PILImage

        return [np.array(PILImage.open(str(p)).convert("RGB")) for p in paths]
    except ImportError:
        import cv2

        return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]


def _load_mp4_frames(path: str) -> List[np.ndarray]:
    """Decode all frames from an mp4 using cv2."""
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from {path}")
    return frames


def _frames_to_tensors(
    frames: List[np.ndarray],
    image_size: int,
) -> List[mx.array]:
    """
    Resize each (H, W, 3) uint8 frame to (1, 3, image_size, image_size) float32
    in [0,1] range and wrap in mx.array.
    """
    tensors = []
    for frame in frames:
        # Resize
        resized = _bilinear_resize_np(
            frame.transpose(2, 0, 1)[None].astype(np.float32) / 255.0,
            image_size,
            image_size,
        )
        tensors.append(mx.array(resized))  # (1, 3, H, W)
    return tensors
