# Copyright (c) Meta Platforms, Inc. and affiliates.
# MLX port of sam3/model/sam3_video_predictor.py
#
# High-level video predictor: wraps Sam3VideoBase and exposes a clean API
# that mirrors the PyTorch version (init_state, add_new_*, propagate_in_video).

from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np

from torchkick.sam_3._mlx.tracker.tracker_base import Sam3VideoBase


class Sam3VideoPredictor(Sam3VideoBase):
    """
    End-user API for SAM 3 video segmentation.

    Typical usage::

        from sam3.model_builder import build_sam3_video_predictor_mlx
        predictor = build_sam3_video_predictor_mlx()

        # Load video
        state = predictor.init_state("path/to/frames/")

        # Prompt frame 0 for object 1 with a positive click at (x, y)
        _, obj_ids, masks = predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1,
            points=np.array([[x, y]]), labels=np.array([1])
        )

        # Propagate forward
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            # masks: (num_obj, 1, H, W) float logits
            binary = np.array(masks) > 0.0
    """

    def __init__(
        self,
        fill_hole_area: int = 0,
        non_overlap_masks: bool = False,
        non_overlap_masks_for_mem_enc: bool = False,
        **kwargs,
    ):
        super().__init__(
            fill_hole_area=fill_hole_area,
            non_overlap_masks=non_overlap_masks,
            **kwargs,
        )
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc

    # ---------------------------------------------------------------- #
    # Convenience wrappers                                               #
    # ---------------------------------------------------------------- #

    def get_mask(
        self,
        inference_state: dict,
        frame_idx: int,
        obj_id: int,
    ) -> np.ndarray:
        """
        Return a binary (H, W) mask for *obj_id* at *frame_idx*.
        Assumes propagation has already been run.
        """
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id)
        if obj_idx is None:
            raise KeyError(f"Unknown obj_id {obj_id}")

        cond_out = inference_state["output_dict"]["cond_frame_outputs"]
        non_cond = inference_state["output_dict"]["non_cond_frame_outputs"]
        out = cond_out.get(frame_idx) or non_cond.get(frame_idx)
        if out is None:
            raise RuntimeError(f"Frame {frame_idx} has not been tracked yet.")
        mask_logits = out.get("pred_masks_high_res")
        if mask_logits is None:
            raise RuntimeError("No mask logits cached for this frame.")
        return np.array(mask_logits[obj_idx, 0]) > 0.0

    # ---------------------------------------------------------------- #
    # Propagation with optional post-processing                         #
    # ---------------------------------------------------------------- #

    def propagate_in_video(
        self,
        inference_state: dict,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
        return_dict: bool = False,
    ):
        """
        Generator yielding (frame_idx, object_ids, masks) for each frame.
        *masks* shape: (num_obj, 1, H_orig, W_orig) float logits.

        When *return_dict=True* each iteration yields
        (frame_idx, {obj_id: binary_mask_HW}).
        """
        # Retrieve original frame dimensions (before padding/resize)
        first_img = inference_state["images"][0]  # (1, 3, H, W)
        H_model, W_model = first_img.shape[2], first_img.shape[3]
        orig_hw = (H_model, W_model)  # same if no orig_hw stored

        for frame_idx, obj_ids, masks in super().propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        ):
            # Cache per-frame output for get_mask()
            cond_out = inference_state["output_dict"]["cond_frame_outputs"]
            non_cond = inference_state["output_dict"]["non_cond_frame_outputs"]
            store_dict = cond_out if frame_idx in cond_out else non_cond
            store_dict.setdefault(frame_idx, {})["pred_masks_high_res"] = masks

            if return_dict:
                binary = np.array(masks)[:, 0] > 0.0  # (num_obj, H, W)
                yield frame_idx, {oid: binary[i] for i, oid in enumerate(obj_ids)}
            else:
                yield frame_idx, obj_ids, masks

    # ---------------------------------------------------------------- #
    # Streaming single-frame re-segmentation                            #
    # ---------------------------------------------------------------- #

    def re_segment_frame(
        self,
        inference_state: dict,
        frame_idx: int,
    ) -> Tuple[int, List[int], mx.array]:
        """
        Re-run the tracker on *frame_idx* using the current memory bank state,
        without adding new prompts. Useful for interactive refinement.
        """
        obj_ids = inference_state["obj_ids"]
        all_masks = []
        for obj_id in obj_ids:
            obj_idx = inference_state["obj_id_to_idx"][obj_id]
            backbone_feats, vision_pos = self._get_image_feature(inference_state, frame_idx, batch_size=1)
            out = self.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=False,
                current_vision_feats=backbone_feats,
                current_vision_pos_embeds=vision_pos,
                feat_sizes=self._bb_feat_sizes,
                point_inputs=None,
                mask_inputs=None,
                output_dict=inference_state["output_dict"],
                num_frames=inference_state["num_frames"],
            )
            all_masks.append(out["pred_masks_high_res"])

        masks_stacked = mx.concatenate(all_masks, axis=0)
        return frame_idx, obj_ids, masks_stacked
