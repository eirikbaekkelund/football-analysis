# Copyright (c) Meta Platforms, Inc. and affiliates.
# MLX port of sam3/model/sam3_tracker_base.py
#
# SAM 3's tracker is the SAM 2 transformer encoder-decoder architecture with
# an additional "presence token" mechanism.  This base class owns all the
# memory-bank management, mask decoding, and per-frame conditioning that is
# shared between the video-object-segmentation predictor and the tracking
# predictor.

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# Sentinel value used when no valid score is available
NO_OBJ_SCORE = -1024.0


class Sam3TrackerBase(nn.Module):
    """
    Base class for SAM 3 video / tracking.

    Sub-classes must supply the following nn.Module attributes:
      - image_encoder        (HieroViT)
      - memory_attention     (transformer cross-attention over memory bank)
      - memory_encoder       (encodes a frame+mask into a memory token)
      - sam_mask_decoder     (SAM-2-style mask decoder)
      - sam_prompt_encoder   (SAM-2-style prompt encoder)
      - presence_head        (MLP: presence token -> presence logit per object)

    All nn.Module children are pure MLX, no torch tensors anywhere.
    """

    # ------------------------------------------------------------------ #
    # Configuration defaults (override via __init__ kwargs)               #
    # ------------------------------------------------------------------ #
    image_size: int = 1024
    backbone_stride: int = 16
    hidden_dim: int = 256
    num_maskmem: int = 7  # frames kept in the memory bank
    num_output_masks_train: int = 4
    num_output_masks_test: int = 1
    max_cond_frames_in_attn: int = -1  # -1 = unlimited

    # Binarisation threshold applied to soft logits when producing memory
    mem_dim: int = 64
    sigmoid_scale_for_mem_enc: float = 20.0
    sigmoid_bias_for_mem_enc: float = -10.0
    binarize_mask_from_pts_for_mem_enc: bool = False
    fill_hole_area: int = 0
    non_overlap_masks: bool = False
    use_obj_ptrs_in_encoder: bool = False
    max_obj_ptrs_in_encoder: int = 16
    add_tpos_enc_to_obj_ptrs: bool = True
    only_obj_ptrs_in_the_past_for_eval: bool = False
    use_signed_pt_tpos_enc: bool = False
    multimask_output_in_sam: bool = True
    use_multimask_token_for_obj_ptr: bool = False
    iou_prediction_use_sigmoid: bool = False
    use_act_pt_for_init_mem: bool = True
    # Presence token
    use_presence_token: bool = True

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Derived constants
        self._bb_feat_sizes = [(self.image_size // s, self.image_size // s) for s in [4, 8, self.backbone_stride]]

    # ------------------------------------------------------------------ #
    # Low-level helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_image_feature(
        self,
        inference_state: dict,
        frame_idx: int,
        batch_size: int,
    ) -> Tuple[List[mx.array], mx.array]:
        """Return cached (multi-scale backbone features, vision_pos_embeds)."""
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            # Fetch the raw frame tensor from storage
            image = inference_state["images"][frame_idx]  # (1,3,H,W)
            backbone_out = self.forward_image(image)
            inference_state["cached_features"][frame_idx] = (image, backbone_out)

        # Expand batch dim to match object count
        expanded = self._expand_features_to_batch(backbone_out, batch_size)
        return expanded["backbone_fpn"], expanded["vision_pos_enc"]

    def _expand_features_to_batch(self, backbone_out: dict, batch_size: int) -> dict:
        """Repeat backbone features along batch dimension."""
        fpn = [mx.repeat(f, batch_size, axis=0) for f in backbone_out["backbone_fpn"]]
        pos = [mx.repeat(p, batch_size, axis=0) for p in backbone_out["vision_pos_enc"]]
        return {"backbone_fpn": fpn, "vision_pos_enc": pos}

    def forward_image(self, img: mx.array) -> dict:
        """Run the image encoder on a single (1,3,H,W) frame."""
        backbone_out = self.image_encoder(img)
        # Optionally fuse neck features here (sub-class may override)
        return backbone_out

    # ------------------------------------------------------------------ #
    # Memory bank management                                               #
    # ------------------------------------------------------------------ #

    def _init_memory_inputs(
        self,
        current_vision_feats: List[mx.array],
        current_vision_pos_embeds: List[mx.array],
        memory_dict: dict,
        obj_ptrs: Optional[mx.array],
        obj_ptr_tpos: Optional[mx.array],
        is_init_cond_frame: bool,
        current_frame_idx: int,
    ) -> mx.array:
        """
        Build the conditioning tokens fed into memory_attention, consisting of:
          1. Frame-level memory tokens from the memory bank
          2. (optionally) object pointer tokens from prior frames
        Returns a (T_mem, B, C) tensor ready for cross-attention.
        """
        B = current_vision_feats[-1].shape[0]
        C = self.hidden_dim
        H, W = self._bb_feat_sizes[-1]

        to_cat = []
        pos_to_cat = []

        # ---- memory frames ----
        num_frames = self.num_maskmem
        if len(memory_dict) > 0:
            # Sort from oldest to newest
            mem_keys = sorted(memory_dict.keys())[-num_frames:]
            for t in mem_keys:
                mem = memory_dict[t]  # (B, C, H, W)
                pos = mem.get("pos_enc")
                feat = mem["feat"]  # (B, C, H/s, W/s)
                # flatten spatial
                feat = feat.reshape(B, C, -1).transpose(0, 2, 1)  # (B, HW, C)
                feat = feat.transpose(1, 0, 2)  # (HW, B, C)
                to_cat.append(feat)
                if pos is not None:
                    pos = pos.reshape(B, C, -1).transpose(0, 2, 1)
                    pos = pos.transpose(1, 0, 2)
                    pos_to_cat.append(pos)
                else:
                    pos_to_cat.append(mx.zeros_like(feat))

        # ---- object pointer tokens ----
        if self.use_obj_ptrs_in_encoder and obj_ptrs is not None:
            # obj_ptrs: (num_ptrs, B, C)
            to_cat.append(obj_ptrs)
            if self.add_tpos_enc_to_obj_ptrs and obj_ptr_tpos is not None:
                pos_to_cat.append(obj_ptr_tpos)
            else:
                pos_to_cat.append(mx.zeros_like(obj_ptrs))

        if len(to_cat) == 0:
            return None, None

        memory_tokens = mx.concatenate(to_cat, axis=0)  # (T, B, C)
        memory_pos = mx.concatenate(pos_to_cat, axis=0)  # (T, B, C)
        return memory_tokens, memory_pos

    def _encode_new_memory(
        self,
        current_vision_feats: List[mx.array],
        feat_sizes: List[Tuple[int, int]],
        pred_masks_high_res: mx.array,
        object_score_logits: mx.array,
        is_mask_from_pts: bool,
    ) -> mx.array:
        """Encode the current frame + its predicted mask into a memory token."""
        H, W = feat_sizes[-1]
        # Resize predicted mask to match the lowest-resolution feature map
        pred_mask = mx.array(
            np.array(
                # bilinear resize via numpy (mlx lacks a built-in bilinear resize)
                _bilinear_resize_np(np.array(pred_masks_high_res), H, W)
            )
        )
        if self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts:
            pred_mask = (pred_mask > 0.0).astype(mx.float32)
        else:
            pred_mask = mx.sigmoid(pred_mask * self.sigmoid_scale_for_mem_enc + self.sigmoid_bias_for_mem_enc)

        x = current_vision_feats[-1]  # (B, HW, C) flattened last-level feat
        # memory_encoder takes (backbone_feat, mask) and returns a memory token
        new_mem = self.memory_encoder(
            pix_feat=x,
            masks=pred_mask,
            skip_mask_sigmoid=True,  # sigmoid already applied above
        )
        return new_mem

    # ------------------------------------------------------------------ #
    # SAM-style mask decoding for a single frame                          #
    # ------------------------------------------------------------------ #

    def _forward_sam_heads(
        self,
        backbone_features: mx.array,  # (B, C, H, W) highest-res feature
        point_inputs: Optional[dict],
        mask_inputs: Optional[mx.array],
        high_res_features: List[mx.array],  # two lower strides for up-sampling
        multimask_output: bool,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Run SAM prompt encoder + mask decoder.
        Returns (low_res_masks, high_res_masks, ious, object_score_logits).
        """
        B = backbone_features.shape[0]
        # Encode prompts
        sparse_emb, dense_emb = self.sam_prompt_encoder(
            points=point_inputs,
            boxes=None,
            masks=mask_inputs,
        )

        # Decode
        low_res_masks, ious, _, obj_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # Upsample to full image size (4× the backbone stride)
        high_res_masks = mx.array(
            np.array(
                _bilinear_resize_np(
                    np.array(low_res_masks),
                    self.image_size,
                    self.image_size,
                )
            )
        )
        return low_res_masks, high_res_masks, ious, obj_score_logits

    def _use_multimask(self, is_init_cond_frame: bool, point_inputs: Optional[dict]) -> bool:
        """Decide whether to produce multi-mask outputs during inference."""
        num_pts = point_inputs["point_coords"].shape[1] if point_inputs is not None else 0
        return self.multimask_output_in_sam and is_init_cond_frame and num_pts > 0

    # ------------------------------------------------------------------ #
    # Object pointer extraction                                            #
    # ------------------------------------------------------------------ #

    def _get_obj_ptr_from_masks(
        self,
        masks: mx.array,
        ious: mx.array,
        multimask: bool,
    ) -> mx.array:
        """
        Select the best-scoring mask per object and extract its object
        pointer (a compact vector representation used in future frames).
        """
        if multimask:
            best_idx = mx.argmax(ious, axis=-1)  # (B,)
            # gather best mask
            B = masks.shape[0]
            best_masks = mx.stack([masks[b, best_idx[b].item()] for b in range(B)], axis=0)  # (B, H, W)
        else:
            best_masks = masks[:, 0]  # (B, H, W)

        # Global-average-pool the high-res backbone feat conditioned on mask
        # (sub-class / memory_encoder produces the pointer from this)
        obj_ptr = self.memory_encoder.get_obj_ptr(best_masks)  # (B, C)
        return obj_ptr

    # ------------------------------------------------------------------ #
    # Presence token (SAM 3 specific)                                     #
    # ------------------------------------------------------------------ #

    def _get_presence_scores(
        self,
        obj_score_logits: mx.array,  # (B, 1)
    ) -> mx.array:
        """Return per-object presence probability in [0,1]."""
        if not self.use_presence_token:
            return mx.ones((obj_score_logits.shape[0], 1))
        return mx.sigmoid(self.presence_head(obj_score_logits))

    # ------------------------------------------------------------------ #
    # Public forward                                                       #
    # ------------------------------------------------------------------ #

    def track_step(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        current_vision_feats: List[mx.array],
        current_vision_pos_embeds: List[mx.array],
        feat_sizes: List[Tuple[int, int]],
        point_inputs: Optional[dict],
        mask_inputs: Optional[mx.array],
        output_dict: dict,
        num_frames: int,
        track_in_reverse: bool = False,
        run_mem_encoder: bool = True,
        prev_sam_mask_logits: Optional[mx.array] = None,
    ) -> dict:
        """
        Full tracker forward pass for one frame.

        Returns a dict with keys:
          pred_masks, pred_masks_high_res, obj_ptrs,
          object_score_logits, presence_scores
        """
        B = current_vision_feats[-1].shape[0]
        H, W = feat_sizes[-1]

        # ---- attend to memory bank ----
        mem_tokens, mem_pos = self._init_memory_inputs(
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            memory_dict=output_dict.get("cond_frame_outputs", {}),
            obj_ptrs=output_dict.get("obj_ptrs"),
            obj_ptr_tpos=output_dict.get("obj_ptr_tpos"),
            is_init_cond_frame=is_init_cond_frame,
            current_frame_idx=frame_idx,
        )

        # ---- memory attention to get per-frame conditioned feature ----
        last_feat = current_vision_feats[-1]  # (B, HW, C)
        last_pos = current_vision_pos_embeds[-1]

        if mem_tokens is not None:
            # memory_attention is (query, key/value) cross-attention
            conditioned_feat = self.memory_attention(
                curr=last_feat,
                curr_pos=last_pos,
                memory=mem_tokens,
                memory_pos=mem_pos,
                num_obj_ptr_tokens=0,
            )  # (B, HW, C)
        else:
            conditioned_feat = last_feat

        # Reshape to (B, C, H, W) for mask decoder
        conditioned_feat_2d = conditioned_feat.reshape(B, H, W, -1).transpose(0, 3, 1, 2)

        # ---- SAM head ----
        multimask = self._use_multimask(is_init_cond_frame, point_inputs)
        low_res_masks, high_res_masks, ious, obj_score_logits = self._forward_sam_heads(
            backbone_features=conditioned_feat_2d,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs if not is_init_cond_frame else prev_sam_mask_logits,
            high_res_features=current_vision_feats[:-1],
            multimask_output=multimask,
        )

        # ---- best mask selection (single-mask output) ----
        if multimask and not self.training:
            best_idx = mx.argmax(ious, axis=-1)
            B_ = low_res_masks.shape[0]
            low_res_masks = mx.stack(
                [low_res_masks[b, best_idx[b].item() : best_idx[b].item() + 1] for b in range(B_)], axis=0
            )
            high_res_masks = mx.stack(
                [high_res_masks[b, best_idx[b].item() : best_idx[b].item() + 1] for b in range(B_)], axis=0
            )
            ious = mx.stack([ious[b, best_idx[b].item() : best_idx[b].item() + 1] for b in range(B_)], axis=0)

        # ---- presence token ----
        presence_scores = self._get_presence_scores(obj_score_logits)

        # ---- object pointers ----
        obj_ptrs = self._get_obj_ptr_from_masks(high_res_masks, ious, multimask=False)

        # ---- encode memory if needed ----
        if run_mem_encoder:
            new_mem = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=obj_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            output_dict.setdefault("cond_frame_outputs" if is_init_cond_frame else "non_cond_frame_outputs", {})[
                frame_idx
            ] = {
                "feat": new_mem,
                "obj_ptr": obj_ptrs,
            }

        return {
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "obj_ptrs": obj_ptrs,
            "object_score_logits": obj_score_logits,
            "presence_scores": presence_scores,
            "ious": ious,
        }


# ---------------------------------------------------------------------------
# Helpers that need numpy (MLX bilinear resize shim)
# ---------------------------------------------------------------------------


def _bilinear_resize_np(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Bilinear resize for (B, C, H, W) numpy arrays.
    Uses cv2 when available, falls back to a simple nearest-neighbour.
    """
    try:
        import cv2

        B, C, H, W = arr.shape
        out = np.empty((B, C, out_h, out_w), dtype=arr.dtype)
        for b in range(B):
            for c in range(C):
                out[b, c] = cv2.resize(arr[b, c], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return out
    except ImportError:
        # Nearest-neighbour fallback
        B, C, H, W = arr.shape
        row_idx = (np.arange(out_h) * H / out_h).astype(int)
        col_idx = (np.arange(out_w) * W / out_w).astype(int)
        return arr[:, :, row_idx[:, None], col_idx[None, :]]
