import os
import cv2
import uuid
from typing import Dict, Optional, Iterator
from PIL import Image

from torchkick.sam_3._mlx.backbone.processor import Sam3Processor


class Sam3VideoPredictorMLX:
    """Lightweight MLX-based video predictor that mimics the PyTorch `Sam3VideoPredictor` API.

    Notes:
    - This is a simple, CPU-oriented implementation intended for development and
      small videos on platforms where the PyTorch video predictor (with Triton)
      is unavailable.
    - It supports text prompts (multiple) and propagates by re-running
      `set_image` + `set_text_prompt` per frame.
    """

    _SESSIONS: Dict[str, Dict] = {}

    def __init__(self, model, processor: Optional[Sam3Processor] = None):
        self.model = model
        self.processor = processor or Sam3Processor(model)

    @classmethod
    def from_assets(cls, bpe_path: str, checkpoint_path: str):
        # build model using existing MLX conversion helper
        from torchkick.sam_3._mlx.model import build_sam3_image_model

        model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path)
        return cls(model)

    def start_session(self, resource_path: str, session_id: Optional[str] = None):
        if session_id is None:
            session_id = str(uuid.uuid4())

        # resource_path can be a video file or a directory of frames
        session = {"resource_path": resource_path, "prompts": {}, "next_obj_id": 1}

        if os.path.isdir(resource_path):
            imgs = sorted([os.path.join(resource_path, p) for p in os.listdir(resource_path)])
            session["type"] = "frames"
            session["frames"] = imgs
            session["num_frames"] = len(imgs)
        else:
            # assume video file
            cap = cv2.VideoCapture(resource_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video {resource_path}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            session["type"] = "video"
            session["video_path"] = resource_path
            session["num_frames"] = total

        self._SESSIONS[session_id] = session
        return {"session_id": session_id}

    def _read_frame(self, session: Dict, idx: int) -> Image.Image:
        if session["type"] == "frames":
            path = session["frames"][idx]
            return Image.open(path).convert("RGB")
        else:
            cap = cv2.VideoCapture(session["video_path"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(f"Cannot read frame {idx}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)

    def add_prompt(
        self,
        session_id: str,
        frame_index: int,
        text: Optional[str] = None,
        points=None,
        point_labels=None,
        bounding_boxes=None,
        obj_id: Optional[int] = None,
    ):
        session = self._SESSIONS.get(session_id)
        if session is None:
            raise RuntimeError(f"Unknown session {session_id}")

        if obj_id is None:
            obj_id = session["next_obj_id"]
            session["next_obj_id"] += 1

        # read the target frame and build state
        pil = self._read_frame(session, frame_index)
        state = self.processor.set_image(pil, state={})
        outputs = None
        if text is not None:
            state = self.processor.set_text_prompt(text, state)
            outputs = {"masks": state.get("masks"), "boxes": state.get("boxes"), "scores": state.get("scores")}

        # store prompt metadata
        session["prompts"][obj_id] = {"text": text}
        session.setdefault("state", {})
        session["state"][obj_id] = state

        return {"frame_index": frame_index, "outputs": outputs, "obj_id": obj_id}

    def propagate_in_video(
        self,
        session_id: str,
        propagation_direction: str = "forward",
        start_frame_idx: Optional[int] = 0,
        max_frame_num_to_track: Optional[int] = None,
    ) -> Iterator[Dict]:
        session = self._SESSIONS.get(session_id)
        if session is None:
            raise RuntimeError(f"Unknown session {session_id}")

        num_frames = session.get("num_frames", 0)
        start = start_frame_idx or 0
        end = num_frames
        if max_frame_num_to_track is not None:
            end = min(end, start + max_frame_num_to_track)

        # For each frame, re-run set_image and then set_text_prompt per prompt to get masks
        for frame_idx in range(start, end):
            pil = self._read_frame(session, frame_idx)
            frame_results = {}
            for obj_id, meta in session["prompts"].items():
                state = self.processor.set_image(pil, state={})
                if meta.get("text") is not None:
                    state = self.processor.set_text_prompt(meta["text"], state)
                    frame_results[obj_id] = {
                        "masks": state.get("masks"),
                        "boxes": state.get("boxes"),
                        "scores": state.get("scores"),
                    }
            yield (frame_idx, frame_results)

    def reset_session(self, session_id: str):
        session = self._SESSIONS.get(session_id)
        if session is None:
            raise RuntimeError(f"Unknown session {session_id}")
        session["prompts"] = {}
        session["state"] = {}
        return {"is_success": True}

    def close_session(self, session_id: str):
        self._SESSIONS.pop(session_id, None)
        return {"is_success": True}
