"""
FastAPI backend for the torchkick semi-automated annotation tool.

Start with:
    uvicorn annotation_tool.server:app --host 0.0.0.0 --port 8080

Environment variables:
    PLAYER_WEIGHTS  path to YOLO player detection weights (.pt)
    PITCH_WEIGHTS   path to DINOv2 pitch keypoint weights (.pt)
    DEVICE          "cuda" | "mps" | "cpu"  (default: auto-detect)
    UPLOAD_DIR      directory for uploaded videos  (default: ./uploads)

If PLAYER_WEIGHTS / PITCH_WEIGHTS are not set, the server searches for
default paths relative to the working directory:
    models/players/yolo11l_best.pt
    models/keypoints/heatmap_best.pt
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_PLAYER_WEIGHTS = "models/players/yolo11l_best.pt"
_DEFAULT_PITCH_WEIGHTS  = "models/keypoints/heatmap_best.pt"


def _resolve_weights(env_var: str, default: str) -> Optional[str]:
    """Return weights path from env var, falling back to default if it exists."""
    path = os.getenv(env_var) or default
    return path if Path(path).exists() else None


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


PLAYER_WEIGHTS: Optional[str] = _resolve_weights("PLAYER_WEIGHTS", _DEFAULT_PLAYER_WEIGHTS)
PITCH_WEIGHTS:  Optional[str] = _resolve_weights("PITCH_WEIGHTS",  _DEFAULT_PITCH_WEIGHTS)
DEVICE: str = os.getenv("DEVICE", _default_device())
UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

print(f"[annotator] device={DEVICE}")
print(f"[annotator] player weights: {PLAYER_WEIGHTS or 'NOT FOUND (auto-annotation disabled)'}")
print(f"[annotator] pitch  weights: {PITCH_WEIGHTS  or 'NOT FOUND (auto-annotation disabled)'}")

# ---------------------------------------------------------------------------
# Lazy model singletons
# ---------------------------------------------------------------------------

_player_detector = None
_pitch_detector = None


def _get_player_detector():
    global _player_detector
    if _player_detector is None and PLAYER_WEIGHTS and Path(PLAYER_WEIGHTS).exists():
        from torchkick.models.player import PlayerDetector
        _player_detector = PlayerDetector(PLAYER_WEIGHTS, device=DEVICE, conf_threshold=0.7)
    return _player_detector


def _get_pitch_detector():
    global _pitch_detector
    if _pitch_detector is None and PITCH_WEIGHTS and Path(PITCH_WEIGHTS).exists():
        from torchkick.models.pitch import HeatmapPitchDetector
        _pitch_detector = HeatmapPitchDetector(PITCH_WEIGHTS, device=DEVICE, conf_threshold=0.5)
    return _pitch_detector


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class KeypointAnno(BaseModel):
    index: int          # 0-31 Roboflow schema
    x: float            # pixel x in original frame
    y: float            # pixel y in original frame
    visible: bool = True
    confidence: float = 0.0  # 0 = manually placed


class BboxAnno(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int        # 0=player 1=goalkeeper 2=referee 3=ball
    confidence: float = 0.0


class FrameAnno(BaseModel):
    frame_idx: int
    width: int
    height: int
    keypoints: List[KeypointAnno] = []
    bboxes: List[BboxAnno] = []


class VideoMeta(BaseModel):
    video_id: str
    fps: float
    duration: float
    total_frames: int
    width: int
    height: int
    path: str


class SampleRequest(BaseModel):
    interval_s: int


# ---------------------------------------------------------------------------
# In-memory session state
# ---------------------------------------------------------------------------

_videos: Dict[str, VideoMeta] = {}
_frame_lists: Dict[str, List[int]] = {}
_annotations: Dict[str, Dict[int, FrameAnno]] = {}
_progress: Dict[str, Dict] = {}  # video_id -> {done, total, running}

# ---------------------------------------------------------------------------
# Disk persistence helpers
# ---------------------------------------------------------------------------


def _anno_dir(video_id: str) -> Path:
    return UPLOAD_DIR / video_id / "annotations"


def _save_anno_to_disk(video_id: str, idx: int, anno: FrameAnno) -> None:
    d = _anno_dir(video_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{idx}.json").write_text(anno.model_dump_json())


def _delete_anno_from_disk(video_id: str, idx: int) -> None:
    (anno_dir := _anno_dir(video_id) / f"{idx}.json").unlink(missing_ok=True)


def _ensure_video_loaded(video_id: str) -> Optional[VideoMeta]:
    """Return VideoMeta from memory, reloading from disk after a server restart."""
    if video_id in _videos:
        return _videos[video_id]
    meta_path = UPLOAD_DIR / video_id / "meta.json"
    if not meta_path.exists():
        return None
    meta = VideoMeta.model_validate_json(meta_path.read_text())
    _videos[video_id] = meta
    # Reload annotations saved in previous sessions
    if video_id not in _annotations:
        _annotations[video_id] = {}
        for p in sorted(_anno_dir(video_id).glob("*.json")):
            try:
                anno = FrameAnno.model_validate_json(p.read_text())
                _annotations[video_id][anno.frame_idx] = anno
            except Exception:
                pass
    return meta

# ---------------------------------------------------------------------------
# 32 Roboflow keypoint schema
# ---------------------------------------------------------------------------

# Pitch constants (must match homography.py)
_PITCH_LENGTH = 105.0
_PITCH_WIDTH = 68.0
_HALF_LENGTH = 52.5
_HALF_WIDTH = 34.0
_RF_L = 120.0
_RF_W = 70.0
_RF_PBW = 41.0
_RF_PBD = 20.15
_RF_GBW = 18.32
_RF_GBD = 5.50
_RF_CCR = 9.15
_RF_PS = 11.0
_RF_HX = _RF_L / 2
_RF_HY = _RF_W / 2


def _rf(x_raw: float, y_raw: float):
    x = (x_raw / _RF_L) * _PITCH_LENGTH - _HALF_LENGTH
    y = (y_raw / _RF_W) * _PITCH_WIDTH - _HALF_WIDTH
    return x, y


_ROBOFLOW_VERTICES = [
    _rf(0, 0),                                        # 0  top-left corner
    _rf(0, (_RF_W - _RF_PBW) / 2),                   # 1  left penalty box top
    _rf(0, (_RF_W - _RF_GBW) / 2),                   # 2  left goal box top
    _rf(0, (_RF_W + _RF_GBW) / 2),                   # 3  left goal box bottom
    _rf(0, (_RF_W + _RF_PBW) / 2),                   # 4  left penalty box bottom
    _rf(0, _RF_W),                                    # 5  bottom-left corner
    _rf(_RF_GBD, (_RF_W - _RF_GBW) / 2),             # 6  left goal box front top
    _rf(_RF_GBD, (_RF_W + _RF_GBW) / 2),             # 7  left goal box front bottom
    _rf(_RF_PS, _RF_HY),                              # 8  left penalty spot
    _rf(_RF_PBD, (_RF_W - _RF_PBW) / 2),             # 9  left penalty box front top
    _rf(_RF_PBD, (_RF_W - _RF_GBW) / 2),             # 10 left penalty box inner top
    _rf(_RF_PBD, (_RF_W + _RF_GBW) / 2),             # 11 left penalty box inner bottom
    _rf(_RF_PBD, (_RF_W + _RF_PBW) / 2),             # 12 left penalty box front bottom
    _rf(_RF_HX, 0),                                   # 13 halfway line top
    _rf(_RF_HX, _RF_HY - _RF_CCR),                   # 14 centre circle top
    _rf(_RF_HX, _RF_HY + _RF_CCR),                   # 15 centre circle bottom
    _rf(_RF_HX, _RF_W),                               # 16 halfway line bottom
    _rf(_RF_L - _RF_PBD, (_RF_W - _RF_PBW) / 2),    # 17 right penalty box front top
    _rf(_RF_L - _RF_PBD, (_RF_W - _RF_GBW) / 2),    # 18 right penalty box inner top
    _rf(_RF_L - _RF_PBD, (_RF_W + _RF_GBW) / 2),    # 19 right penalty box inner bottom
    _rf(_RF_L - _RF_PBD, (_RF_W + _RF_PBW) / 2),    # 20 right penalty box front bottom
    _rf(_RF_L - _RF_PS, _RF_HY),                     # 21 right penalty spot
    _rf(_RF_L - _RF_GBD, (_RF_W - _RF_GBW) / 2),    # 22 right goal box front top
    _rf(_RF_L - _RF_GBD, (_RF_W + _RF_GBW) / 2),    # 23 right goal box front bottom
    _rf(_RF_L, 0),                                    # 24 top-right corner
    _rf(_RF_L, (_RF_W - _RF_PBW) / 2),               # 25 right penalty box top
    _rf(_RF_L, (_RF_W - _RF_GBW) / 2),               # 26 right goal box top
    _rf(_RF_L, (_RF_W + _RF_GBW) / 2),               # 27 right goal box bottom
    _rf(_RF_L, (_RF_W + _RF_PBW) / 2),               # 28 right penalty box bottom
    _rf(_RF_L, _RF_W),                                # 29 bottom-right corner
    _rf(_RF_HX - _RF_CCR, _RF_HY),                   # 30 centre circle left
    _rf(_RF_HX + _RF_CCR, _RF_HY),                   # 31 centre circle right
]

_KP_NAMES = [
    "top-left corner",
    "left penalty box top",
    "left goal box top",
    "left goal box bottom",
    "left penalty box bottom",
    "bottom-left corner",
    "left goal box front top",
    "left goal box front bottom",
    "left penalty spot",
    "left penalty box front top",
    "left penalty box inner top",
    "left penalty box inner bottom",
    "left penalty box front bottom",
    "halfway line top",
    "centre circle top",
    "centre circle bottom",
    "halfway line bottom",
    "right penalty box front top",
    "right penalty box inner top",
    "right penalty box inner bottom",
    "right penalty box front bottom",
    "right penalty spot",
    "right goal box front top",
    "right goal box front bottom",
    "top-right corner",
    "right penalty box top",
    "right goal box top",
    "right goal box bottom",
    "right penalty box bottom",
    "bottom-right corner",
    "centre circle left",
    "centre circle right",
]

# PitchVisualizer canvas geometry (matches pitch_viz.py defaults)
_VIZ_MARGIN = 50
_VIZ_SCALE = 10.0


def _pitch_to_canvas(x: float, y: float):
    px = int(_VIZ_MARGIN + (x + _HALF_LENGTH) * _VIZ_SCALE)
    py = int(_VIZ_MARGIN + (_HALF_WIDTH - y) * _VIZ_SCALE)
    return px, py


# Penalty spots only — too occluded by players to annotate reliably.
# All box corners (16m box, goal box), arc intersections, halfway line, centre circle kept.
_PENALTY_BOX_INDICES: frozenset = frozenset([8, 21])

_PITCH_KP_INFO = [
    {
        "index": i,
        "name": _KP_NAMES[i],
        "pitch_x": float(_ROBOFLOW_VERTICES[i][0]),
        "pitch_y": float(_ROBOFLOW_VERTICES[i][1]),
        "canvas_px": _pitch_to_canvas(*_ROBOFLOW_VERTICES[i])[0],
        "canvas_py": _pitch_to_canvas(*_ROBOFLOW_VERTICES[i])[1],
    }
    for i in range(32)
    if i not in _PENALTY_BOX_INDICES
]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="torchkick annotation tool")

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    from fastapi.responses import FileResponse
    return FileResponse(str(_static_dir / "index.html"))


@app.get("/status")
async def status():
    return {
        "device": DEVICE,
        "player_model": PLAYER_WEIGHTS or None,
        "pitch_model": PITCH_WEIGHTS or None,
    }


@app.post("/upload", response_model=VideoMeta)
async def upload_video(file: UploadFile):
    video_id = str(uuid.uuid4())[:8]
    dest_dir = UPLOAD_DIR / video_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "video.mp4"

    content = await file.read()
    dest_path.write_bytes(content)

    cap = cv2.VideoCapture(str(dest_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps
    meta = VideoMeta(
        video_id=video_id,
        fps=fps,
        duration=duration,
        total_frames=total_frames,
        width=width,
        height=height,
        path=str(dest_path),
    )
    _videos[video_id] = meta
    _annotations[video_id] = {}
    (dest_dir / "meta.json").write_text(meta.model_dump_json())
    return meta


@app.post("/sample/{video_id}")
async def sample_frames(video_id: str, body: SampleRequest):
    meta = _ensure_video_loaded(video_id)
    if meta is None:
        raise HTTPException(404, "Video not found")
    # CAP_PROP_FRAME_COUNT often overcounts — find the true last readable frame
    cap = cv2.VideoCapture(meta.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, meta.total_frames - 1)
    ok, _ = cap.read()
    if not ok:
        # Binary search backwards for the last readable frame
        lo, hi = 0, meta.total_frames - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, _ = cap.read()
            if ok:
                lo = mid
            else:
                hi = mid - 1
        true_last = lo
    else:
        true_last = meta.total_frames - 1
    cap.release()
    step = max(1, int(meta.fps * body.interval_s))
    indices = list(range(0, true_last + 1, step))
    _frame_lists[video_id] = indices
    asyncio.create_task(_annotate_all_bg(video_id, indices))
    return {"frame_indices": indices}


@app.get("/progress/{video_id}")
async def get_progress(video_id: str):
    return _progress.get(video_id, {"done": 0, "total": 0, "running": False})


@app.get("/frame/{video_id}/{idx}")
async def get_frame(video_id: str, idx: int):
    meta = _ensure_video_loaded(video_id)
    if meta is None:
        raise HTTPException(404, "Video not found")
    cap = cv2.VideoCapture(meta.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, f"Frame {idx} not readable")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/pitch/keypoints")
async def pitch_keypoints():
    return _PITCH_KP_INFO


@app.get("/pitch/image")
async def pitch_image():
    from torchkick.tracking.pitch_viz import PitchVisualizer
    viz = PitchVisualizer(use_gpu=False)
    img = viz._draw_pitch()
    _, buf = cv2.imencode(".png", img)
    return Response(content=buf.tobytes(), media_type="image/png")


def _run_annotation(video_id: str, idx: int, frame_bgr) -> FrameAnno:
    """Sync: run models on an already-decoded frame and store the result."""
    meta = _ensure_video_loaded(video_id)
    anno = FrameAnno(frame_idx=idx, width=meta.width, height=meta.height)

    player_det = _get_player_detector()
    if player_det is not None:
        for d in player_det.detect(frame_bgr):
            x1, y1, x2, y2 = d.bbox
            anno.bboxes.append(BboxAnno(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                class_id=int(d.class_id), confidence=float(d.confidence),
            ))

    pitch_det = _get_pitch_detector()
    if pitch_det is not None:
        kps, confs = pitch_det.detect(frame_bgr)
        for i, (xy, conf) in enumerate(zip(kps, confs)):
            if conf > 0 and i not in _PENALTY_BOX_INDICES:
                anno.keypoints.append(KeypointAnno(
                    index=i, x=float(xy[0]), y=float(xy[1]),
                    visible=True, confidence=float(conf),
                ))

    _annotations.setdefault(video_id, {})[idx] = anno
    _save_anno_to_disk(video_id, idx, anno)
    return anno


async def _annotate_all_bg(video_id: str, indices: List[int]) -> None:
    """Background task: annotate every sampled frame, skipping already-done ones."""
    _progress[video_id] = {"done": 0, "total": len(indices), "running": True}
    meta = _ensure_video_loaded(video_id)
    if meta is None:
        _progress[video_id]["running"] = False
        return

    def _process_frame(idx: int) -> None:
        if _annotations.get(video_id, {}).get(idx) is not None:
            return  # already annotated (e.g. loaded from disk)
        cap = cv2.VideoCapture(meta.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok:
            _run_annotation(video_id, idx, frame_bgr)

    for idx in indices:
        try:
            await asyncio.to_thread(_process_frame, idx)
        except Exception as exc:
            print(f"[annotator] bg annotation failed frame {idx}: {exc}")
        _progress[video_id]["done"] += 1

    _progress[video_id]["running"] = False


@app.post("/annotate/{video_id}/{idx}", response_model=FrameAnno)
async def auto_annotate(video_id: str, idx: int):
    meta = _ensure_video_loaded(video_id)
    if meta is None:
        raise HTTPException(404, "Video not found")
    cap = cv2.VideoCapture(meta.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, f"Frame {idx} not readable")
    return await asyncio.to_thread(_run_annotation, video_id, idx, frame_bgr)


@app.get("/annotations/{video_id}/{idx}", response_model=FrameAnno)
async def get_annotations(video_id: str, idx: int):
    video_annos = _annotations.get(video_id, {})
    anno = video_annos.get(idx)
    if anno is None:
        raise HTTPException(404, "No annotations for this frame")
    return anno


@app.put("/annotations/{video_id}/{idx}", response_model=FrameAnno)
async def save_annotations(video_id: str, idx: int, anno: FrameAnno):
    if video_id not in _annotations:
        _annotations[video_id] = {}
    _annotations[video_id][idx] = anno
    _save_anno_to_disk(video_id, idx, anno)
    return anno


@app.delete("/annotations/{video_id}/{idx}")
async def delete_annotation(video_id: str, idx: int):
    meta = _ensure_video_loaded(video_id)
    if meta is None:
        raise HTTPException(404, "Video not found")
    _annotations.get(video_id, {}).pop(idx, None)
    _delete_anno_from_disk(video_id, idx)
    return {"deleted": idx}


@app.get("/export/{video_id}")
async def export_annotations(video_id: str):
    meta = _ensure_video_loaded(video_id)
    if meta is None:
        raise HTTPException(404, "Video not found")

    video_annos = _annotations.get(video_id, {})
    if not video_annos:
        raise HTTPException(400, "No annotations to export")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for frame_idx, anno in sorted(video_annos.items()):
            w, h = anno.width, anno.height
            fname = f"{frame_idx:06d}"

            # --- players label ---
            player_lines = []
            for bbox in anno.bboxes:
                cx = ((bbox.x1 + bbox.x2) / 2) / w
                cy = ((bbox.y1 + bbox.y2) / 2) / h
                bw = (bbox.x2 - bbox.x1) / w
                bh = (bbox.y2 - bbox.y1) / h
                player_lines.append(f"{bbox.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            zf.writestr(f"players/{fname}.txt", "\n".join(player_lines))

            # --- keypoints label (YOLO keypoint format, one pitch per frame) ---
            # Build full 32-point array; missing = 0 0 0
            kp_map: Dict[int, KeypointAnno] = {kp.index: kp for kp in anno.keypoints}
            parts = ["0 0.5 0.5 1.0 1.0"]
            for i in range(32):
                kp = kp_map.get(i)
                if kp and kp.visible:
                    xn = kp.x / w
                    yn = kp.y / h
                    parts.append(f"{xn:.6f} {yn:.6f} 2")
                else:
                    parts.append("0 0 0")
            zf.writestr(f"keypoints/{fname}.txt", " ".join(parts))

            # --- frame image ---
            cap = cv2.VideoCapture(meta.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            cap.release()
            if ok:
                _, img_buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                zf.writestr(f"images/{fname}.jpg", img_buf.tobytes())

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={video_id}_annotations.zip"},
    )
