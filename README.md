# torchkick

Computer vision toolkit for spatio-temporal football video analysis. Detects players, tracks them across frames, estimates camera homography from pitch keypoints, and projects positions onto a 2D pitch map.

![Demo](examples/images/demo.gif)

## Install

```bash
# CPU / MPS (Mac)
uv pip install -e ".[annotation]"

# CUDA 12.8
uv pip install -e ".[annotation]" --extra-index-url https://download.pytorch.org/whl/cu128
```

Optional extras: `soccernet`, `tracking`, `reid`, `training`, `labeling`, `dev`. Install all with `.[all]`.

---

## Inference

```bash
torchkick analyze \
    -v match.mp4 \
    --yolo-weights models/players/yolo11l.pt \
    --pitch-weights models/keypoints/heatmap.pt \
    --reid-weights  models/reid/reid_student_best.pth
```

When `--reid-weights` is omitted, SigLIP zero-shot clustering is used automatically (no training required).

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--yolo-weights` | required | YOLO player detection weights |
| `--pitch-weights` | — | DINOv2 heatmap pitch keypoint weights |
| `--reid-weights` | — | ReID checkpoint (omit → SigLIP zero-shot) |
| `--duration` / `-d` | full video | Max seconds to process |
| `--conf` | 0.25 | Player detection confidence threshold |
| `--homography-interval` | 1 | Frames between homography updates |
| `--reid-interval` | 5 | Frames between embedding updates |
| `--no-overlay` | off | Disable pitch line wireframe overlay |
| `--debug-anchors` | off | Draw anchor players used for camera motion |

Python API:

```python
from torchkick.inference import run_analysis

output = run_analysis(
    video_path="match.mp4",
    yolo_weights="models/players/yolo11l.pt",
    pitch_weights="models/keypoints/heatmap.pt",
    reid_weights="models/reid/reid_student_best.pth",
    duration=60.0,
)
```

---

## Annotation tool

Self-hostable web tool for semi-automated video annotation. Upload a video, pick a sampling interval, and auto-annotate every frame with player bboxes and 30-point pitch keypoints. Correct annotations in the browser, export YOLO-format labels.

```bash
# Local (MPS / CPU)
uv pip install -e ".[annotation]"
uvicorn annotation_tool.server:app --host 0.0.0.0 --port 8080 --workers 1
```

```bash
# GPU server
docker compose up --build
```

Open `http://localhost:8080`.

Environment variables (all optional — server auto-detects weights under `models/`):

| Variable | Default | Description |
|---|---|---|
| `PLAYER_WEIGHTS` | `models/players/yolo11l.pt` | YOLO player detection weights |
| `PITCH_WEIGHTS` | `models/keypoints/heatmap.pt` | DINOv2 heatmap pitch keypoint weights |
| `DEVICE` | auto (cuda → mps → cpu) | Inference device |
| `UPLOAD_DIR` | `./uploads` | Directory for uploaded videos |

**Features:**
- Configurable sampling interval (1 / 5 / 10 / 20 / 30 / 60 s)
- Background auto-annotation of all frames on upload with progress bar
- Dual canvas: frame view (left) + 2D pitch minimap (right)
- Select, drag, draw, delete annotations — zoom/pan for accuracy
- Shift-click multi-select on the pitch minimap for batch delete
- Pitch flip and undo/redo
- Session persists across browser refreshes; annotations written to disk on every save

**Export format** (`Export ZIP` button):
```
images/     {frame:06d}.jpg
players/    {frame:06d}.txt   # YOLO detection: class_id cx cy w h
keypoints/  {frame:06d}.txt   # YOLO-pose: 0 cx cy w h + 30×(x y vis)
```

---

## Training

### 1 — Download training data

**SoccerNet** (tracking + pitch calibration — requires a free SoccerNet account):
```bash
torchkick dataset -d tracking    -o data/soccernet/
torchkick dataset -d calibration -o data/soccernet/
```

---

### 2 — Player detection (YOLO)

```bash
# SoccerNet zip
torchkick train yolo --data data/soccernet/tracking/train.zip --epochs 100

# Custom YOLO directory
torchkick train yolo --data data/my_players/ --epochs 100

# Pre-extracted SoccerNet directory (faster than zip)
torchkick train yolo --soccernet-dir data/soccernet/tracking/ --epochs 100
```

Key flags: `--base-model yolo11l.pt` (default), `--batch-size`, `--frame-stride` (thin the dataset), `--save-dir`.

---

### 3 — Pitch keypoint detection (DINOv2 heatmap)

30-keypoint pitch schema. ViT-S/14 or ViT-B/14 backbone with CNN heatmap decoder.

```bash
# From annotation tool export (rename keypoints/ → labels/train/, images/ → images/train/)
torchkick train pitch-heatmap --data data/my_annotations/ --epochs 100

# From SoccerNet calibration directory (auto-converts format)
torchkick train pitch-heatmap \
    --soccernet-calibration-dir data/soccernet/calibration/ \
    --epochs 100 --batch-size 8

# Fine-tune from existing checkpoint
torchkick train pitch-heatmap \
    --data data/my_annotations/ \
    --base-model models/keypoints/heatmap.pt \
    --epochs 30 --batch-size 8
```

Key flags: `--backbone dinov2_vits14|dinov2_vitb14`, `--min-keypoints` (filter close-ups), `--imgsz` (must be multiple of 14), `--wandb-project`.

Dataset layout expected by `--data`:
```
data_dir/
  images/train/   *.jpg
  images/valid/   *.jpg
  labels/train/   *.txt   # YOLO-pose 30-keypoint format
  labels/valid/   *.txt
```

---

### 4 — ReID (player re-identification)

DINOv2+LoRA+ArcFace metric learning.

```bash
# Supervised (labelled crops by team: 0/home, 1/away, 2/referee)
torchkick train reid \
    --data-dir data/reid/ \
    --epochs 30 --num-classes 3

# BYOL self-supervised (crops by track ID, no labels needed)
torchkick train reid \
    --stage ssl \
    --data-dir data/tracklets/ \
    --epochs 20

# Auto-label from raw video (SigLIP zero-shot → pseudo-labels → ArcFace)
torchkick train reid-video \
    --video match.mp4 \
    --yolo-weights models/players/yolo11l.pt \
    --epochs 20
```

---

### Training recipe summary

| Goal | Command |
|---|---|
| Player detector | `torchkick train yolo --data <zip or dir>` |
| Pitch keypoints (from annotation export) | `torchkick train pitch-heatmap --data <dir>` |
| Pitch keypoints (from SoccerNet) | `torchkick train pitch-heatmap --soccernet-calibration-dir <dir>` |
| ReID (labelled) | `torchkick train reid --data-dir data/reid/` |
| ReID (no labels) | `torchkick train reid-video --video match.mp4 --yolo-weights best.pt` |

---

## Remote training on RunPod

### 1 — Start a pod

Recommended: **RunPod PyTorch 2.x**, at least **A10G** (24 GB).

### 2 — Run the training script

```bash
ssh root@<runpod-ip> -p <port>

bash scripts/train_remote.sh

# With ReID (copy crops to pod first)
scp -P <port> -r data/reid/ root@<runpod-ip>:/workspace/torchkick/data/reid/
bash scripts/train_remote.sh --reid-crops-dir data/reid/
```

Script flags:

| Flag | Description |
|---|---|
| `--skip-detection` | Skip player detector training |
| `--skip-keypoints` | Skip pitch keypoint training |
| `--reid-crops-dir <path>` | Enable ReID training |
| `--epochs-detection <n>` | Player detector epochs (default 50) |
| `--epochs-keypoints <n>` | Pitch keypoint epochs (default 100) |
| `--batch <n>` | Batch size (default 32) |

### 3 — Pull weights back

```bash
# From local machine, repo root
bash scripts/pull_weights.sh <runpod-ip> <ssh-port>
```

---

## Project structure

```
src/torchkick/
  models/
    player.py           PlayerDetector (YOLO wrapper)
    pitch.py            HeatmapPitchDetector (DINOv2 + CNN decoder)
    reid.py             DINOv2+LoRA+ArcFace ReID model
  tracking/
    homography.py       HomographyEstimator (30-keypoint pitch schema)
    identity.py         IdentityAssigner — track→player role mapping
    models.py           Track / TrackStore dataclasses
    pitch_viz.py        PitchVisualizer — 2D minimap renderer
    team_embedder.py    GameTeamEmbedder — jersey crop clustering
    trajectory.py       Kalman-smoothed trajectory smoothing
  training/
    data/
      detection_dataset.py    MixedDetectionDataset (SoccerNet + COCO)
      pitch_heatmap_dataset.py PitchHeatmapDataset (YOLO-pose → Gaussian heatmaps)
      reid_dataset.py         ReID crop datasets (supervised + SSL)
      keypoint_dataset.py     KeypointAugDataset (SoccerNet calibration)
    train_yolo_detection.py   YOLO training loop
    train_pitch_heatmap.py    DINOv2 heatmap training loop
    train_reid.py             ArcFace / BYOL training loop
  soccernet/
    download.py         SoccerNet dataset downloaders
    tracking_data.py    PlayerTrackingDataset
    calibration_data.py LineKeypointDataset
  utils/
    crops.py            Player crop extraction
    video.py            Video I/O utilities
    visualization.py    Overlay rendering helpers
  inference.py          run_analysis() — full pipeline entry point
  cli.py                torchkick CLI

annotation_tool/
  server.py             FastAPI backend
  static/
    index.html          UI layout
    annotator.js        Canvas annotation logic

scripts/
  train_remote.sh       End-to-end remote training on RunPod
  pull_weights.sh       rsync weights from RunPod to local
```
