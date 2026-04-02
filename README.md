# torchkick

A Computer vision toolkit for spatio-temporal video analysis in football. Detects players, tracks them across frames, and projects positions onto a 2D pitch map. Personal project that may or may not turn useful one day. As you can see below, there's still some refinement to do.

![Demo](examples/images/demo.gif)

## Install

```bash
pip install -e .
```

For CUDA 12.8 (recommended):
```bash
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

## Usage

Analyze a match video (RF-DETR player detection, SigLIP zero-shot team clustering):
```bash
torchkick analyze -v match.mp4 --duration 60
```

With pitch keypoints for accurate homography and trained ReID weights for team classification:
```bash
torchkick analyze -v match.mp4 \
    --pitch-weights weights/keypoints/best.pt \
    --reid-weights  weights/reid/reid_student_best.pth
```

Other detector backends:
```bash
torchkick analyze -v match.mp4 --model-type rtdetr   # RT-DETR-R101
torchkick analyze -v match.mp4 --model-type yolo     # YOLO11-nano (fastest)
```

Python API:
```python
from torchkick import run_analysis

output = run_analysis(
    "match.mp4",
    model_type="rfdetr",        # default
    pitch_weights="weights/keypoints/best.pt",
    reid_weights="weights/reid/reid_student_best.pth",
    duration=60.0,
)
```

I've yet to release model weights, but still actively refining approach on homography projection and player tracking, so this can heavily change from current structure (also, not so sure whether it will even make it to the public).

## What it does

1. Detects players with RF-DETR (DINOv2 backbone) or YOLO/RT-DETR
2. Tracks them with BotSORT, optionally guided by DINOv2 ReID embeddings
3. Estimates camera homography from YOLO-pose / ViTPose pitch keypoints (29 or 32 landmarks)
4. Projects player feet positions onto pitch coordinates via Kalman-smoothed homography
5. Classifies teams via DINOv2+ArcFace ReID or SigLIP zero-shot clustering (no training required, can be improved by training)
6. Outputs a video with 2D pitch visualization and optional space-control heatmap
7. TBD features given accuracy and 3D->2D projection is 'accurate enough'.

## Training

### 1 — Download training data

**SoccerNet** (tracking + pitch calibration — requires a free SoccerNet account):
```bash
torchkick dataset -d tracking   -o data/soccernet/
torchkick dataset -d calibration -o data/soccernet/
```

**Roboflow Universe** (free API key required — sign up at roboflow.com):
```bash
# Add to .env in the project root:
# ROBOFLOW_API_KEY=your_key_here

# 4-class player detection — football-players-detection-3zvbc v20
torchkick dataset -d roboflow-players -o data/roboflow/players/

# 32-keypoint pitch landmarks — football-field-detection-f07vi v15
torchkick dataset -d roboflow-field -o data/roboflow/field/

# Any other Roboflow Universe dataset by URL slug
# (workspace and project visible at roboflow.com/<workspace>/<project>)
torchkick dataset -d roboflow \
    --workspace <workspace> --project <project> --version <n> \
    -o data/custom/

# COCO JSON export (for torchkick train detection)
torchkick dataset -d roboflow \
    --workspace roboflow-jvuqo --project football-players-detection-3zvbc \
    --version 20 --format coco -o data/roboflow/players_coco/
```

---

### 2 — Player detection

#### YOLO (fastest, good baseline — full CLI support)

```bash
# SoccerNet only
torchkick train yolo --data data/soccernet/tracking/train.zip --epochs 50

# Roboflow only (already in YOLO format, directory accepted directly)
torchkick train yolo --data data/roboflow/players/ --epochs 50

# Custom CVAT export (export as YOLO 1.1 from CVAT, pass the output dir)
torchkick train yolo --data data/custom/yolo_export/ --epochs 50
```

Combining multiple sources requires merging the YOLO dataset directories manually
(copy `images/` and `labels/` from each into a shared directory, then update `dataset.yaml`).

#### RT-DETR (higher accuracy — combines any sources with a single command)

`torchkick train detection` accepts any non-empty subset of SoccerNet, Roboflow, and CVAT.
No manual dataset merging required — all sources are mixed at training time.

Export formats required:
- **Roboflow** → Export dataset → **COCO JSON**
- **CVAT** → Export dataset → **COCO 1.0**

```bash
# SoccerNet only
torchkick train detection \
    --soccernet data/soccernet/tracking/train.zip

# Roboflow only
torchkick train detection \
    --roboflow-json data/roboflow/players/train/_annotations.coco.json \
    --roboflow-images data/roboflow/players/train/

# CVAT custom annotations only
torchkick train detection \
    --cvat-json data/custom/annotations.json \
    --cvat-images data/custom/images/

# SoccerNet + Roboflow
torchkick train detection \
    --soccernet data/soccernet/tracking/train.zip \
    --roboflow-json data/roboflow/players/train/_annotations.coco.json \
    --roboflow-images data/roboflow/players/train/

# SoccerNet + CVAT
torchkick train detection \
    --soccernet data/soccernet/tracking/train.zip \
    --cvat-json data/custom/annotations.json \
    --cvat-images data/custom/images/

# Roboflow + CVAT
torchkick train detection \
    --roboflow-json data/roboflow/players/train/_annotations.coco.json \
    --roboflow-images data/roboflow/players/train/ \
    --cvat-json data/custom/annotations.json \
    --cvat-images data/custom/images/

# All three combined
torchkick train detection \
    --soccernet data/soccernet/tracking/train.zip \
    --roboflow-json data/roboflow/players/train/_annotations.coco.json \
    --roboflow-images data/roboflow/players/train/ \
    --cvat-json data/custom/annotations.json \
    --cvat-images data/custom/images/ \
    --epochs 50
```

---

### 3 — Pitch keypoint detection

#### YOLO-pose (fast, ~3 ms/frame — recommended for real-time inference)

The Roboflow field dataset downloads in YOLO-pose format and comes with a `data.yaml`:

```bash
# Roboflow field keypoints (32 kp) — data.yaml included in download
torchkick train yolo-keypoints --data data/roboflow/field/data.yaml --epochs 100

# Custom YOLO-pose dataset (any YOLO-pose YAML works)
torchkick train yolo-keypoints --data data/custom/pitch_keypoints.yaml --imgsz 640 --epochs 100
```

`mosaic=0.0` is set automatically — omitting this degrades keypoint AP by shuffling spatial
relationships between landmarks.

The default model produces 29 keypoints (SoccerNet convention). For 32-keypoint models
(Roboflow field dataset):

```bash
torchkick train yolo-keypoints \
    --data data/roboflow/field/data.yaml \
    --base-model yolo11n-pose.pt \
    --epochs 100
```

Then load at inference with `num_keypoints=32`:

```python
from torchkick.models import YOLOPoseKeypointDetector
detector = YOLOPoseKeypointDetector("weights/keypoints/best.pt", num_keypoints=32)
```

#### ViTPose-L (highest accuracy — annotation/offline pipelines)

```bash
# SoccerNet calibration zip only
torchkick train keypoints --data data/soccernet/calibration/train.zip --epochs 100
```

---

### 4 — Player re-identification

ReID requires player crop images. Extract them by running the analysis pipeline first,
or use the `torchkick label grounded-sam` command to generate labeled crops from video.

Expected directory structure for supervised training:
```
data/reid/
  0/          # home team crops
    frame_001_id_3.jpg
    ...
  1/          # away team crops
    ...
  2/          # referee crops
    ...
```

For self-supervised (SSL) training, organise by track ID instead of team label:
```
data/reid_ssl/
  track_007/
    crop_0001.jpg
    ...
  track_012/
    ...
```

```bash
# Stage 1: supervised ArcFace metric learning
torchkick train reid --stage supervised \
    --data-dir data/reid/ \
    --epochs 30 \
    --num-classes 3

# Stage 2: BYOL tracklet SSL (optional, improves generalisation)
torchkick train reid --stage ssl \
    --data-dir data/reid_ssl/ \
    --epochs 20

# Distil ViT-L teacher → ViT-S student for real-time inference (~3 ms/frame)
torchkick train distill \
    --teacher-weights weights/reid/reid_supervised_best.pth \
    --data-dir data/reid/ \
    --epochs 30
```

The distilled student (`reid_student_best.pth`) is the file used with `--reid-weights`
at inference time.

---

### Training recipe summary

| Goal | Data source | Command |
|---|---|---|
| Player detector (fast) | SoccerNet | `torchkick train yolo --data soccernet/train.zip` |
| Player detector (fast) | Roboflow | `torchkick train yolo --data data/roboflow/players/` |
| Player detector (fast) | Custom CVAT (YOLO export) | `torchkick train yolo --data data/custom/` |
| Player detector (accurate) | Any / combined | `torchkick train detection --soccernet ... --roboflow-json ... --cvat-json ...` |
| Pitch keypoints (real-time) | Roboflow / custom YAML | `torchkick train yolo-keypoints --data pitch.yaml` |
| Pitch keypoints (accurate) | SoccerNet calibration | `torchkick train keypoints --data calibration.zip` |
| ReID embeddings | Labeled crops | `torchkick train reid --stage supervised --data-dir data/reid/` |
| ReID student (inference) | Same crops | `torchkick train distill --teacher-weights ...` |


---

## Remote training on RunPod

The fastest way to train all models is to spin up a GPU pod on RunPod, run
the training script, and SCP the weights back.

### 1 — Start a RunPod pod

Recommended template: **RunPod PyTorch 2.x** with at least an **A10G** (24 GB).
Note the pod IP and SSH port from the RunPod dashboard.

### 2 — Copy your `.env` to the pod

```bash
# From your local machine
scp -P <port> .env root@<runpod-ip>:/workspace/torchkick/.env
```

Or set the key directly in the pod's environment before running the script:

```bash
export ROBOFLOW_API_KEY=your_key_here
```

### 3 — Run the training script on the pod

SSH in and run:

```bash
ssh root@<runpod-ip> -p <port>

# Roboflow datasets only (no SoccerNet account needed) — ~2 h on A10G
bash <(curl -fsSL https://raw.githubusercontent.com/eirikbaekkelund/torchkick/main/scripts/train_remote.sh) --roboflow-only

# Or clone first and run locally
git clone https://github.com/eirikbaekkelund/torchkick.git /workspace/torchkick
bash /workspace/torchkick/scripts/train_remote.sh --roboflow-only
```

With SoccerNet data as well (needs SoccerNet credentials set in the pod):

```bash
bash /workspace/torchkick/scripts/train_remote.sh
```

With ReID training (requires labelled player crops copied to the pod):

```bash
# Copy crops from local first
scp -P <port> -r data/reid/ root@<runpod-ip>:/workspace/torchkick/data/reid/

bash /workspace/torchkick/scripts/train_remote.sh \
    --roboflow-only \
    --reid-crops-dir data/reid/
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--roboflow-only` | off | Skip SoccerNet downloads |
| `--skip-detection` | off | Skip player detector training |
| `--skip-keypoints` | off | Skip pitch keypoint training |
| `--reid-crops-dir <path>` | — | Enable ReID training with crops at this path |
| `--epochs-detection <n>` | 50 | Player detector epochs |
| `--epochs-keypoints <n>` | 100 | Pitch keypoint epochs |
| `--batch <n>` | 32 | Batch size |

### 4 — Pull weights back to your machine

```bash
# From your LOCAL machine, from the repo root
bash scripts/pull_weights.sh <runpod-ip> <ssh-port>
```

This rsync-pulls `weights_export/` from the pod into your local `weights/`
and prints the ready-to-run `torchkick analyze` command with all weight paths filled in.

### 5 — Run inference with trained weights

```bash
torchkick analyze -v match.mp4 \
    --model weights/player_detector_yolo.pt --model-type yolo \
    --pitch-weights weights/pitch_keypoints_yolo.pt \
    --reid-weights  weights/reid_student.pth
```

