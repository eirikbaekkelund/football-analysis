#!/usr/bin/env bash
# =============================================================================
# train_remote.sh — Full training pipeline for a RunPod / remote GPU server.
#
# Usage:
#   bash scripts/train_remote.sh [OPTIONS]
#
# Options:
#   --roboflow-only     Skip SoccerNet downloads (no SoccerNet account needed)
#   --skip-detection    Skip player detector training
#   --skip-keypoints    Skip pitch keypoint training
#   --skip-reid         Skip ReID training (requires --reid-crops-dir)
#   --reid-crops-dir    Directory of player crops organised by team label
#                       (0/home  1/away  2/ref). Required for ReID training.
#   --epochs-detection  Epochs for player detector (default 50)
#   --epochs-keypoints  Epochs for pitch keypoints (default 100)
#   --epochs-reid       Epochs for ReID supervised stage (default 30)
#   --epochs-distill    Epochs for ReID distillation (default 30)
#   --batch             Batch size for detection/keypoint training (default 32)
#
# What it trains:
#   1. Player detector  (YOLO11n on Roboflow players dataset)
#   2. Pitch keypoints  (YOLO11n-pose on Roboflow field dataset)
#   3. ReID embedder    (DINOv2+LoRA+ArcFace — only if --reid-crops-dir given)
#   4. ReID distillation → ViT-S student for real-time inference
#
# All weights are collected into weights/ at the end. SCP them back with:
#   bash scripts/pull_weights.sh <runpod-ip> <port>
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/eirikbaekkelund/torchkick.git"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/torchkick"

# Defaults
ROBOFLOW_ONLY=0
SKIP_DETECTION=0
SKIP_KEYPOINTS=0
SKIP_REID=1          # off by default — needs crops
REID_CROPS_DIR=""
EPOCHS_DETECTION=50
EPOCHS_KEYPOINTS=100
EPOCHS_REID=30
EPOCHS_DISTILL=30
BATCH=32

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --roboflow-only)   ROBOFLOW_ONLY=1 ;;
        --skip-detection)  SKIP_DETECTION=1 ;;
        --skip-keypoints)  SKIP_KEYPOINTS=1 ;;
        --skip-reid)       SKIP_REID=1 ;;
        --reid-crops-dir)  REID_CROPS_DIR="$2"; SKIP_REID=0; shift ;;
        --epochs-detection) EPOCHS_DETECTION="$2"; shift ;;
        --epochs-keypoints) EPOCHS_KEYPOINTS="$2"; shift ;;
        --epochs-reid)     EPOCHS_REID="$2"; shift ;;
        --epochs-distill)  EPOCHS_DISTILL="$2"; shift ;;
        --batch)           BATCH="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo -e "\n\033[1;36m==> $*\033[0m"; }
ok()  { echo -e "\033[1;32m  ✓ $*\033[0m"; }

# ── 1. Clone / update repo ───────────────────────────────────────────────────
log "Setting up repository"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only
    ok "Repository updated"
else
    git clone "$REPO_URL" "$REPO_DIR"
    ok "Repository cloned to $REPO_DIR"
fi
cd "$REPO_DIR"

# Copy .env if it doesn't exist on the remote (ROBOFLOW_API_KEY must be set)
if [ ! -f ".env" ] && [ -n "${ROBOFLOW_API_KEY:-}" ]; then
    echo "ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY" > .env
    ok ".env written from environment"
fi

if [ ! -f ".env" ]; then
    echo "ERROR: .env not found and ROBOFLOW_API_KEY not set in environment."
    echo "Either scp your .env to $REPO_DIR/.env or export ROBOFLOW_API_KEY=<key>"
    exit 1
fi

# ── 2. Install ───────────────────────────────────────────────────────────────
log "Installing torchkick"
pip install -q uv
uv pip install --system -e ".[reid,training]" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
# roboflow installed separately — its idna pin conflicts with the PyTorch index
pip install -q roboflow
ok "torchkick installed"

# ── 3. Download datasets ─────────────────────────────────────────────────────
log "Downloading Roboflow datasets"
torchkick dataset -d roboflow-players -o data/roboflow/players/
ok "Roboflow player detection dataset ready"

torchkick dataset -d roboflow-field -o data/roboflow/field/
ok "Roboflow field keypoints dataset ready"

if [ "$ROBOFLOW_ONLY" -eq 0 ]; then
    log "Downloading SoccerNet datasets (requires SoccerNet account)"
    # train split only — test adds ~8 GB not needed for training; 2023 edition skipped by default
    torchkick dataset -d tracking    -o data/soccernet/ --splits train
    torchkick dataset -d calibration -o data/soccernet/ --splits train
    ok "SoccerNet datasets ready"
fi

# ── 4. Train player detector ─────────────────────────────────────────────────
if [ "$SKIP_DETECTION" -eq 0 ]; then
    log "Training player detector (YOLO11n)"
    torchkick train yolo \
        --data   data/roboflow/players/ \
        --epochs "$EPOCHS_DETECTION" \
        --batch-size "$BATCH" \
        --base-model yolo11n.pt

    # If SoccerNet data is also available, train a combined model
    # SDK saves to data/soccernet/tracking/tracking/train.zip (nested by task name)
    SOCCERNET_TRACKING_ZIP="data/soccernet/tracking/tracking/train.zip"
    if [ "$ROBOFLOW_ONLY" -eq 0 ] && [ -f "$SOCCERNET_TRACKING_ZIP" ]; then
        log "Training combined detector (Roboflow + SoccerNet)"
        torchkick train detection \
            --soccernet "$SOCCERNET_TRACKING_ZIP" \
            --roboflow-json data/roboflow/players/train/_annotations.coco.json \
            --roboflow-images data/roboflow/players/train/ \
            --epochs "$EPOCHS_DETECTION" \
            --save-dir weights/detection_combined/
        ok "Combined RT-DETR detector trained"
    fi
fi

# ── 5. Train pitch keypoints ─────────────────────────────────────────────────
if [ "$SKIP_KEYPOINTS" -eq 0 ]; then
    log "Training YOLO-pose pitch keypoints"
    torchkick train yolo-keypoints \
        --data   data/roboflow/field/data.yaml \
        --epochs "$EPOCHS_KEYPOINTS" \
        --save-dir weights/keypoints_yolo/

    # SDK saves to data/soccernet/calibration/calibration/train.zip
    SOCCERNET_CALIB_ZIP="data/soccernet/calibration/calibration/train.zip"
    if [ "$ROBOFLOW_ONLY" -eq 0 ] && [ -f "$SOCCERNET_CALIB_ZIP" ]; then
        log "Training ViTPose pitch keypoints (SoccerNet)"
        torchkick train keypoints \
            --data "$SOCCERNET_CALIB_ZIP" \
            --epochs "$EPOCHS_KEYPOINTS" \
            --save-dir weights/keypoints_vitpose/
        ok "ViTPose keypoints trained"
    fi
fi

# ── 6. Train ReID (optional — needs labelled player crops) ───────────────────
if [ "$SKIP_REID" -eq 0 ]; then
    if [ -z "$REID_CROPS_DIR" ] || [ ! -d "$REID_CROPS_DIR" ]; then
        echo "WARNING: --reid-crops-dir not found, skipping ReID training."
    else
        log "Training ReID (supervised ArcFace)"
        torchkick train reid \
            --stage supervised \
            --data-dir "$REID_CROPS_DIR" \
            --epochs "$EPOCHS_REID" \
            --save-dir weights/reid/

        log "Distilling ReID teacher → student"
        torchkick train distill \
            --teacher-weights weights/reid/reid_supervised_best.pth \
            --data-dir "$REID_CROPS_DIR" \
            --epochs "$EPOCHS_DISTILL" \
            --save-dir weights/reid/
        ok "ReID student ready at weights/reid/reid_student_best.pth"
    fi
fi

# ── 7. Collect weights ───────────────────────────────────────────────────────
log "Collecting weights"
mkdir -p weights_export

# Player detector (YOLO) — find the best.pt from the most recent run
YOLO_BEST=$(find player_tracker_yolo11n/ -name "best.pt" 2>/dev/null | head -1)
if [ -n "$YOLO_BEST" ]; then
    cp "$YOLO_BEST" weights_export/player_detector_yolo.pt
    ok "Player detector: weights_export/player_detector_yolo.pt"
fi

# Pitch keypoints (YOLO-pose)
KP_BEST=$(find weights/keypoints_yolo/ -name "best.pt" 2>/dev/null | head -1)
if [ -n "$KP_BEST" ]; then
    cp "$KP_BEST" weights_export/pitch_keypoints_yolo.pt
    ok "Pitch keypoints: weights_export/pitch_keypoints_yolo.pt"
fi

# ViTPose (if trained)
if [ -f "weights/keypoints_vitpose/vitpose_best.pth" ]; then
    cp weights/keypoints_vitpose/vitpose_best.pth weights_export/pitch_keypoints_vitpose.pth
    ok "ViTPose keypoints: weights_export/pitch_keypoints_vitpose.pth"
fi

# RT-DETR combined (if trained)
if [ -f "weights/detection_combined/rtdetr_best.pth" ]; then
    cp weights/detection_combined/rtdetr_best.pth weights_export/player_detector_rtdetr.pth
    ok "RT-DETR detector: weights_export/player_detector_rtdetr.pth"
fi

# ReID student (if trained)
if [ -f "weights/reid/reid_student_best.pth" ]; then
    cp weights/reid/reid_student_best.pth weights_export/reid_student.pth
    ok "ReID student: weights_export/reid_student.pth"
fi

echo ""
log "Training complete. Weights in $REPO_DIR/weights_export/"
ls -lh weights_export/
echo ""
echo "SCP back to local machine:"
echo "  bash scripts/pull_weights.sh <runpod-ip> <ssh-port>"
