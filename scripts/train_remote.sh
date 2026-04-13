#!/usr/bin/env bash
# =============================================================================
# train_remote.sh — Full training pipeline for a RunPod / remote GPU server.
#
# Usage:
#   bash scripts/train_remote.sh [OPTIONS]
#
# Options:
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
#   1. Player detector  (YOLO11n on SoccerNet data)
#   2. Pitch keypoints  (DINOv2 heatmap on SoccerNet calibration data)
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
LOG_FILE="$WORKSPACE/torchkick_train.log"

# Defaults
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

# ── Self-detach: survive SSH disconnects ─────────────────────────────────────
# Re-exec inside tmux (reattachable) or nohup (log-only) so the job lives past
# terminal close. Skip when already detached or running non-interactively.
if [[ "${TORCHKICK_DETACHED:-0}" != "1" ]] && [[ -t 1 ]]; then
    export TORCHKICK_DETACHED=1
    if command -v tmux &>/dev/null; then
        SESSION="torchkick_train"
        # Kill any stale session with the same name
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        tmux new-session -d -s "$SESSION" \
            "bash $(realpath "$0") $* 2>&1 | tee '$LOG_FILE'; echo '=== DONE ==='"
        echo "Training started in tmux session '$SESSION'"
        echo "  Re-attach:  tmux attach -t $SESSION"
        echo "  Tail log:   tail -f $LOG_FILE"
        exit 0
    else
        nohup bash "$(realpath "$0")" "$@" > "$LOG_FILE" 2>&1 &
        echo $! > "$WORKSPACE/torchkick_train.pid"
        echo "Training started in background (PID $!)"
        echo "  Tail log:   tail -f $LOG_FILE"
        echo "  Check:      kill -0 \$(cat $WORKSPACE/torchkick_train.pid) && echo running"
        exit 0
    fi
fi

# Redirect all output to log file when running detached (nohup path)
if [[ "${TORCHKICK_DETACHED:-0}" == "1" ]] && [[ ! -t 1 ]]; then
    # Already captured by nohup redirect; just add timestamps to stdout
    exec > >(while IFS= read -r line; do echo "[$(date '+%H:%M:%S')] $line"; done) 2>&1
fi

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

# ── 2. Install ───────────────────────────────────────────────────────────────
log "Installing torchkick"
pip install -q uv
uv pip install --system -e ".[soccernet,reid,training]" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
ok "torchkick installed"

# ── 3. Download datasets ─────────────────────────────────────────────────────
log "Downloading SoccerNet datasets (requires SoccerNet account)"
# train split only — test adds ~8 GB not needed for training; 2023 edition skipped by default
torchkick dataset -d tracking    -o data/soccernet/ --splits train
torchkick dataset -d calibration -o data/soccernet/ --splits train
ok "SoccerNet datasets ready"

# ── 4. Train player detector ─────────────────────────────────────────────────
if [ "$SKIP_DETECTION" -eq 0 ]; then
    log "Training player detector (YOLO11n)"
    SOCCERNET_TRACKING_ZIP="data/soccernet/tracking/tracking/train.zip"
    torchkick train yolo \
        --data   "$SOCCERNET_TRACKING_ZIP" \
        --epochs "$EPOCHS_DETECTION" \
        --batch-size "$BATCH" \
        --base-model yolo11n.pt
fi

# ── 5. Train pitch keypoints ─────────────────────────────────────────────────
if [ "$SKIP_KEYPOINTS" -eq 0 ]; then
    # SDK saves to data/soccernet/calibration/calibration/train.zip
    SOCCERNET_CALIB_ZIP="data/soccernet/calibration/calibration/train.zip"
    if [ -f "$SOCCERNET_CALIB_ZIP" ]; then
        log "Training DINOv2 heatmap pitch keypoints (SoccerNet)"
        torchkick train pitch-heatmap \
            --soccernet-calibration-dir data/soccernet/calibration/ \
            --epochs "$EPOCHS_KEYPOINTS" \
            --batch-size "$BATCH"
        ok "Pitch keypoints trained"
    else
        echo "WARNING: SoccerNet calibration data not found, skipping keypoint training."
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

# Pitch keypoints
KP_BEST=$(find weights/pitch_heatmap/ -name "best.pt" 2>/dev/null | head -1)
if [ -n "$KP_BEST" ]; then
    cp "$KP_BEST" weights_export/pitch_keypoints_heatmap.pt
    ok "Pitch keypoints: weights_export/pitch_keypoints_heatmap.pt"
fi

# ViTPose (if trained)
if [ -f "weights/keypoints_vitpose/vitpose_best.pth" ]; then
    cp weights/keypoints_vitpose/vitpose_best.pth weights_export/pitch_keypoints_vitpose.pth
    ok "ViTPose keypoints: weights_export/pitch_keypoints_vitpose.pth"
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
