#!/usr/bin/env bash
# =============================================================================
# pull_weights.sh — Download trained weights from a RunPod server back to local.
#
# RunPod's SSH proxy does NOT support SCP/SFTP/rsync. This script instead:
#   1. SSHes into the pod and starts a temporary HTTP server on port 8000
#   2. Downloads each weight file via curl to ./weights/
#   3. Kills the HTTP server when done
#
# Usage (run from your LOCAL machine, from the repo root):
#   bash scripts/pull_weights.sh <runpod-ssh-host> [http-port]
#
# Example:
#   bash scripts/pull_weights.sh 9vvgwfgp10oosn-644121cc@ssh.runpod.io
#   bash scripts/pull_weights.sh 9vvgwfgp10oosn-644121cc@ssh.runpod.io 8000
#
# Prerequisites:
#   - Port 8000 (or your chosen port) must be exposed in the RunPod pod settings.
#   - The pod must have run train_remote.sh so weights_export/ exists.
#
# Alternatively, if you have direct IP access (not proxy):
#   DIRECT=1 bash scripts/pull_weights.sh <ip> <ssh-port>
# =============================================================================

set -euo pipefail

SSH_HOST="${1:?Usage: pull_weights.sh <runpod-ssh-host> [http-port]}"
HTTP_PORT="${2:-8000}"
LOCAL_DIR="${LOCAL_DIR:-./weights}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_EXPORT_DIR="/workspace/torchkick/weights_export"

SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"

# ── Direct IP/port mode (when SSH proxy supports SCP) ────────────────────────
if [[ "${DIRECT:-0}" == "1" ]]; then
    REMOTE_IP="${1:?}"
    REMOTE_PORT="${2:?Usage: DIRECT=1 pull_weights.sh <ip> <ssh-port>}"
    REMOTE_USER="${REMOTE_USER:-root}"
    echo "Direct rsync from $REMOTE_USER@$REMOTE_IP:$REMOTE_PORT"
    mkdir -p "$LOCAL_DIR"
    rsync -avz --progress \
        -e "ssh -p $REMOTE_PORT $SSH_OPTS" \
        "$REMOTE_USER@$REMOTE_IP:$REMOTE_EXPORT_DIR/" \
        "$LOCAL_DIR/"
    _print_summary
    exit 0
fi

# ── HTTP server mode (RunPod SSH proxy — no SCP support) ─────────────────────
echo "Starting HTTP server on pod at port $HTTP_PORT..."
echo "  (Make sure port $HTTP_PORT is exposed in your RunPod pod settings)"
echo ""

# Derive the HTTP base URL from the SSH host
# RunPod proxy format: <hash>-<podid>@ssh.runpod.io  →  https://<podid>-<port>.proxy.runpod.net
if [[ "$SSH_HOST" == *"@ssh.runpod.io"* ]]; then
    USER_PART="${SSH_HOST%%@*}"          # e.g. 9vvgwfgp10oosn-644121cc
    POD_ID="${USER_PART##*-}"            # last segment after dash: 644121cc
    HTTP_BASE="https://${POD_ID}-${HTTP_PORT}.proxy.runpod.net"
else
    # Fallback: assume direct IP
    HTTP_BASE="http://${SSH_HOST%%@*}:${HTTP_PORT}"
fi

echo "Weight files will be downloaded from: $HTTP_BASE"
echo ""

# Start HTTP server on the pod (background, serves weights_export/)
ssh $SSH_OPTS "$SSH_HOST" \
    "cd $REMOTE_EXPORT_DIR && nohup python3 -m http.server $HTTP_PORT --bind 0.0.0.0 > /tmp/http_weights.log 2>&1 & echo \$! > /tmp/http_weights.pid && sleep 2 && echo 'HTTP server started'" \
    2>&1 | grep -v "^Warning"

echo ""
mkdir -p "$LOCAL_DIR"

# List of expected weight files
WEIGHT_FILES=(
    "player_detector_yolo.pt"
    "player_detector_rtdetr.pth"
    "pitch_keypoints_yolo.pt"
    "pitch_keypoints_vitpose.pth"
    "reid_student.pth"
)

DOWNLOADED=0
for fname in "${WEIGHT_FILES[@]}"; do
    url="$HTTP_BASE/$fname"
    echo -n "Checking $fname ... "
    # HEAD request to check existence
    if curl -sf --head "$url" -o /dev/null 2>/dev/null; then
        echo "downloading..."
        curl -f --progress-bar "$url" -o "$LOCAL_DIR/$fname"
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo "not found (skipping)"
    fi
done

# Kill HTTP server on pod
echo ""
echo "Stopping HTTP server on pod..."
ssh $SSH_OPTS "$SSH_HOST" \
    "kill \$(cat /tmp/http_weights.pid 2>/dev/null) 2>/dev/null || true" \
    2>&1 | grep -v "^Warning" || true

if [ "$DOWNLOADED" -eq 0 ]; then
    echo ""
    echo "ERROR: No weight files found at $HTTP_BASE"
    echo "  - Confirm port $HTTP_PORT is exposed in RunPod pod settings"
    echo "  - Confirm train_remote.sh completed successfully (weights_export/ exists)"
    exit 1
fi

echo ""
echo "Downloaded $DOWNLOADED weight file(s) to $LOCAL_DIR/"
ls -lh "$LOCAL_DIR/"

# ── Print suggested analyze commands ─────────────────────────────────────────
echo ""
echo "Suggested inference commands:"
echo "──────────────────────────────────────────────────────"

PITCH=""
REID=""

if [ -f "$LOCAL_DIR/pitch_keypoints_yolo.pt" ];     then PITCH="$LOCAL_DIR/pitch_keypoints_yolo.pt"; fi
if [ -f "$LOCAL_DIR/pitch_keypoints_vitpose.pth" ];  then PITCH="$LOCAL_DIR/pitch_keypoints_vitpose.pth"; fi
if [ -f "$LOCAL_DIR/reid_student.pth" ];             then REID="$LOCAL_DIR/reid_student.pth"; fi

MODEL_FLAG=""
if [ -f "$LOCAL_DIR/player_detector_yolo.pt" ]; then
    MODEL_FLAG="--model $LOCAL_DIR/player_detector_yolo.pt --model-type yolo"
elif [ -f "$LOCAL_DIR/player_detector_rtdetr.pth" ]; then
    MODEL_FLAG="--model $LOCAL_DIR/player_detector_rtdetr.pth --model-type rtdetr"
fi

PITCH_FLAG=""
if [ -n "$PITCH" ]; then PITCH_FLAG="--pitch-weights $PITCH"; fi

REID_FLAG=""
if [ -n "$REID" ]; then REID_FLAG="--reid-weights $REID"; fi

echo "torchkick analyze -v match.mp4 \\"
[ -n "$MODEL_FLAG" ] && echo "    $MODEL_FLAG \\"
[ -n "$PITCH_FLAG"  ] && echo "    $PITCH_FLAG \\"
[ -n "$REID_FLAG"   ] && echo "    $REID_FLAG"
echo "──────────────────────────────────────────────────────"
