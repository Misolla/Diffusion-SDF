#!/usr/bin/env bash
#
# Continue training: 20000 S1 epochs.
#
# Reuses:
#   - Stage 1 checkpoint from overnight run (epoch 10500)
#
# Epoch counts:
#   Stage 1  SDF-VAE (shared):              10000->20000    resumed      
#
#
# Usage:
#   conda activate diffusionsdf
#   bash scripts/run_continue_stage1.sh 2>&1 | tee continue_stage1_20000.log
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

BATCH_SIZE=32
WORKERS=8

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

banner() {
    echo ""
    echo "============================================================"
    echo "  $(timestamp)  $1"
    echo "============================================================"
    echo ""
}

elapsed_since() {
    local start=$1
    local now
    now=$(date +%s)
    local diff=$((now - start))
    printf "%dh %dm %ds" $((diff/3600)) $(((diff%3600)/60)) $((diff%60))
}

GLOBAL_START=$(date +%s)

# ---------------------------------------------------------------
# Verify prerequisites
# ---------------------------------------------------------------
for f in config/overnight_s1/last.ckpt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found."
        exit 1
    fi
done
echo "S1 (resume from): config/overnight_s1/last.ckpt (epoch 10500)"


# ---------------------------------------------------------------
# Phase 1: Stage 1 SDF-VAE (shared) — resume from pipeclean checkpoint (10500 → 20000 epochs)
# ---------------------------------------------------------------
banner "PHASE 1/1: Stage 1 SDF-VAE (shared) (resume 10500 → 20000 epochs)"

if [ ! f config/overnight_s1/last.ckpt ]; then
    # echo "Copying overnight S1 checkpoint to resume from epoch 10500 ..."
    # cp config/overnight_s1/last.ckpt config/overnight_s1/last.ckpt
    echo "ERROR: Overnight S1 checkpoint not found. Run overnight first."
    exit 1
fi

PHASE_START=$(date +%s)

python train.py \
    -e config/overnight_s1 \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r last

banner "Phase 1 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
banner "CONTINUE TRAINING COMPLETE in $(elapsed_since $GLOBAL_START)"

echo "Checkpoints:"
for d in overnight_s1; do
    if [ -f "config/$d/last.ckpt" ]; then
        SIZE=$(ls -lh "config/$d/last.ckpt" | awk '{print $5}')
        echo "  OK   config/$d/last.ckpt  ($SIZE)"
    else
        echo "  FAIL config/$d/last.ckpt  (missing)"
    fi
done

echo ""
echo "TensorBoard logs at:"
echo "  tensorboard_logs/config/overnight_s*/"
echo ""
