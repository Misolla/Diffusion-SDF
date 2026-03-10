#!/usr/bin/env bash
#
# Pipeclean: fast end-to-end validation of the full training pipeline.
#
# Uses the Stage 1 checkpoint from the overnight run (~epoch 10500),
# runs 1% of the overnight epoch counts through Stages 2 and 3
# for both DDPM and Flow Matching.
#
# Expected total runtime: ~15-20 minutes
#
# Usage:
#   conda activate diffusionsdf
#   bash scripts/run_pipeclean.sh 2>&1 | tee pipeclean.log
#
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
    printf "%dm %ds" $(((diff%3600)/60)) $((diff%60))
}

GLOBAL_START=$(date +%s)

# ---------------------------------------------------------------
# Verify S1 checkpoint exists
# ---------------------------------------------------------------
if [ ! -f config/pipeclean_s1/last.ckpt ]; then
    echo "ERROR: config/pipeclean_s1/last.ckpt not found."
    echo "Copy it from the overnight run:"
    echo "  cp config/overnight_s1/last.ckpt config/pipeclean_s1/last.ckpt"
    exit 1
fi
echo "Using S1 checkpoint: config/pipeclean_s1/last.ckpt"
ls -lh config/pipeclean_s1/last.ckpt

# ---------------------------------------------------------------
# Phase 1: Extract modulations from S1 checkpoint
# ---------------------------------------------------------------
banner "PHASE 1/5: Extracting modulations"
PHASE_START=$(date +%s)

python scripts/extract_modulations.py \
    --checkpoint config/pipeclean_s1/last.ckpt \
    --split data/splits/couch_mini.json \
    --output config/pipeclean_s1/modulations \
    --data_source data

banner "Phase 1 complete in $(elapsed_since $PHASE_START)"

MOD_COUNT=$(find config/pipeclean_s1/modulations -name "latent.txt" | wc -l)
echo "Extracted $MOD_COUNT modulation files"
if [ "$MOD_COUNT" -eq 0 ]; then
    echo "ERROR: No modulations extracted. Aborting."
    exit 1
fi

# ---------------------------------------------------------------
# Phase 2: Stage 2 DDPM (200 epochs)
# ---------------------------------------------------------------
banner "PHASE 2/5: Stage 2 DDPM (200 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/pipeclean_s2_ddpm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 2 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 3: Stage 2 Flow Matching (200 epochs)
# ---------------------------------------------------------------
banner "PHASE 3/5: Stage 2 Flow Matching (200 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/pipeclean_s2_fm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 3 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 4: Stage 3 DDPM end-to-end (50 epochs)
# ---------------------------------------------------------------
banner "PHASE 4/5: Stage 3 DDPM end-to-end (50 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/pipeclean_s3_ddpm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 4 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 5: Stage 3 Flow Matching end-to-end (50 epochs)
# ---------------------------------------------------------------
banner "PHASE 5/5: Stage 3 FM end-to-end (50 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/pipeclean_s3_fm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 5 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
banner "PIPECLEAN COMPLETE in $(elapsed_since $GLOBAL_START)"

echo "Checkpoints:"
for d in pipeclean_s2_ddpm pipeclean_s2_fm pipeclean_s3_ddpm pipeclean_s3_fm; do
    if [ -f "config/$d/last.ckpt" ]; then
        echo "  OK   config/$d/last.ckpt"
    else
        echo "  FAIL config/$d/last.ckpt (missing)"
    fi
done
