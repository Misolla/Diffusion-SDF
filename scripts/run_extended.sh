#!/usr/bin/env bash
#
# Extended training: 25-30x pipeclean epochs for meaningful quality comparison.
#
# Reuses the Stage 1 checkpoint from the pipeclean run (epoch 10500).
# Modulations are re-extracted only if not already present.
#
# Epoch counts:
#   Stage 2 (diffusion-only):  5000 epochs  (~2.5 hrs DDPM, ~3 hrs FM)
#   Stage 3 (end-to-end):      1500 epochs  (~5 hrs DDPM, ~5 hrs FM)
#
# Estimated total runtime: ~16 hours
#
# Usage:
#   conda activate diffusionsdf
#   bash scripts/run_extended.sh 2>&1 | tee extended.log
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
    printf "%dh %dm %ds" $((diff/3600)) $(((diff%3600)/60)) $((diff%60))
}

GLOBAL_START=$(date +%s)

# ---------------------------------------------------------------
# Verify S1 checkpoint exists
# ---------------------------------------------------------------
if [ ! -f config/pipeclean_s1/last.ckpt ]; then
    echo "ERROR: config/pipeclean_s1/last.ckpt not found."
    echo "This should have been created by the pipeclean run."
    exit 1
fi
echo "Using S1 checkpoint: config/pipeclean_s1/last.ckpt"
ls -lh config/pipeclean_s1/last.ckpt

# ---------------------------------------------------------------
# Phase 1: Extract modulations (skip if already done)
# ---------------------------------------------------------------
MOD_COUNT=$(find config/pipeclean_s1/modulations -name "latent.txt" 2>/dev/null | wc -l)
if [ "$MOD_COUNT" -gt 0 ]; then
    banner "PHASE 1/5: Skipping modulation extraction ($MOD_COUNT files already exist)"
else
    banner "PHASE 1/5: Extracting modulations"
    PHASE_START=$(date +%s)

    python scripts/extract_modulations.py \
        --checkpoint config/pipeclean_s1/last.ckpt \
        --split data/splits/couch_mini.json \
        --output config/pipeclean_s1/modulations \
        --data_source data

    banner "Phase 1 complete in $(elapsed_since $PHASE_START)"
    MOD_COUNT=$(find config/pipeclean_s1/modulations -name "latent.txt" | wc -l)
fi
echo "Modulation files available: $MOD_COUNT"
if [ "$MOD_COUNT" -eq 0 ]; then
    echo "ERROR: No modulations found. Aborting."
    exit 1
fi

# ---------------------------------------------------------------
# Phase 2: Stage 2 DDPM (5000 epochs, ~2.5 hrs)
# ---------------------------------------------------------------
banner "PHASE 2/5: Stage 2 DDPM (5000 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/extended_s2_ddpm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 2 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 3: Stage 2 Flow Matching (5000 epochs, ~3 hrs)
# ---------------------------------------------------------------
banner "PHASE 3/5: Stage 2 Flow Matching (5000 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/extended_s2_fm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 3 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 4: Stage 3 DDPM end-to-end (1500 epochs, ~5 hrs)
# ---------------------------------------------------------------
banner "PHASE 4/5: Stage 3 DDPM end-to-end (1500 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/extended_s3_ddpm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 4 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 5: Stage 3 Flow Matching end-to-end (1500 epochs, ~5 hrs)
# ---------------------------------------------------------------
banner "PHASE 5/5: Stage 3 FM end-to-end (1500 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/extended_s3_fm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 5 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
banner "EXTENDED TRAINING COMPLETE in $(elapsed_since $GLOBAL_START)"

echo "Checkpoints:"
for d in extended_s2_ddpm extended_s2_fm extended_s3_ddpm extended_s3_fm; do
    if [ -f "config/$d/last.ckpt" ]; then
        SIZE=$(ls -lh "config/$d/last.ckpt" | awk '{print $5}')
        echo "  OK   config/$d/last.ckpt  ($SIZE)"
    else
        echo "  FAIL config/$d/last.ckpt  (missing)"
    fi
done

echo ""
echo "To render comparisons:"
echo "  python scripts/render_comparison.py \\"
echo "      --ddpm-ckpt config/extended_s3_ddpm/last.ckpt \\"
echo "      --fm-ckpt config/extended_s3_fm/last.ckpt \\"
echo "      --gt-dir gt_meshes \\"
echo "      --num-shapes 5 --samples-per-shape 3 \\"
echo "      --output renders_extended"
