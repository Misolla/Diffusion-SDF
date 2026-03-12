#!/usr/bin/env bash
#
# Improved training: fair comparison at 5000 S3 epochs.
#
#   DDPM + ConvPointnet  (baseline)
#   FM   + ConvPointnet  (isolate FM contribution, higher LR + warmup)
#
# Reuses:
#   - Stage 1 checkpoint from pipeclean (epoch 10500)
#   - Stage 2 DDPM checkpoint from extended run (5000 epochs)
#
# Epoch counts:
#   Stage 2 (diffusion-only):  5000 epochs  x1 FM variant    (~2.5 hrs)
#   Stage 3 DDPM:              1500→5000    resumed           (~10 hrs)
#   Stage 3 FM+Conv:           5000 epochs  from scratch      (~15 hrs)
#
# Estimated total runtime: ~28 hours
#
# Usage:
#   conda activate diffusionsdf
#   bash scripts/run_improved.sh 2>&1 | tee improved.log
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
# Verify prerequisites
# ---------------------------------------------------------------
for f in config/pipeclean_s1/last.ckpt config/extended_s2_ddpm/last.ckpt config/extended_s3_ddpm/last.ckpt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found."
        exit 1
    fi
done
echo "S1 checkpoint:        config/pipeclean_s1/last.ckpt"
echo "DDPM S2 checkpoint:   config/extended_s2_ddpm/last.ckpt"
echo "DDPM S3 (resume from): config/extended_s3_ddpm/last.ckpt (epoch 1500)"

MOD_COUNT=$(find config/pipeclean_s1/modulations -name "latent.txt" 2>/dev/null | wc -l)
if [ "$MOD_COUNT" -eq 0 ]; then
    echo "ERROR: No modulations in config/pipeclean_s1/modulations/. Run pipeclean first."
    exit 1
fi
echo "Modulations:        $MOD_COUNT files"

# ---------------------------------------------------------------
# Phase 1: Stage 2 FM + ConvPointnet (5000 epochs, LR=5e-5, warmup=500)
# ---------------------------------------------------------------
banner "PHASE 1/3: Stage 2 FM + ConvPointnet (5000 epochs, LR=5e-5)"
PHASE_START=$(date +%s)

python train.py \
    -e config/improved_s2_fm_conv \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 1 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 2: Stage 3 FM + ConvPointnet (5000 epochs, LR=5e-5, warmup=500)
# ---------------------------------------------------------------
banner "PHASE 2/3: Stage 3 FM + ConvPointnet (5000 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/improved_s3_fm_conv \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 2 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 3: Stage 3 DDPM — resume from extended checkpoint (1500 → 5000 epochs)
# ---------------------------------------------------------------
banner "PHASE 3/3: Stage 3 DDPM + ConvPointnet (resume 1500 → 5000 epochs)"

if [ ! -f config/improved_s3_ddpm/last.ckpt ]; then
    echo "Copying extended S3 DDPM checkpoint to resume from epoch 1500 ..."
    cp config/extended_s3_ddpm/last.ckpt config/improved_s3_ddpm/last.ckpt
fi

PHASE_START=$(date +%s)

python train.py \
    -e config/improved_s3_ddpm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r last

banner "Phase 3 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
banner "IMPROVED TRAINING COMPLETE in $(elapsed_since $GLOBAL_START)"

echo "Checkpoints:"
for d in improved_s3_ddpm improved_s2_fm_conv improved_s3_fm_conv; do
    if [ -f "config/$d/last.ckpt" ]; then
        SIZE=$(ls -lh "config/$d/last.ckpt" | awk '{print $5}')
        echo "  OK   config/$d/last.ckpt  ($SIZE)"
    else
        echo "  FAIL config/$d/last.ckpt  (missing)"
    fi
done

echo ""
echo "To render comparison (FM+ConvPointnet vs DDPM):"
echo "  python scripts/render_comparison.py \\"
echo "      --ddpm-ckpt config/improved_s3_ddpm/last.ckpt \\"
echo "      --fm-ckpt config/improved_s3_fm_conv/last.ckpt \\"
echo "      --gt-dir gt_meshes \\"
echo "      --num-shapes 5 --samples-per-shape 3 \\"
echo "      --output renders_improved"
