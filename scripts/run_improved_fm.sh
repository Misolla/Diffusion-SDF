#!/usr/bin/env bash
#
# Improved training: fair comparison at 5000 S3 epochs.
#
#   DDPM + ConvPointnet  (baseline)
#   FM   + PointMAE
#
# Reuses:
#   - Stage 1 checkpoint from pipeclean (epoch 10500)
#   - Stage 2 FM+PointMAE checkpoint from extended run (5000 epochs)
#
# Epoch counts:
#   Stage 3 FM+PointMAE:              1500→5000    resumed           (~10 hrs)
#
# Estimated total runtime: ~10 hours
#
# Usage:
#   conda activate diffusionsdf
#   bash scripts/run_improved_fm.sh 2>&1 | tee improved_fm.log
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
for f in config/pipeclean_s1/last.ckpt config/extended_s2_fm/last.ckpt config/extended_s3_fm/last.ckpt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found."
        exit 1
    fi
done
echo "FM+PointMAE S3 (resume from): config/extended_s3_fm/last.ckpt (epoch 1500)"

MOD_COUNT=$(find config/pipeclean_s1/modulations -name "latent.txt" 2>/dev/null | wc -l)
if [ "$MOD_COUNT" -eq 0 ]; then
    echo "ERROR: No modulations in config/pipeclean_s1/modulations/. Run pipeclean first."
    exit 1
fi
echo "Modulations:        $MOD_COUNT files"


# ---------------------------------------------------------------
# Phase 1: Stage 3 FM+PointMAE — resume from extended checkpoint (1500 → 5000 epochs)
# ---------------------------------------------------------------
banner "PHASE 1/1: Stage 3 FM + PointMAE (resume 1500 → 5000 epochs)"

if [ ! -f config/improved_s3_fm/last.ckpt ]; then
    echo "Copying extended S3 FM+PointMAE checkpoint to resume from epoch 1500 ..."
    cp config/extended_s3_fm/last.ckpt config/improved_s3_fm/last.ckpt
fi

PHASE_START=$(date +%s)

python train.py \
    -e config/improved_s3_fm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r last

banner "Phase 1 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
banner "IMPROVED TRAINING COMPLETE in $(elapsed_since $GLOBAL_START)"

echo "Checkpoints:"
for d in improved_s3_fm; do
    if [ -f "config/$d/last.ckpt" ]; then
        SIZE=$(ls -lh "config/$d/last.ckpt" | awk '{print $5}')
        echo "  OK   config/$d/last.ckpt  ($SIZE)"
    else
        echo "  FAIL config/$d/last.ckpt  (missing)"
    fi
done

echo ""
echo "To render comparison (FM+PointMAE vs DDPM):"
echo "  python scripts/render_comparison.py \\"
echo "      --ddpm-ckpt config/improved_s3_ddpm/last.ckpt \\"
echo "      --fm-ckpt config/improved_s3_fm/last.ckpt \\"
echo "      --gt-dir gt_meshes \\"
echo "      --num-shapes 5 --samples-per-shape 3 \\"
echo "      --output renders_improved_fm"
