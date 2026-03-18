#!/usr/bin/env bash
#
# Improved training: 10000 S3 epochs.
#
#   FM   +   ConvPointnet
#
# Reuses:
#   - Stage 1 checkpoint from pipeclean (epoch 10500)
#   - Stage 2 FM+ConvPointnet checkpoint from extended run (10000 epochs)
#
# Epoch counts:
#   Stage 3 FM+ConvPointnet:              5000->10000    resumed      
#
#
# Usage:
#   conda activate diffusionsdf
#   bash scripts/run_improved_fm_conv.sh 2>&1 | tee improved_fm_conv_10000.log
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
for f in config/pipeclean_s1/last.ckpt config/improved_s2_fm_conv/last.ckpt config/improved_s3_fm_conv/last.ckpt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found."
        exit 1
    fi
done
echo "FM+ConvPointNet S3 (resume from): config/improved_s3_fm_conv/last.ckpt (epoch 5000)"

MOD_COUNT=$(find config/pipeclean_s1/modulations -name "latent.txt" 2>/dev/null | wc -l)
if [ "$MOD_COUNT" -eq 0 ]; then
    echo "ERROR: No modulations in config/pipeclean_s1/modulations/. Run pipeclean first."
    exit 1
fi
echo "Modulations:        $MOD_COUNT files"


# ---------------------------------------------------------------
# Phase 1: Stage 3 FM+ConvPointnet — resume from extended checkpoint (5000 → 10000 epochs)
# ---------------------------------------------------------------
banner "PHASE 1/1: Stage 3 FM + ConvPointnet (resume 5000 → 10000 epochs)"

if [ ! -f config/improved_s3_fm_conv/last.ckpt ]; then
    echo "Copying extended S3 FM+ConvPointnet checkpoint to resume from epoch 5000 ..."
    cp config/improved_s3_fm_conv/last.ckpt config/improved_s3_fm_conv/last.ckpt
fi

PHASE_START=$(date +%s)

python train.py \
    -e config/improved_s3_fm_conv \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r last

banner "Phase 1 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
banner "IMPROVED TRAINING COMPLETE in $(elapsed_since $GLOBAL_START)"

echo "Checkpoints:"
for d in improved_s3_fm_conv; do
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
echo "      --output renders_improved_fm_conv_10000"
