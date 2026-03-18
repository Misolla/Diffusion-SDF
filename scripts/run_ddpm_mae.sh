#!/usr/bin/env bash
#
# Ablation: DDPM + Point-MAE encoder
#
# Isolates the effect of Point-MAE by pairing it with the original DDPM
# sampler. Compare results with improved_s3_ddpm (DDPM + ConvPointnet, 5000 S3).
#
# Uses pipeclean S1 checkpoint (epoch 10500).
#
# Epoch counts:
#   Stage 2:  5000 epochs  (~2.5 hrs)
#   Stage 3:  5000 epochs  (~15 hrs)
#
# Estimated total runtime: ~18 hours
#
# Usage:
#   conda activate diffusionsdf
#   bash scripts/run_ddpm_mae.sh 2>&1 | tee ddpm_mae.log
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
if [ ! -f config/pipeclean_s1/last.ckpt ]; then
    echo "ERROR: config/pipeclean_s1/last.ckpt not found."
    exit 1
fi
echo "S1 checkpoint: config/pipeclean_s1/last.ckpt"

MOD_COUNT=$(find config/pipeclean_s1/modulations -name "latent.txt" 2>/dev/null | wc -l)
if [ "$MOD_COUNT" -eq 0 ]; then
    echo "ERROR: No modulations in config/pipeclean_s1/modulations/."
    exit 1
fi
echo "Modulations:   $MOD_COUNT files"

# ---------------------------------------------------------------
# Phase 1: Stage 2 DDPM + Point-MAE (5000 epochs)
# ---------------------------------------------------------------
banner "PHASE 1/2: Stage 2 DDPM + Point-MAE (5000 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/ddpm_mae_s2 \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 1 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 2: Stage 3 DDPM + Point-MAE (5000 epochs)
# ---------------------------------------------------------------
banner "PHASE 2/2: Stage 3 DDPM + Point-MAE (5000 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/ddpm_mae_s3 \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 2 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
banner "DDPM+MAE ABLATION COMPLETE in $(elapsed_since $GLOBAL_START)"

echo "Checkpoints:"
for d in ddpm_mae_s2 ddpm_mae_s3; do
    if [ -f "config/$d/last.ckpt" ]; then
        SIZE=$(ls -lh "config/$d/last.ckpt" | awk '{print $5}')
        echo "  OK   config/$d/last.ckpt  ($SIZE)"
    else
        echo "  FAIL config/$d/last.ckpt  (missing)"
    fi
done

echo ""
echo "To render comparison (DDPM+MAE vs DDPM+ConvPointnet):"
echo "  python scripts/render_comparison.py \\"
echo "      --ddpm-ckpt config/improved_s3_ddpm/last.ckpt \\"
echo "      --fm-ckpt config/ddpm_mae_s3/last.ckpt \\"
echo "      --gt-dir gt_meshes \\"
echo "      --num-shapes 5 --samples-per-shape 3 \\"
echo "      --output renders_ddpm_mae"
