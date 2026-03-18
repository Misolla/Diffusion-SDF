#!/usr/bin/env bash
#
# Full training: 3-way comparison using overnight S1 (20K epoch) checkpoint.
#
#   DDPM + ConvPointnet  (baseline)
#   FM   + ConvPointnet  (isolate FM contribution, LR=5e-5, warmup=500)
#   FM   + Point-MAE     (full proposed method, LR=5e-5, warmup=500)
#
# Uses: config/overnight_s1/last.ckpt (20K epoch SDF-VAE)
#
# Epoch counts & estimated times (based on extended run per-epoch rates):
#   Modulation extraction:                         ~1 min
#   Stage 2 DDPM          5000 epochs  ~1.75s/ep   ~2.4 hrs
#   Stage 2 FM+Conv       5000 epochs  ~1.9s/ep    ~2.6 hrs
#   Stage 2 FM+MAE        5000 epochs  ~1.9s/ep    ~2.6 hrs
#   Stage 3 DDPM         10000 epochs  ~11s/ep     ~30.6 hrs
#   Stage 3 FM+Conv      10000 epochs  ~11s/ep     ~30.6 hrs
#   Stage 3 FM+MAE       10000 epochs  ~11s/ep     ~30.6 hrs
#                                          Total:  ~100 hrs (~4.2 days)
#
# Usage:
#   conda activate diffusionsdf
#   bash scripts/run_full.sh 2>&1 | tee full.log
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
if [ ! -f config/overnight_s1/last.ckpt ]; then
    echo "ERROR: config/overnight_s1/last.ckpt not found."
    exit 1
fi
echo "S1 checkpoint (20K): config/overnight_s1/last.ckpt"
ls -lh config/overnight_s1/last.ckpt

# ---------------------------------------------------------------
# Phase 1: Extract modulations from overnight S1
# ---------------------------------------------------------------
MOD_COUNT=$(find config/overnight_s1/modulations -name "latent.txt" 2>/dev/null | wc -l)
if [ "$MOD_COUNT" -gt 0 ]; then
    banner "PHASE 1/7: Skipping modulation extraction ($MOD_COUNT files already exist)"
else
    banner "PHASE 1/7: Extracting modulations from overnight S1"
    PHASE_START=$(date +%s)

    python scripts/extract_modulations.py \
        --checkpoint config/overnight_s1/last.ckpt \
        --split data/splits/couch_mini.json \
        --output config/overnight_s1/modulations \
        --data_source data

    banner "Phase 1 complete in $(elapsed_since $PHASE_START)"
    MOD_COUNT=$(find config/overnight_s1/modulations -name "latent.txt" | wc -l)
fi
echo "Modulation files: $MOD_COUNT"
if [ "$MOD_COUNT" -eq 0 ]; then
    echo "ERROR: No modulations extracted. Aborting."
    exit 1
fi

# ---------------------------------------------------------------
# Phase 2: Stage 2 DDPM (5000 epochs)
# ---------------------------------------------------------------
banner "PHASE 2/7: Stage 2 DDPM + ConvPointnet (5000 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/full_s2_ddpm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 2 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 3: Stage 2 FM + ConvPointnet (5000 epochs, LR=5e-5, warmup=500)
# ---------------------------------------------------------------
banner "PHASE 3/7: Stage 2 FM + ConvPointnet (5000 epochs, LR=5e-5)"
PHASE_START=$(date +%s)

python train.py \
    -e config/full_s2_fm_conv \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 3 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 4: Stage 2 FM + Point-MAE (5000 epochs, LR=5e-5, warmup=500)
# ---------------------------------------------------------------
banner "PHASE 4/7: Stage 2 FM + Point-MAE (5000 epochs, LR=5e-5)"
PHASE_START=$(date +%s)

python train.py \
    -e config/full_s2_fm_mae \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 4 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 5: Stage 3 DDPM (10000 epochs)
# ---------------------------------------------------------------
banner "PHASE 5/7: Stage 3 DDPM + ConvPointnet (10000 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/full_s3_ddpm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 5 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 6: Stage 3 FM + ConvPointnet (10000 epochs, LR=5e-5, warmup=500)
# ---------------------------------------------------------------
banner "PHASE 6/7: Stage 3 FM + ConvPointnet (10000 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/full_s3_fm_conv \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 6 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 7: Stage 3 FM + Point-MAE (10000 epochs, LR=5e-5, warmup=500)
# ---------------------------------------------------------------
banner "PHASE 7/7: Stage 3 FM + Point-MAE (10000 epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/full_s3_fm_mae \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 7 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
banner "FULL TRAINING COMPLETE in $(elapsed_since $GLOBAL_START)"

echo "Checkpoints:"
for d in full_s2_ddpm full_s2_fm_conv full_s2_fm_mae full_s3_ddpm full_s3_fm_conv full_s3_fm_mae; do
    if [ -f "config/$d/last.ckpt" ]; then
        SIZE=$(ls -lh "config/$d/last.ckpt" | awk '{print $5}')
        echo "  OK   config/$d/last.ckpt  ($SIZE)"
    else
        echo "  FAIL config/$d/last.ckpt  (missing)"
    fi
done

echo ""
echo "To render comparisons:"
echo "  # DDPM vs FM+ConvPointnet:"
echo "  python scripts/render_comparison.py \\"
echo "      --ddpm-ckpt config/full_s3_ddpm/last.ckpt \\"
echo "      --fm-ckpt config/full_s3_fm_conv/last.ckpt \\"
echo "      --gt-dir gt_meshes \\"
echo "      --num-shapes 5 --samples-per-shape 3 \\"
echo "      --output renders_full_conv"
echo ""
echo "  # DDPM vs FM+Point-MAE:"
echo "  python scripts/render_comparison.py \\"
echo "      --ddpm-ckpt config/full_s3_ddpm/last.ckpt \\"
echo "      --fm-ckpt config/full_s3_fm_mae/last.ckpt \\"
echo "      --gt-dir gt_meshes \\"
echo "      --num-shapes 5 --samples-per-shape 3 \\"
echo "      --output renders_full_mae"
