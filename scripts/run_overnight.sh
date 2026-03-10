#!/usr/bin/env bash
#
# Overnight training script: DDPM vs Flow Matching comparison
#
# Runs all stages sequentially:
#   1. Stage 1  SDF-VAE (shared)              ~4.5 hours
#   2. Extract modulations                     ~5 min
#   3. Stage 2  DDPM diffusion                 ~30 min
#   4. Stage 2  Flow Matching + Point-MAE      ~30 min
#   5. Stage 3  DDPM end-to-end                ~1.5 hours
#   6. Stage 3  FM end-to-end                  ~1.5 hours
#                                       Total: ~8.5 hours
#
# Usage:
#   screen -S training
#   conda activate diffusionsdf
#   bash scripts/run_overnight.sh 2>&1 | tee overnight.log
#   # Ctrl-A D to detach
#
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

BATCH_SIZE=32
WORKERS=8
LOGFILE="overnight.log"

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
# Phase 1: Stage 1 SDF-VAE (shared by both methods)
# ---------------------------------------------------------------
banner "PHASE 1/6: Stage 1 SDF-VAE (20K epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/overnight_s1 \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 1 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 2: Extract modulations from Stage 1
# ---------------------------------------------------------------
banner "PHASE 2/6: Extracting modulations"
PHASE_START=$(date +%s)

python scripts/extract_modulations.py \
    --checkpoint config/overnight_s1/last.ckpt \
    --split data/splits/couch_mini.json \
    --output config/overnight_s1/modulations \
    --data_source data

banner "Phase 2 complete in $(elapsed_since $PHASE_START)"

# Verify modulations were extracted
MOD_COUNT=$(find config/overnight_s1/modulations -name "latent.txt" | wc -l)
echo "Extracted $MOD_COUNT modulation files"
if [ "$MOD_COUNT" -eq 0 ]; then
    echo "ERROR: No modulations extracted. Aborting."
    exit 1
fi

# ---------------------------------------------------------------
# Phase 3: Stage 2 DDPM (diffusion on latents)
# ---------------------------------------------------------------
banner "PHASE 3/6: Stage 2 DDPM (20K epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/overnight_s2_ddpm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 3 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 4: Stage 2 Flow Matching (diffusion on latents)
# ---------------------------------------------------------------
banner "PHASE 4/6: Stage 2 Flow Matching (20K epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/overnight_s2_fm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS"

banner "Phase 4 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 5: Stage 3 DDPM end-to-end
# ---------------------------------------------------------------
banner "PHASE 5/6: Stage 3 DDPM end-to-end (5K epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/overnight_s3_ddpm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 5 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Phase 6: Stage 3 Flow Matching end-to-end
# ---------------------------------------------------------------
banner "PHASE 6/6: Stage 3 Flow Matching end-to-end (5K epochs)"
PHASE_START=$(date +%s)

python train.py \
    -e config/overnight_s3_fm \
    -b "$BATCH_SIZE" \
    -w "$WORKERS" \
    -r finetune

banner "Phase 6 complete in $(elapsed_since $PHASE_START)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
banner "ALL PHASES COMPLETE in $(elapsed_since $GLOBAL_START)"

echo "Checkpoints saved at:"
echo "  Stage 1 SDF-VAE:     config/overnight_s1/last.ckpt"
echo "  Stage 2 DDPM:        config/overnight_s2_ddpm/last.ckpt"
echo "  Stage 2 FM:          config/overnight_s2_fm/last.ckpt"
echo "  Stage 3 DDPM:        config/overnight_s3_ddpm/last.ckpt"
echo "  Stage 3 FM:          config/overnight_s3_fm/last.ckpt"
echo ""
echo "TensorBoard logs at:"
echo "  tensorboard_logs/config/overnight_s*/"
echo ""
echo "To compare results:"
echo "  tensorboard --logdir tensorboard_logs/"
