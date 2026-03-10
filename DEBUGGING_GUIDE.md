# Debugging Guide for Overnight Training Script

This guide covers diagnosing and recovering from failures in `scripts/run_overnight.sh`.

## Checking Status

```bash
# Reattach to screen session
screen -r training

# Or check the log without reattaching
tail -100 /home/max/Documents/Diffusion-SDF/overnight.log

# Find the current phase
grep "PHASE" overnight.log

# Check if python is still running
ps aux | grep "python train.py"

# Check GPU utilization
nvidia-smi
```

## Identifying Which Phase Failed

The script prints banners like `PHASE 3/6: Stage 2 DDPM (20K epochs)`. The last
banner before the error tells you which phase failed. The script uses `set -e` so
it stops on the first failure.

| Phase | Config Dir | What Runs |
|-------|-----------|-----------|
| 1 | `config/overnight_s1` | SDF-VAE training (20K epochs) |
| 2 | -- | Modulation extraction |
| 3 | `config/overnight_s2_ddpm` | DDPM diffusion training |
| 4 | `config/overnight_s2_fm` | Flow Matching training |
| 5 | `config/overnight_s3_ddpm` | DDPM end-to-end fine-tuning |
| 6 | `config/overnight_s3_fm` | FM end-to-end fine-tuning |


## Common Failures and Fixes

### 1. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Fix:** Reduce batch size. Edit `scripts/run_overnight.sh` and change `BATCH_SIZE=32`
to `BATCH_SIZE=16` or `BATCH_SIZE=8`. The RTX A6000 has 48GB so batch 32 should be
fine for Stage 1/2, but Stage 3 (combined model) uses more memory.

To resume from the failed phase, see "Resuming After a Failure" below.


### 2. NaN Loss

**Symptoms:**
```
vae loss is nan at epoch X...
```
or loss values showing `nan` in the progress bar.

**Causes:**
- Learning rate too high (unlikely with defaults)
- Numerical instability in precision=16 (we use precision=32, so unlikely)
- Corrupted data file

**Fix:** NaN VAE loss is handled by the code (returns `None` to skip the batch). If
it happens persistently, check the data:
```bash
conda activate diffusionsdf
python -c "
import pandas as pd, numpy as np, sys
f = pd.read_csv('data/acronym/Couch/107637b6bdf8129d4904d89e9169817b/sdf_data.csv', header=None).values
print('Shape:', f.shape)
print('NaN count:', np.isnan(f).sum())
print('Inf count:', np.isinf(f).sum())
print('Surface pts (SDF=0):', (f[:,-1]==0).sum())
print('Range:', f.min(), f.max())
"
```


### 3. No Surface Points (SDF=0) Error

**Symptoms:**
```
RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0.
```
Traceback points to `conv_pointnet.py` or `base.py` at `pc = f[f[:,-1]==0][:,:3]`.

**Cause:** An SDF CSV file has no rows with exact SDF value of 0.

**Fix:** Re-run the surface point patching script:
```bash
conda activate diffusionsdf
python scripts/patch_surface_points.py
```
This adds 231,000 surface points (SDF=0) to each `sdf_data.csv`.


### 4. Modulation Extraction Fails (Phase 2)

**Symptoms:** Phase 2 prints `ERROR: No modulations extracted. Aborting.`

**Possible causes:**
- Stage 1 checkpoint doesn't exist (Phase 1 failed silently)
- Wrong checkpoint path

**Diagnosis:**
```bash
ls -la config/overnight_s1/last.ckpt
# Should exist and be several hundred MB
```

**Fix:** If the checkpoint exists but extraction fails, run extraction manually:
```bash
conda activate diffusionsdf
python scripts/extract_modulations.py \
    --checkpoint config/overnight_s1/last.ckpt \
    --split data/splits/couch_mini.json \
    --output config/overnight_s1/modulations

# Verify
find config/overnight_s1/modulations -name "latent.txt" | wc -l
# Should print 173
```


### 5. Stage 2 has 0 training samples

**Symptoms:**
```
AssertionError
```
or `data shape ... dataset len: 0` during Stage 2 data loading.

**Cause:** Modulations were extracted to the wrong directory, or the split file
references instances that don't have `latent.txt` files.

**Diagnosis:**
```bash
# Check where modulations landed
find config/overnight_s1/modulations -name "latent.txt" | head -5
# Expected: config/overnight_s1/modulations/Couch/{instance_id}/latent.txt

# Check that the config points to the right path
python -c "import json; s=json.load(open('config/overnight_s2_ddpm/specs.json')); print(s['data_path'])"
# Expected: config/overnight_s1/modulations
```


### 6. Stage 3 Finetune Checkpoint Loading Error

**Symptoms:**
```
FileNotFoundError: config/overnight_s2_ddpm/last.ckpt
```
or state_dict mismatch errors during finetune loading.

**Cause:** Stage 2 checkpoint doesn't exist (Stage 2 failed or checkpoints saved
to a different location).

**Diagnosis:**
```bash
ls -la config/overnight_s2_ddpm/last.ckpt
ls -la config/overnight_s2_fm/last.ckpt
```

**For state_dict mismatch:** This can happen if the FM checkpoint is loaded into a
DDPM model or vice versa. Verify the config's `diffusion_ckpt_path` points to the
correct Stage 2 variant:
```bash
python -c "
import json
for cfg in ['config/overnight_s3_ddpm/specs.json', 'config/overnight_s3_fm/specs.json']:
    s = json.load(open(cfg))
    print(cfg, '->', s['diffusion_ckpt_path'])
"
# Expected:
# overnight_s3_ddpm -> config/overnight_s2_ddpm/last.ckpt
# overnight_s3_fm   -> config/overnight_s2_fm/last.ckpt
```


### 7. torch_scatter CUDA Error

**Symptoms:**
```
RuntimeError: Not compiled with CUDA support
```
from `torch_scatter`.

**Fix:**
```bash
conda activate diffusionsdf
pip install --force-reinstall torch-scatter==2.0.9 \
    -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```


### 8. Process Killed / Segfault

**Symptoms:** Process disappears with no Python traceback. `dmesg` may show OOM killer.

**Diagnosis:**
```bash
dmesg | tail -20
# Look for "Out of memory: Killed process" or "oom-kill"
```

**Cause:** System RAM (not GPU) exhaustion. The SdfLoader loads ALL CSV files into
memory at startup. With 173 instances, each ~24MB (596K rows x 4 cols), that's ~4GB.
Stage 3 also loads grid files (~500K rows each), adding another ~3.5GB. Total memory
requirement is ~8-10GB RAM.

**Fix:** Reduce the number of workers (`WORKERS=4` or `WORKERS=2` in the script)
to reduce memory from data loader prefetching. Or close other memory-heavy processes.


## Resuming After a Failure

The script uses `set -e` and stops on the first error. To resume from a specific phase,
you can run individual phases manually.

### Resume from Phase 1 (start over)
```bash
conda activate diffusionsdf
cd /home/max/Documents/Diffusion-SDF
python train.py -e config/overnight_s1 -b 32 -w 8
```

If Stage 1 was partially trained and you want to continue from its last checkpoint:
```bash
python train.py -e config/overnight_s1 -b 32 -w 8 -r last
```

### Resume from Phase 2 (extraction only)
```bash
python scripts/extract_modulations.py \
    --checkpoint config/overnight_s1/last.ckpt \
    --split data/splits/couch_mini.json \
    --output config/overnight_s1/modulations
```

### Resume from Phase 3 (Stage 2 DDPM)
```bash
python train.py -e config/overnight_s2_ddpm -b 32 -w 8
```

### Resume from Phase 4 (Stage 2 FM)
```bash
python train.py -e config/overnight_s2_fm -b 32 -w 8
```

### Resume from Phase 5 (Stage 3 DDPM)
```bash
python train.py -e config/overnight_s3_ddpm -b 32 -w 8 -r finetune
```

### Resume from Phase 6 (Stage 3 FM)
```bash
python train.py -e config/overnight_s3_fm -b 32 -w 8 -r finetune
```

### Resuming a partially-trained stage

If a stage crashed mid-training, it should have a `last.ckpt` in its config directory.
Use `-r last` to resume:
```bash
python train.py -e config/overnight_s2_ddpm -b 32 -w 8 -r last
```

**Exception:** Stage 3 initially requires `-r finetune` to load Stage 1+2 checkpoints.
Once Stage 3 has started and produced its own `last.ckpt`, subsequent resumes should
use `-r last` instead of `-r finetune`.


## Verifying Training Quality

### Quick sanity checks after training completes

```bash
conda activate diffusionsdf
cd /home/max/Documents/Diffusion-SDF

# 1. Check all checkpoints exist
for d in overnight_s1 overnight_s2_ddpm overnight_s2_fm overnight_s3_ddpm overnight_s3_fm; do
    if [ -f "config/$d/last.ckpt" ]; then
        echo "OK: config/$d/last.ckpt ($(du -h config/$d/last.ckpt | cut -f1))"
    else
        echo "MISSING: config/$d/last.ckpt"
    fi
done

# 2. Check modulations
echo "Modulations: $(find config/overnight_s1/modulations -name latent.txt | wc -l) files"

# 3. Check TensorBoard logs exist
ls -la tensorboard_logs/config/overnight_*/

# 4. View TensorBoard
tensorboard --logdir tensorboard_logs/ --port 6006
```

### What good loss curves look like

- **Stage 1 (SDF):** `sdf` loss should decrease steadily from ~0.05 to ~0.01 over 20K epochs.
  `vae` loss should be small (~1e-3 to 1e-5).
- **Stage 2 (Diffusion):** `total` loss should decrease from ~1.0 to ~0.1 or lower.
  `diff100` loss (timesteps < 100) should approach 0.
- **Stage 3 (Combined):** `sdf`, `diff`, and `gensdf` should all decrease. `total` is
  their sum plus `vae` loss.

### Red flags

- Loss not decreasing at all after 1000+ epochs -> learning rate issue or data problem
- Loss exploding (increasing rapidly) -> numerical instability, reduce learning rate
- `diff100` staying high -> diffusion model not learning well, may need more epochs
- `gensdf` not decreasing in Stage 3 -> generated latents are poor quality


## Running Inference After Training

To generate SDF shapes from trained models (after all stages complete):

```python
# Example: generate from a point cloud using the DDPM model
import torch, json
from models.combined_model import CombinedModel

specs = json.load(open("config/overnight_s3_ddpm/specs.json"))
model = CombinedModel.load_from_checkpoint(
    "config/overnight_s3_ddpm/last.ckpt", specs=specs
).cuda().eval()

# Load a test point cloud
pc = ...  # (1, 1024, 3) tensor

with torch.no_grad():
    generated_latent = model.diffusion_model.generate_from_pc(pc.cuda())
    plane_features = model.vae_model.decode(generated_latent)
    # Use plane_features with the SDF model to reconstruct a mesh
```

See `utils/reconstruct.py` for mesh reconstruction utilities.


## File Locations Quick Reference

```
overnight.log                              -- main training log
config/overnight_s1/last.ckpt              -- Stage 1 checkpoint
config/overnight_s1/modulations/           -- extracted latent vectors
config/overnight_s2_ddpm/last.ckpt         -- Stage 2 DDPM checkpoint
config/overnight_s2_fm/last.ckpt           -- Stage 2 FM checkpoint
config/overnight_s3_ddpm/last.ckpt         -- Stage 3 DDPM checkpoint
config/overnight_s3_fm/last.ckpt           -- Stage 3 FM checkpoint
tensorboard_logs/config/overnight_*/       -- TensorBoard event files
config/overnight_*/epoch=*.ckpt            -- periodic checkpoints (every log_freq epochs)
```
