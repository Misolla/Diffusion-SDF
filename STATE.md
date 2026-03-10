# Diffusion-SDF Project State -- March 9, 2026

## Quick Context

We are upgrading the Diffusion-SDF generative model (ECE 285 project) by replacing
DDPM with **Flow Matching** and ConvPointnet with **Point-MAE**, then training both
the original and upgraded architectures side-by-side for comparison on a couch dataset.

## Environment

- **Machine:** Linux 6.17.0-14-generic, NVIDIA RTX A6000 (48GB VRAM)
- **Conda env:** `diffusionsdf` (PyTorch 1.11.0, CUDA)
- **Working directory:** `/home/max/Documents/Diffusion-SDF`
- **Git branch:** `feature/flow-matching-point-mae`

## Git History (most recent first)

```
b3fa39b Integrate Point-MAE into DiffusionNet conditioning with config support
85590ea Add Point-MAE encoder with pure-PyTorch FPS/KNN and unit tests
c769ef6 Add training smoke tests for flow matching pipeline
7996450 Integrate FlowMatchingModel into CombinedModel with config support
ce93b3e Add FlowMatchingModel with CFM loss, ODE sampling, and unit tests
7eb2d67 Update README.md
e2fc1a7 refactor
96aaf37 MIT license
```

## What Has Been Done

### 1. Flow Matching Implementation (complete, committed)
- `models/flow_matching.py` -- `FlowMatchingModel` class, drop-in replacement for `DiffusionModel`
- Uses Conditional Flow Matching (CFM) with linear interpolation and ODE solvers (Euler/Midpoint)
- Supports classifier-free guidance
- `models/combined_model.py` updated to select DDPM vs FM via `specs["generative_model"]`

### 2. Point-MAE Integration (complete, committed)
- `models/archs/encoders/point_mae.py` -- Transformer-based point cloud encoder
- Pure-PyTorch FPS and KNN (no external CUDA dependencies)
- `models/archs/diffusion_arch.py` updated to select ConvPointnet vs Point-MAE via `specs["cond_encoder"]`

### 3. Unit and Integration Tests (complete, committed)
- `tests/test_flow_matching.py` -- 11 unit tests for FlowMatchingModel
- `tests/test_point_mae.py` -- 9 unit tests for PointMAEEncoder
- `tests/test_integration.py` -- CombinedModel integration tests
- `tests/test_full_pipeline.py` -- end-to-end pipeline tests
- `tests/test_training_smoke.py` -- short training loop smoke tests

### 4. Data Preprocessing (complete, NOT committed)
- ShapeNet sofa meshes extracted from `shapenet_core_v2.zip`
- `scripts/preprocess_sdf.py` converted 173 meshes to SDF CSV format
- Each instance has `data/acronym/Couch/{id}/sdf_data.csv` (near-surface + exact surface points)
- Each instance has `data/grid_data/acronym/Couch/{id}/grid_gt.csv` (uniform grid samples)
- Split file: `data/splits/couch_mini.json` (173 instances)
- `scripts/patch_surface_points.py` was used to retroactively add SDF=0 surface points

### 5. Training Verification (complete, NOT committed)
- `scripts/verify_training.py` ran all 3 stages with 3 epochs each -- ALL PASSED
- Output logged to `verify-log.log`, artifacts in `verify_output/`

### 6. Overnight Training Setup (complete, NOT committed)
- 5 config directories under `config/overnight_s*/` with reduced epoch counts
- `scripts/extract_modulations.py` -- standalone modulation extraction
- `scripts/run_overnight.sh` -- orchestrates all 6 training phases

## What Has NOT Been Done

- **Training has NOT started.** No checkpoints, no modulations, no TensorBoard logs yet.
- The overnight training script has been written but NOT launched.
- No inference/generation results exist yet.
- DINOv3 image conditioning was explicitly deferred (no 2D image conditioning in codebase).

## Overnight Training Plan

### How to Launch

```bash
screen -S training
conda activate diffusionsdf
cd /home/max/Documents/Diffusion-SDF
bash scripts/run_overnight.sh 2>&1 | tee overnight.log
# Ctrl-A D to detach
```

### 6 Phases (~8-9 hours total)

| Phase | Config Dir | Task | Epochs | Est. Time |
|-------|-----------|------|--------|-----------|
| 1 | `config/overnight_s1` | SDF-VAE (shared) | 20,000 | ~4.5h |
| 2 | -- | Extract modulations | -- | ~5 min |
| 3 | `config/overnight_s2_ddpm` | DDPM diffusion | 20,000 | ~30 min |
| 4 | `config/overnight_s2_fm` | FM + Point-MAE | 20,000 | ~30 min |
| 5 | `config/overnight_s3_ddpm` | DDPM end-to-end | 5,000 | ~1.5h |
| 6 | `config/overnight_s3_fm` | FM end-to-end | 5,000 | ~1.5h |

### Checkpoint Locations (after training)

```
config/overnight_s1/last.ckpt          -- Stage 1 SDF-VAE
config/overnight_s1/modulations/       -- extracted latent.txt files
config/overnight_s2_ddpm/last.ckpt     -- Stage 2 DDPM
config/overnight_s2_fm/last.ckpt       -- Stage 2 Flow Matching
config/overnight_s3_ddpm/last.ckpt     -- Stage 3 DDPM end-to-end
config/overnight_s3_fm/last.ckpt       -- Stage 3 FM end-to-end
```

### TensorBoard

```bash
conda activate diffusionsdf
tensorboard --logdir tensorboard_logs/
```

## Key Files Reference

### New/Modified Source Files
| File | Purpose |
|------|---------|
| `models/flow_matching.py` | FlowMatchingModel (CFM + ODE sampling) |
| `models/archs/encoders/point_mae.py` | Point-MAE encoder |
| `models/archs/diffusion_arch.py` | Modified to support Point-MAE via `cond_encoder` param |
| `models/combined_model.py` | Modified to support FM via `generative_model` param |

### Scripts
| File | Purpose |
|------|---------|
| `scripts/run_overnight.sh` | Main overnight training orchestrator |
| `scripts/extract_modulations.py` | Extract latent vectors from Stage 1 checkpoint |
| `scripts/preprocess_sdf.py` | Convert ShapeNet PLY to SDF CSV |
| `scripts/patch_surface_points.py` | Retroactively add SDF=0 points to CSVs |
| `scripts/verify_training.py` | 3-epoch smoke test of all stages |

### Configs
| Directory | Description |
|-----------|-------------|
| `config/overnight_s1/` | Stage 1 SDF-VAE, 20K epochs, couch_mini |
| `config/overnight_s2_ddpm/` | Stage 2 DDPM, 20K epochs |
| `config/overnight_s2_fm/` | Stage 2 FM + Point-MAE, 20K epochs |
| `config/overnight_s3_ddpm/` | Stage 3 DDPM E2E, 5K epochs |
| `config/overnight_s3_fm/` | Stage 3 FM E2E, 5K epochs |
| `config/stage2_fm_cond/` | FM config (original, full epochs, couch_all) |
| `config/stage3_fm_cond/` | FM E2E config (original, full epochs) |

### Data
| Path | Description |
|------|-------------|
| `data/acronym/Couch/{id}/sdf_data.csv` | Near-surface SDF + surface points |
| `data/grid_data/acronym/Couch/{id}/grid_gt.csv` | Uniform grid SDF |
| `data/splits/couch_mini.json` | 173-instance split |
| `data/splits/couch_all.json` | Original 300-instance split |

## Architecture Overview

```
Stage 1 (SDF-VAE):
  PointCloud -> ConvPointnet -> PlaneFeatures -> BetaVAE -> Latent (768-dim)
  Latent -> VAE Decode -> PlaneFeatures -> SDF MLP -> SDF values
  Loss: L1(SDF) + KL(VAE)

Stage 2 (Diffusion/FM on latents):
  DDPM path:  Latent + PointCloud -> ConvPointnet -> DiffusionNet -> denoise latent
  FM path:    Latent + PointCloud -> Point-MAE   -> DiffusionNet -> velocity field

Stage 3 (End-to-end):
  Load Stage 1 + Stage 2 checkpoints, train jointly
  Generated latent -> VAE Decode -> SDF MLP -> additional SDF loss
```

## Known Issues / Gotchas

1. **PCloader has [:5] limit** on line 30 of `dataloader/pc_loader.py`. The extraction
   script (`scripts/extract_modulations.py`) bypasses this by reading CSVs directly.

2. **torch_scatter must match CUDA version.** If reinstalling:
   ```bash
   pip install --force-reinstall torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
   ```

3. **SDF data must contain exact SDF=0 points.** The dataloader extracts surface point
   clouds by filtering `data[:, -1] == 0`. If no exact zeros exist, training crashes
   with `max(): Expected reduction dim to be specified for input.numel() == 0`.

4. **GridSource path is `"data/grid_data"`**, not `"data"` or `"grid_data"`.
   The SdfLoader constructs `os.path.join(grid_source, dataset, class_name, instance, "grid_gt.csv")`.

5. **Stage 3 finetune loading** uses `strict=False` for the modulation checkpoint and
   manually loads diffusion weights. The `-r finetune` flag triggers this in `train.py`.

## What To Do Tomorrow

1. **Check training status:** `screen -r training` or `tail -50 overnight.log`
2. **If training completed:** compare TensorBoard loss curves, run generation
3. **If training failed:** see `DEBUGGING_GUIDE.md`
4. **If not started yet:** launch with the command above
5. **Possible next steps:**
   - Run inference from Stage 3 checkpoints to generate and visualize SDF meshes
   - Write quantitative evaluation script (if needed for report)
   - Commit training configs and scripts to git
   - Consider longer training if results are promising
