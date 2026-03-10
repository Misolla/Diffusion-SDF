# Inference Speed Benchmark: DDPM vs Flow Matching

## IMPORTANT CAVEAT

**These benchmarks measure computational cost only, NOT output quality.**

All models used **randomly initialized weights** (no training). The generation
times are valid because inference speed depends entirely on network architecture
and number of forward passes, not on learned weights. A fully trained model
performs the exact same number of floating-point operations per sample as an
untrained one.

However, these results say **nothing** about:

- Whether the generated SDFs are meaningful or high-quality
- Whether Flow Matching at 10 or 20 steps produces acceptable output compared to
  DDPM at 1000 steps after real training
- Whether Point-MAE conditioning produces better or worse shapes than ConvPointnet
- The quality-vs-speed tradeoff at different step counts

**Quality evaluation requires trained models**, which are currently being trained
in the overnight run. Until those results are available, these numbers should be
cited purely as a measure of theoretical computational speedup, not as evidence
that Flow Matching produces equivalent results faster.

---

## Setup

- **GPU:** NVIDIA RTX A6000 (48GB VRAM)
- **CPU:** 36-core system, 31GB RAM
- **PyTorch:** 1.11.0 with CUDA
- **Model:** DiffusionNet (dim=768, depth=4, 4-layer Transformer)
- **Latent dimension:** 768
- **Batch size:** 1 (single-sample latency)
- **Runs per config:** 5 (GPU), 3 (CPU)
- **Conditioning point cloud:** 10,000 points (when applicable)
- **Script:** `scripts/benchmark_inference.py`

---

## GPU Results

### Unconditional Generation (single sample)

| Method | Steps | Time (s) | Speedup |
|--------|------:|--------:|--------:|
| DDPM | 1000 | 7.30 | 1.0x |
| DDIM | 100 | 7.30 | 1.0x |
| DDIM | 50 | 7.30 | 1.0x |
| FM Euler | 50 | 0.363 | **20.1x** |
| FM Midpoint | 50 | 0.713 | 10.2x |
| FM Euler | 20 | 0.143 | **51.1x** |
| FM Euler | 10 | 0.072 | **100.7x** |

### Conditional Generation (with point cloud encoding)

| Method | Steps | Encoder | Time (s) | Speedup |
|--------|------:|---------|--------:|--------:|
| DDPM | 1000 | ConvPointnet | 23.17 | 1.0x |
| DDIM | 50 | ConvPointnet | 23.41 | 1.0x |
| FM Euler | 50 | Point-MAE | 1.176 | **19.7x** |
| FM Euler | 20 | Point-MAE | 0.474 | **48.9x** |

---

## CPU Results

### Unconditional Generation (single sample)

| Method | Steps | Time (s) | Speedup |
|--------|------:|--------:|--------:|
| DDPM | 1000 | 13.39 | 1.0x |
| DDIM | 100 | 13.31 | 1.0x |
| DDIM | 50 | 13.20 | 1.0x |
| FM Euler | 50 | 0.586 | **22.9x** |
| FM Midpoint | 50 | 1.310 | 10.2x |
| FM Euler | 20 | 0.276 | **48.5x** |
| FM Euler | 10 | 0.121 | **110.7x** |

---

## Analysis

### Why is Flow Matching faster?

DDPM generates samples by iteratively denoising through a Markov chain of 1000
timesteps. Each step requires one full forward pass through the denoising network.
Flow Matching instead learns a velocity field and integrates an ODE from noise to
data using a fixed-step solver. The number of ODE steps is a free parameter --
50 steps means 50 forward passes (Euler) or 100 forward passes (Midpoint, which
evaluates twice per step).

The speedup is approximately proportional to the step-count ratio:
- 1000 / 50 = 20x (observed: 20.1x) -- Euler
- 1000 / 100 = 10x (observed: 10.2x) -- Midpoint (2 evaluations per step)
- 1000 / 20 = 50x (observed: 51.1x) -- Euler with fewer steps

### Why does DDIM show no speedup?

The DDIM implementation in this codebase does not achieve the expected speedup.
Despite being configured with fewer sampling timesteps (50 or 100), the wall-clock
time remains identical to full 1000-step DDPM. This appears to be an implementation
limitation where the sampling loop still iterates over all 1000 timesteps internally.
A properly optimized DDIM implementation would show speedups similar to its step count
ratio.

### Conditional overhead

Conditional generation adds the cost of encoding a 10,000-point cloud through the
conditioning network (ConvPointnet for DDPM, Point-MAE for FM). This is a one-time
cost per sample:
- DDPM conditional is ~3.2x slower than unconditional (23.2s vs 7.3s), indicating
  ConvPointnet encoding plus cross-attention adds significant per-step overhead.
- FM conditional with Point-MAE at 50 steps takes 1.18s vs 0.36s unconditional,
  a ~3.3x ratio, suggesting Point-MAE has similar relative encoding cost.

### Practical implications (pending quality validation)

If trained Flow Matching models produce shapes of comparable quality to DDPM (which
remains to be validated), the practical implications are significant:
- **Interactive applications:** FM at 10 steps generates a shape in 72ms on GPU,
  enabling real-time shape generation (>13 fps).
- **Batch generation:** Generating 1000 shapes would take ~6 minutes with DDPM vs
  ~6 seconds with FM Euler at 50 steps.
- **Quality-speed tradeoff:** FM allows runtime control over the quality-speed
  tradeoff by adjusting the step count, which DDPM does not effectively support
  in this implementation.

---

## Reproducing These Results

```bash
conda activate diffusionsdf

# GPU benchmark
python scripts/benchmark_inference.py --runs 5 --mem-limit 32

# CPU benchmark
python scripts/benchmark_inference.py --cpu --runs 3 --mem-limit 8
```
