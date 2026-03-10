"""Benchmark inference speed: DDPM vs Flow Matching.

No training required -- uses randomly initialized weights since generation
time depends only on the number of network forward passes, not weight quality.

Usage:
  conda activate diffusionsdf
  python scripts/benchmark_inference.py              # GPU
  python scripts/benchmark_inference.py --cpu         # CPU (if GPU is busy)
  python scripts/benchmark_inference.py --runs 20     # more runs for stability
"""
import os
import sys
import gc
import time
import resource
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def set_memory_limit_gb(gb):
    """Set a hard RSS limit so the process dies instead of swapping."""
    limit_bytes = int(gb * 1024 * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    print(f"Memory limit set to {gb} GB")

from models.archs.diffusion_arch import DiffusionNet
from models.diffusion import DiffusionModel
from models.flow_matching import FlowMatchingModel


def make_ddpm(cond=True, ddim_steps=None):
    net = DiffusionNet(
        dim=768, depth=4, ff_dropout=0.0,
        cond=cond, cross_attn=cond, cond_dropout=False,
        point_feature_dim=128,
    )
    specs = dict(
        timesteps=1000, objective="pred_x0", loss_type="l2",
        perturb_pc="partial", crop_percent=0.5, sample_pc_size=128,
    )
    if ddim_steps is not None:
        specs["sampling_timesteps"] = ddim_steps
    return DiffusionModel(model=net, **specs)


def make_fm(cond=True, num_steps=50, solver="euler"):
    net = DiffusionNet(
        dim=768, depth=4, ff_dropout=0.0,
        num_timesteps=None,
        cond=cond, cross_attn=cond, cond_dropout=False,
        point_feature_dim=128,
    )
    return FlowMatchingModel(
        model=net, num_sample_steps=num_steps, solver=solver,
        perturb_pc="partial", crop_percent=0.5, sample_pc_size=128,
    )


def benchmark_generation(model, device, num_runs=10, cond_pc=None, label=""):
    """Time unconditional or conditional generation."""
    model = model.to(device).eval()

    def _run():
        if cond_pc is not None:
            model.generate_from_pc(cond_pc)
        else:
            model.generate_unconditional(num_samples=1)

    # Warmup (1 run, not timed)
    with torch.no_grad():
        _run()

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            _run()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    print(f"  {label:30s}  {times.mean():.3f}s +/- {times.std():.3f}s  "
          f"(min={times.min():.3f}s, max={times.max():.3f}s)")
    return times.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU (if GPU is busy)")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed runs")
    parser.add_argument("--mem-limit", type=float, default=8.0, help="Max RAM in GB (default 8)")
    args = parser.parse_args()

    set_memory_limit_gb(args.mem_limit)

    device = torch.device("cpu" if args.cpu else "cuda")
    print(f"Device: {device}")
    print(f"Runs per config: {args.runs}")
    print()

    latent_dim = 768
    pc = torch.randn(1, 10000, 3, device=device)

    results = {}

    def run_bench(key, label, make_fn, cond_pc=None):
        model = make_fn()
        results[key] = benchmark_generation(
            model, device, args.runs, cond_pc=cond_pc, label=label)
        del model
        gc.collect()

    # --- Unconditional ---
    print("=== Unconditional Generation ===")

    run_bench("DDPM-1000", "DDPM (1000 steps)",
              lambda: make_ddpm(cond=False))
    run_bench("DDIM-100", "DDIM (100 steps)",
              lambda: make_ddpm(cond=False, ddim_steps=100))
    run_bench("DDIM-50", "DDIM (50 steps)",
              lambda: make_ddpm(cond=False, ddim_steps=50))
    run_bench("FM-50-euler", "FM Euler (50 steps)",
              lambda: make_fm(cond=False, num_steps=50, solver="euler"))
    run_bench("FM-50-mid", "FM Midpoint (50 steps)",
              lambda: make_fm(cond=False, num_steps=50, solver="midpoint"))
    run_bench("FM-20-euler", "FM Euler (20 steps)",
              lambda: make_fm(cond=False, num_steps=20, solver="euler"))
    run_bench("FM-10-euler", "FM Euler (10 steps)",
              lambda: make_fm(cond=False, num_steps=10, solver="euler"))

    # --- Conditional ---
    print()
    print("=== Conditional Generation (from point cloud) ===")

    run_bench("cond-DDPM-1000", "DDPM (1000 steps)",
              lambda: make_ddpm(cond=True), cond_pc=pc)
    run_bench("cond-DDIM-50", "DDIM (50 steps)",
              lambda: make_ddpm(cond=True, ddim_steps=50), cond_pc=pc)
    run_bench("cond-FM-50", "FM Euler (50 steps)",
              lambda: make_fm(cond=True, num_steps=50, solver="euler"), cond_pc=pc)
    run_bench("cond-FM-20", "FM Euler (20 steps)",
              lambda: make_fm(cond=True, num_steps=20, solver="euler"), cond_pc=pc)

    # --- Summary ---
    print()
    print("=== Speedup Summary ===")
    baseline = results["DDPM-1000"]
    for k, v in results.items():
        if k.startswith("cond-"):
            continue
        print(f"  {k:20s}  {baseline/v:.1f}x vs DDPM-1000")

    if "cond-DDPM-1000" in results:
        baseline_c = results["cond-DDPM-1000"]
        print()
        for k, v in results.items():
            if not k.startswith("cond-"):
                continue
            print(f"  {k:20s}  {baseline_c/v:.1f}x vs cond-DDPM-1000")


if __name__ == "__main__":
    main()
