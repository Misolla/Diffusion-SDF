"""Preprocess ShapeNet PLY meshes into SDF CSV files for Diffusion-SDF training.

Generates two files per instance:
  data/acronym/Couch/{id}/sdf_data.csv  -- near-surface SDF samples
  data/grid_data/acronym/Couch/{id}/grid_gt.csv  -- uniform grid SDF samples

Usage:
  # Process a single mesh (for benchmarking):
  python scripts/preprocess_sdf.py --instance 37cfcafe606611d81246538126da07a8

  # Process all instances from the split that have ShapeNet meshes:
  python scripts/preprocess_sdf.py --all --workers 12

  # Process first N instances:
  python scripts/preprocess_sdf.py --all --limit 10 --workers 4
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial

os.environ["PYOPENGL_PLATFORM"] = "egl"

import trimesh

ROOT = os.path.join(os.path.dirname(__file__), "..")
SHAPENET_DIR = "/tmp/shapenet_extract/ShapeNetCore.v2/ShapeNetCore.v2/04256520"
SPLIT_FILE = os.path.join(ROOT, "data/splits/couch_all.json")
SDF_OUT_DIR = os.path.join(ROOT, "data/acronym/Couch")
GRID_OUT_DIR = os.path.join(ROOT, "data/grid_data/acronym/Couch")

TOTAL_SDF_COUNT = 596000
SURFACE_COUNT = 231000
NEAR_SURFACE_COUNT = TOTAL_SDF_COUNT - SURFACE_COUNT
GRID_COUNT = 468000


def normalize_mesh(mesh):
    """Center and scale mesh to fit in [-1, 1]^3."""
    vertices = mesh.vertices - mesh.bounding_box.centroid
    scale = np.max(np.abs(vertices))
    if scale > 0:
        vertices = vertices / scale
    mesh.vertices = vertices
    return mesh


def process_instance(instance_id, force=False):
    """Generate sdf_data.csv and grid_gt.csv for one instance."""
    sdf_path = os.path.join(SDF_OUT_DIR, instance_id, "sdf_data.csv")
    grid_path = os.path.join(GRID_OUT_DIR, instance_id, "grid_gt.csv")

    if not force and os.path.exists(sdf_path) and os.path.exists(grid_path):
        return instance_id, "skipped", 0.0

    mesh_path = os.path.join(SHAPENET_DIR, instance_id, "models", "model_normalized.ply")
    if not os.path.exists(mesh_path):
        return instance_id, "missing_mesh", 0.0

    t0 = time.time()
    try:
        from mesh_to_sdf import sample_sdf_near_surface, mesh_to_sdf

        mesh = trimesh.load(mesh_path, force="mesh")
        mesh = normalize_mesh(mesh)

        # Near-surface samples (70% of training batch comes from these)
        points_near, sdf_near = sample_sdf_near_surface(
            mesh,
            number_of_points=NEAR_SURFACE_COUNT,
            surface_point_method="scan",
            sign_method="normal",
            scan_count=100,
            scan_resolution=400,
            sample_point_count=10000000,
            normal_sample_count=11,
            min_size=0,
        )

        # Exact surface points (SDF=0) needed by the dataloader for point cloud extraction
        surface_pts = mesh.sample(SURFACE_COUNT)
        data_near = np.column_stack([points_near, sdf_near])
        data_surface = np.column_stack([surface_pts, np.zeros(SURFACE_COUNT)])
        data_all = np.vstack([data_near, data_surface])
        np.random.shuffle(data_all)

        os.makedirs(os.path.join(SDF_OUT_DIR, instance_id), exist_ok=True)
        np.savetxt(sdf_path, data_all, delimiter=",", fmt="%.6g")

        # Uniform grid samples (30% of training batch)
        points_grid = np.random.uniform(-1.0, 1.0, size=(GRID_COUNT, 3))
        sdf_grid = mesh_to_sdf(
            mesh,
            points_grid,
            surface_point_method="scan",
            sign_method="normal",
            scan_count=100,
            scan_resolution=400,
            sample_point_count=10000000,
            normal_sample_count=11,
        )

        os.makedirs(os.path.join(GRID_OUT_DIR, instance_id), exist_ok=True)
        data_grid = np.column_stack([points_grid, sdf_grid])
        np.savetxt(grid_path, data_grid, delimiter=",", fmt="%.6g")

        elapsed = time.time() - t0
        return instance_id, "ok", elapsed

    except Exception as e:
        elapsed = time.time() - t0
        return instance_id, f"error: {e}", elapsed


def get_available_instances():
    """Return instance IDs from the split that have ShapeNet meshes."""
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    ids = split["acronym"]["Couch"]

    available = []
    for iid in ids:
        mesh_path = os.path.join(SHAPENET_DIR, iid, "models", "model_normalized.ply")
        if os.path.exists(mesh_path):
            available.append(iid)
    return available


def main():
    parser = argparse.ArgumentParser(description="Preprocess ShapeNet meshes to SDF CSVs")
    parser.add_argument("--instance", type=str, help="Process a single instance ID")
    parser.add_argument("--all", action="store_true", help="Process all available instances")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances to process")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()

    if args.instance:
        instances = [args.instance]
    elif args.all:
        instances = get_available_instances()
        print(f"Found {len(instances)} instances with ShapeNet meshes")
        if args.limit:
            instances = instances[:args.limit]
            print(f"Processing first {len(instances)}")
    else:
        parser.print_help()
        return

    func = partial(process_instance, force=args.force)

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = []
            for result in pool.imap_unordered(func, instances):
                iid, status, elapsed = result
                results.append(result)
                print(f"[{len(results)}/{len(instances)}] {iid}: {status} ({elapsed:.1f}s)")
    else:
        results = []
        for iid in instances:
            result = func(iid)
            results.append(result)
            _, status, elapsed = result
            print(f"[{len(results)}/{len(instances)}] {iid}: {status} ({elapsed:.1f}s)")

    ok = sum(1 for _, s, _ in results if s == "ok")
    skipped = sum(1 for _, s, _ in results if s == "skipped")
    failed = sum(1 for _, s, _ in results if s not in ("ok", "skipped"))
    times = [t for _, s, t in results if s == "ok"]
    avg_time = np.mean(times) if times else 0

    print(f"\nDone: {ok} processed, {skipped} skipped, {failed} failed")
    if times:
        print(f"Average time per mesh: {avg_time:.1f}s")
        remaining = len(instances) - ok - skipped
        if remaining > 0:
            print(f"Estimated time for remaining {remaining}: {remaining * avg_time / 60:.1f} min")


if __name__ == "__main__":
    main()
