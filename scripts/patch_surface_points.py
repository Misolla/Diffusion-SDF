"""Patch preprocessed sdf_data.csv files to include surface points with SDF=0.

The mesh-to-sdf library produces near-surface points with non-zero SDF, but
the training pipeline requires exact-zero SDF entries for point cloud extraction.
This script samples points directly on the mesh surface and replaces a portion
of the near-surface points to maintain the same total row count.

Usage:
  conda run -n diffusionsdf python scripts/patch_surface_points.py --workers 10
"""
import os
import sys
import json
import argparse
import numpy as np
from multiprocessing import Pool

import trimesh

ROOT = os.path.join(os.path.dirname(__file__), "..")
SHAPENET_DIR = "/tmp/shapenet_extract/ShapeNetCore.v2/ShapeNetCore.v2/04256520"
SDF_OUT_DIR = os.path.join(ROOT, "data/acronym/Couch")

SURFACE_COUNT = 231000
TOTAL_COUNT = 596000


def patch_instance(instance_id):
    sdf_path = os.path.join(SDF_OUT_DIR, instance_id, "sdf_data.csv")
    mesh_path = os.path.join(SHAPENET_DIR, instance_id, "models", "model_normalized.ply")

    if not os.path.exists(sdf_path) or not os.path.exists(mesh_path):
        return instance_id, "skip"

    try:
        data = np.loadtxt(sdf_path, delimiter=",")
        n_zeros = np.sum(data[:, -1] == 0)
        if n_zeros >= 1024:
            return instance_id, f"already_ok ({n_zeros} surface pts)"

        mesh = trimesh.load(mesh_path, force="mesh")
        vertices = mesh.vertices - mesh.bounding_box.centroid
        scale = np.max(np.abs(vertices))
        if scale > 0:
            vertices = vertices / scale
        mesh.vertices = vertices

        surface_pts = mesh.sample(SURFACE_COUNT)
        surface_data = np.column_stack([surface_pts, np.zeros(SURFACE_COUNT)])

        near_surface_count = TOTAL_COUNT - SURFACE_COUNT
        near_surface_data = data[data[:, -1] != 0]
        if len(near_surface_data) > near_surface_count:
            idx = np.random.choice(len(near_surface_data), near_surface_count, replace=False)
            near_surface_data = near_surface_data[idx]
        elif len(near_surface_data) < near_surface_count:
            idx = np.random.choice(len(near_surface_data), near_surface_count, replace=True)
            near_surface_data = near_surface_data[idx]

        patched = np.vstack([near_surface_data, surface_data])
        np.random.shuffle(patched)
        np.savetxt(sdf_path, patched, delimiter=",", fmt="%.6g")

        return instance_id, "patched"
    except Exception as e:
        return instance_id, f"error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    instances = [
        d for d in os.listdir(SDF_OUT_DIR)
        if os.path.isdir(os.path.join(SDF_OUT_DIR, d))
    ]
    print(f"Patching {len(instances)} instances...")

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(patch_instance, instances))
    else:
        results = [patch_instance(iid) for iid in instances]

    for iid, status in sorted(results):
        print(f"  {iid}: {status}")

    patched = sum(1 for _, s in results if s == "patched")
    ok = sum(1 for _, s in results if "already_ok" in s)
    print(f"\nDone: {patched} patched, {ok} already ok, {len(results) - patched - ok} other")


if __name__ == "__main__":
    main()
