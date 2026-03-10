"""Extract latent modulation vectors from a trained Stage 1 (SDF-VAE) checkpoint.

Reads all SDF CSVs referenced by the split file, extracts point clouds (SDF==0),
runs them through the trained PointNet + VAE encoder, and saves latent.txt files
that Stage 2 uses for diffusion/flow-matching training.

Usage:
  python scripts/extract_modulations.py \
      --checkpoint config/overnight_s1/last.ckpt \
      --split data/splits/couch_mini.json \
      --output config/overnight_s1/modulations
"""
import os
import sys
import json
import argparse

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.combined_model import CombinedModel


SPECS_FOR_LOAD = {
    "Description": "modulation extraction",
    "training_task": "modulation",
    "SdfModelSpecs": {
        "hidden_dim": 512,
        "latent_dim": 256,
        "pn_hidden_dim": 128,
        "num_layers": 9,
    },
    "SampPerMesh": 16000,
    "PCsize": 1024,
    "num_epochs": 1,
    "log_freq": 1,
    "kld_weight": 1e-5,
    "latent_std": 0.25,
    "sdf_lr": 1e-4,
}

PC_SIZE = 1024


def load_pc_from_csv(csv_path, pc_size=PC_SIZE):
    """Load point cloud (SDF==0 surface points) from an SDF CSV file."""
    data = pd.read_csv(csv_path, sep=",", header=None).values
    surface = data[data[:, -1] == 0][:, :3]
    if surface.shape[0] == 0:
        return None
    if surface.shape[0] < pc_size:
        idx = np.random.choice(surface.shape[0], pc_size, replace=True)
    else:
        idx = np.random.choice(surface.shape[0], pc_size, replace=False)
    return torch.from_numpy(surface[idx]).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to Stage 1 last.ckpt")
    parser.add_argument("--split", required=True, help="Path to split JSON")
    parser.add_argument("--output", required=True, help="Output directory for modulations")
    parser.add_argument("--data_source", default="data", help="Root of SDF CSV data")
    args = parser.parse_args()

    split = json.load(open(args.split))
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = CombinedModel.load_from_checkpoint(
        args.checkpoint, specs=SPECS_FOR_LOAD, strict=False
    )
    model = model.cuda().eval()

    csv_paths = []
    class_instance_pairs = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                csv_path = os.path.join(
                    args.data_source, dataset, class_name, instance_name, "sdf_data.csv"
                )
                if os.path.isfile(csv_path):
                    csv_paths.append(csv_path)
                    class_instance_pairs.append((class_name, instance_name))

    print(f"Found {len(csv_paths)} instances to extract")
    extracted = 0
    skipped = 0

    with torch.no_grad():
        for csv_path, (cls, inst) in tqdm(
            zip(csv_paths, class_instance_pairs), total=len(csv_paths)
        ):
            pc = load_pc_from_csv(csv_path, PC_SIZE)
            if pc is None:
                skipped += 1
                continue

            pc = pc.unsqueeze(0).cuda()
            features = model.sdf_model.pointnet.get_plane_features(pc)
            features = torch.cat(features, dim=1)
            latent = model.vae_model.get_latent(features)

            outdir = os.path.join(args.output, cls, inst)
            os.makedirs(outdir, exist_ok=True)
            np.savetxt(os.path.join(outdir, "latent.txt"), latent.cpu().numpy())
            extracted += 1

    print(f"Extraction complete: {extracted} saved, {skipped} skipped")


if __name__ == "__main__":
    main()
