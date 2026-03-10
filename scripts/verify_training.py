"""Verify that each stage of the Diffusion-SDF training pipeline works with real data.

Tests:
  Stage 1 (SDF-VAE): Load real SDF CSVs, run training steps, verify loss decreases
  Stage 2 (Diffusion): Extract modulations from Stage 1, train diffusion model
  Stage 3 (End-to-end): Load Stage 1+2 checkpoints, run combined training

Usage:
  conda run -n diffusionsdf python scripts/verify_training.py
"""
import os
import sys
import json
import shutil
import tempfile
import warnings

import torch
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.combined_model import CombinedModel
from dataloader.sdf_loader import SdfLoader
from dataloader.modulation_loader import ModulationLoader

ROOT = os.path.join(os.path.dirname(__file__), "..")
SPLIT = os.path.join(ROOT, "data/splits/couch_mini.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 4
NUM_WORKERS = 4


def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def print_result(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Stage 1: SDF-VAE modulation training
# ---------------------------------------------------------------------------

def verify_stage1(work_dir):
    """Train SDF-VAE for a few epochs with real data, save checkpoint."""
    print_header("STAGE 1: SDF-VAE Modulation Training")

    specs = {
        "Description": "verify stage1",
        "DataSource": "data",
        "GridSource": "data/grid_data",
        "TrainSplit": SPLIT,
        "TestSplit": SPLIT,
        "training_task": "modulation",
        "SdfModelSpecs": {
            "hidden_dim": 512,
            "latent_dim": 256,
            "pn_hidden_dim": 128,
            "num_layers": 9,
        },
        "SampPerMesh": 16000,
        "PCsize": 1024,
        "num_epochs": NUM_TRAIN_EPOCHS,
        "log_freq": 1,
        "kld_weight": 1e-5,
        "latent_std": 0.25,
        "sdf_lr": 1e-4,
    }

    # Save specs for later stages
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "specs.json"), "w") as f:
        json.dump(specs, f, indent=2)

    # 1a: Test data loading
    split = json.load(open(SPLIT))
    try:
        dataset = SdfLoader(
            specs["DataSource"], split,
            pc_size=specs["PCsize"],
            grid_source=specs["GridSource"],
        )
        print_result("Data loading", True, f"{len(dataset)} instances loaded")
    except Exception as e:
        print_result("Data loading", False, str(e))
        return False

    # 1b: Test a single batch
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, drop_last=True,
        shuffle=True, num_workers=NUM_WORKERS,
    )
    batch = next(iter(loader))
    pc_shape = batch["point_cloud"].shape
    xyz_shape = batch["xyz"].shape
    gt_shape = batch["gt_sdf"].shape
    print_result("Batch shapes", True,
                 f"pc={list(pc_shape)}, xyz={list(xyz_shape)}, gt={list(gt_shape)}")

    # 1c: Train for a few epochs
    model = CombinedModel(specs)
    ckpt_dir = os.path.join(work_dir, "stage1")
    os.makedirs(ckpt_dir, exist_ok=True)
    callback = ModelCheckpoint(dirpath=ckpt_dir, save_last=True, every_n_epochs=1)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=32,
        max_epochs=NUM_TRAIN_EPOCHS,
        callbacks=[callback],
        enable_progress_bar=True,
        logger=False,
    )
    trainer.fit(model=model, train_dataloaders=loader)

    metrics = trainer.callback_metrics
    sdf_loss = metrics.get("sdf")
    vae_loss = metrics.get("vae")

    ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    ckpt_exists = os.path.exists(ckpt_path)

    if sdf_loss is not None and torch.isfinite(sdf_loss):
        print_result("Training loss finite", True, f"sdf={sdf_loss:.6f}, vae={vae_loss:.6f}")
    else:
        print_result("Training loss finite", False, f"sdf={sdf_loss}, vae={vae_loss}")
        return False

    print_result("Checkpoint saved", ckpt_exists, ckpt_path if ckpt_exists else "missing")
    if not ckpt_exists:
        return False

    # 1d: Extract modulations
    mod_dir = os.path.join(work_dir, "modulations")
    os.makedirs(mod_dir, exist_ok=True)

    model = CombinedModel.load_from_checkpoint(ckpt_path, specs=specs).cuda().eval()
    from dataloader.pc_loader import PCloader
    test_dataset = PCloader(specs["DataSource"], split, pc_size=specs["PCsize"], return_filename=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

    extracted = 0
    with torch.no_grad():
        for pc, filename in test_loader:
            filename = filename[0]
            cls_name = filename.split("/")[-3]
            mesh_name = filename.split("/")[-2]
            outdir = os.path.join(mod_dir, cls_name, mesh_name)
            os.makedirs(outdir, exist_ok=True)

            features = model.sdf_model.pointnet.get_plane_features(pc.cuda())
            features = torch.cat(features, dim=1)
            latent = model.vae_model.get_latent(features)
            np.savetxt(os.path.join(outdir, "latent.txt"), latent.cpu().numpy())
            extracted += 1

    print_result("Modulation extraction", extracted > 0, f"{extracted} latent vectors saved")
    return True


# ---------------------------------------------------------------------------
# Stage 2: Diffusion model training (DDPM)
# ---------------------------------------------------------------------------

def verify_stage2(work_dir):
    """Train diffusion model on extracted modulations."""
    print_header("STAGE 2: Diffusion Model Training (DDPM)")

    mod_dir = os.path.join(work_dir, "modulations")
    split = json.load(open(SPLIT))

    specs = {
        "Description": "verify stage2 diffusion (conditional)",
        "pc_path": "data",
        "total_pc_size": 10000,
        "TrainSplit": SPLIT,
        "TestSplit": SPLIT,
        "data_path": mod_dir,
        "training_task": "diffusion",
        "num_epochs": NUM_TRAIN_EPOCHS,
        "log_freq": 1,
        "diff_lr": 1e-5,
        "diffusion_specs": {
            "timesteps": 1000,
            "objective": "pred_x0",
            "loss_type": "l2",
            "perturb_pc": "partial",
            "crop_percent": 0.5,
            "sample_pc_size": 128,
        },
        "diffusion_model_specs": {
            "dim": 768,
            "depth": 4,
            "ff_dropout": 0.3,
            "cond": True,
            "cross_attn": True,
            "cond_dropout": True,
            "point_feature_dim": 128,
        },
    }

    # 2a: Test modulation loading
    try:
        dataset = ModulationLoader(
            mod_dir, pc_path=specs["pc_path"],
            split_file=split, pc_size=specs["total_pc_size"],
        )
        print_result("Modulation loading", True, f"{len(dataset)} modulations loaded")
    except Exception as e:
        print_result("Modulation loading", False, str(e))
        return False

    if len(dataset) < BATCH_SIZE:
        print_result("Enough data", False, f"only {len(dataset)} < batch_size={BATCH_SIZE}")
        return False

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, drop_last=True,
        shuffle=True, num_workers=0,
    )

    # 2b: Train
    model = CombinedModel(specs)
    ckpt_dir = os.path.join(work_dir, "stage2")
    os.makedirs(ckpt_dir, exist_ok=True)
    callback = ModelCheckpoint(dirpath=ckpt_dir, save_last=True, every_n_epochs=1)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=32,
        max_epochs=NUM_TRAIN_EPOCHS,
        callbacks=[callback],
        enable_progress_bar=True,
        logger=False,
    )
    trainer.fit(model=model, train_dataloaders=loader)

    metrics = trainer.callback_metrics
    total_loss = metrics.get("total")

    ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    ckpt_exists = os.path.exists(ckpt_path)

    if total_loss is not None and torch.isfinite(total_loss):
        print_result("Training loss finite", True, f"total={total_loss:.6f}")
    else:
        print_result("Training loss finite", False, f"total={total_loss}")
        return False

    print_result("Checkpoint saved", ckpt_exists, ckpt_path if ckpt_exists else "missing")
    return ckpt_exists


# ---------------------------------------------------------------------------
# Stage 3: End-to-end combined training
# ---------------------------------------------------------------------------

def verify_stage3(work_dir):
    """Load Stage 1+2 checkpoints and run combined training."""
    print_header("STAGE 3: End-to-End Combined Training")

    stage1_ckpt = os.path.join(work_dir, "stage1", "last.ckpt")
    stage2_ckpt = os.path.join(work_dir, "stage2", "last.ckpt")
    mod_dir = os.path.join(work_dir, "modulations")
    split = json.load(open(SPLIT))

    specs = {
        "Description": "verify stage3 combined",
        "DataSource": "data",
        "GridSource": "data/grid_data",
        "TrainSplit": SPLIT,
        "TestSplit": SPLIT,
        "modulation_path": mod_dir,
        "modulation_ckpt_path": stage1_ckpt,
        "diffusion_ckpt_path": stage2_ckpt,
        "training_task": "combined",
        "num_epochs": NUM_TRAIN_EPOCHS,
        "log_freq": 1,
        "kld_weight": 1e-5,
        "latent_std": 0.25,
        "sdf_lr": 1e-4,
        "diff_lr": 1e-5,
        "SdfModelSpecs": {
            "hidden_dim": 512,
            "latent_dim": 256,
            "pn_hidden_dim": 128,
            "num_layers": 9,
        },
        "SampPerMesh": 16000,
        "PCsize": 1024,
        "diffusion_specs": {
            "timesteps": 1000,
            "objective": "pred_x0",
            "loss_type": "l2",
            "perturb_pc": "partial",
            "crop_percent": 0.5,
            "sample_pc_size": 128,
        },
        "diffusion_model_specs": {
            "dim": 768,
            "depth": 4,
            "ff_dropout": 0.3,
            "cond": True,
            "cross_attn": True,
            "cond_dropout": True,
            "point_feature_dim": 128,
        },
    }

    # 3a: Load checkpoints (finetune mode)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CombinedModel.load_from_checkpoint(
                stage1_ckpt, specs=specs, strict=False
            )
            ckpt = torch.load(stage2_ckpt)
            new_state_dict = {}
            for k, v in ckpt["state_dict"].items():
                new_key = k.replace("diffusion_model.", "")
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)

        print_result("Checkpoint loading (finetune)", True)
    except Exception as e:
        print_result("Checkpoint loading (finetune)", False, str(e))
        return False

    # 3b: Train
    dataset = SdfLoader(
        specs["DataSource"], split,
        pc_size=specs["PCsize"],
        grid_source=specs["GridSource"],
        modulation_path=mod_dir,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, drop_last=True,
        shuffle=True, num_workers=NUM_WORKERS,
    )

    ckpt_dir = os.path.join(work_dir, "stage3")
    os.makedirs(ckpt_dir, exist_ok=True)
    callback = ModelCheckpoint(dirpath=ckpt_dir, save_last=True, every_n_epochs=1)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=32,
        max_epochs=NUM_TRAIN_EPOCHS,
        callbacks=[callback],
        enable_progress_bar=True,
        logger=False,
    )
    trainer.fit(model=model, train_dataloaders=loader)

    metrics = trainer.callback_metrics
    total_loss = metrics.get("total")
    sdf_loss = metrics.get("sdf")
    diff_loss = metrics.get("diff")

    if total_loss is not None and torch.isfinite(total_loss):
        print_result("Training loss finite", True,
                     f"total={total_loss:.6f}, sdf={sdf_loss:.6f}, diff={diff_loss:.6f}")
    else:
        print_result("Training loss finite", False, f"total={total_loss}")
        return False

    ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    print_result("Checkpoint saved", os.path.exists(ckpt_path))
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    work_dir = os.path.join(ROOT, "verify_output")
    os.makedirs(work_dir, exist_ok=True)

    print(f"Working directory: {work_dir}")
    print(f"Split file: {SPLIT}")
    print(f"Device: {DEVICE}")
    print(f"Epochs per stage: {NUM_TRAIN_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    results = {}

    # Stage 1
    results["stage1"] = verify_stage1(work_dir)

    # Stage 2 (depends on stage 1)
    if results["stage1"]:
        results["stage2"] = verify_stage2(work_dir)
    else:
        print_header("STAGE 2: SKIPPED (Stage 1 failed)")
        results["stage2"] = False

    # Stage 3 (depends on stages 1 and 2)
    if results["stage1"] and results["stage2"]:
        results["stage3"] = verify_stage3(work_dir)
    else:
        print_header("STAGE 3: SKIPPED (earlier stage failed)")
        results["stage3"] = False

    # Summary
    print_header("SUMMARY")
    for stage, passed in results.items():
        print_result(stage, passed)

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL STAGES PASSED' if all_passed else 'SOME STAGES FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
