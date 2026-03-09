"""CHECKPOINT 5: Full pipeline (Flow Matching + Point-MAE) integration tests.

Verifies end-to-end forward pass, training step, and generation
with both Flow Matching and Point-MAE conditioning encoder.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytorch_lightning as pl
import pytest

from models.combined_model import CombinedModel
from models.flow_matching import FlowMatchingModel
from models.archs.encoders.point_mae import PointMAEEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.path.dirname(__file__), "..")


class SyntheticModulationDataset(torch.utils.data.Dataset):
    def __init__(self, n=32, latent_dim=768, pc_size=1024, conditional=True):
        self.latents = [torch.randn(latent_dim) for _ in range(n)]
        self.pcs = [torch.randn(pc_size, 3) for _ in range(n)] if conditional else None

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {
            "point_cloud": self.pcs[idx] if self.pcs is not None else False,
            "latent": self.latents[idx],
        }


def _load_specs(config_dir):
    path = os.path.join(ROOT, config_dir, "specs.json")
    with open(path) as f:
        return json.load(f)


# ---- Instantiation -----

def test_combined_model_uses_point_mae():
    """CombinedModel with flow matching + point_mae config uses PointMAEEncoder."""
    specs = _load_specs("config/stage2_fm_cond")
    model = CombinedModel(specs).to(DEVICE)
    assert isinstance(model.diffusion_model, FlowMatchingModel)
    assert isinstance(model.diffusion_model.model.pointnet, PointMAEEncoder)


# ---- Forward pass -----

def test_forward_pass_fm_point_mae():
    specs = _load_specs("config/stage2_fm_cond")
    model = CombinedModel(specs).to(DEVICE).train()
    batch = {
        "point_cloud": torch.randn(2, 1024, 3, device=DEVICE),
        "latent": torch.randn(2, 768, device=DEVICE),
    }
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert loss.dim() == 0
    assert torch.isfinite(loss)


# ---- Training loop -----

def test_training_loop_fm_point_mae():
    specs = _load_specs("config/stage2_fm_cond")
    specs["num_epochs"] = 3

    model = CombinedModel(specs)
    dataset = SyntheticModulationDataset(n=16, conditional=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=True)

    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=32,
        max_epochs=3, enable_checkpointing=False,
        enable_progress_bar=False, logger=False,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

    assert trainer.current_epoch == 3
    logged = trainer.callback_metrics
    assert "total" in logged
    assert torch.isfinite(logged["total"])


# ---- Generation -----

def test_generation_from_pc_fm_point_mae():
    specs = _load_specs("config/stage2_fm_cond")
    model = CombinedModel(specs).to(DEVICE).eval()

    pc = torch.randn(1, 1024, 3, device=DEVICE)
    samp = model.diffusion_model.generate_from_pc(pc, batch=2, perturb_pc=False)
    assert samp.shape == (2, 768)
    assert torch.isfinite(samp).all()


# ---- Backward compat: ConvPointnet path still works -----

def test_conv_pointnet_path_still_works():
    specs = _load_specs("config/stage2_diff_cond")
    model = CombinedModel(specs).to(DEVICE).train()
    batch = {
        "point_cloud": torch.randn(2, 1024, 3, device=DEVICE),
        "latent": torch.randn(2, 768, device=DEVICE),
    }
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert torch.isfinite(loss)


# ---- Flow Matching unconditional (no encoder) still works -----

def test_unconditional_fm_still_works():
    specs = _load_specs("config/stage2_fm_uncond")
    model = CombinedModel(specs).to(DEVICE).train()
    batch = {
        "point_cloud": False,
        "latent": torch.randn(2, 768, device=DEVICE),
    }
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
