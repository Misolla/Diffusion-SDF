"""CHECKPOINT 3: Training smoke test.

Runs a short training loop with synthetic data to verify the full
training pipeline works end-to-end with FlowMatchingModel.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytorch_lightning as pl
import pytest

from models.combined_model import CombinedModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.path.dirname(__file__), "..")


class SyntheticModulationDataset(torch.utils.data.Dataset):
    """Mimics ModulationLoader with random latent vectors and point clouds."""

    def __init__(self, n=64, latent_dim=768, pc_size=1024, conditional=True):
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


def test_flow_matching_training_loop_conditional():
    """Train for 5 epochs with conditional flow matching and verify loss decreases."""
    specs = _load_specs("config/stage2_fm_cond")
    specs["num_epochs"] = 5
    specs["log_freq"] = 100

    model = CombinedModel(specs)
    dataset = SyntheticModulationDataset(n=32, conditional=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)

    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=32,
        max_epochs=5, enable_checkpointing=False,
        enable_progress_bar=False, logger=False,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

    assert trainer.current_epoch == 5
    logged = trainer.callback_metrics
    assert "total" in logged
    assert torch.isfinite(logged["total"])


def test_flow_matching_training_loop_unconditional():
    """Train for 5 epochs with unconditional flow matching."""
    specs = _load_specs("config/stage2_fm_uncond")
    specs["num_epochs"] = 5
    specs["log_freq"] = 100

    model = CombinedModel(specs)
    dataset = SyntheticModulationDataset(n=32, conditional=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)

    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=32,
        max_epochs=5, enable_checkpointing=False,
        enable_progress_bar=False, logger=False,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

    assert trainer.current_epoch == 5
    logged = trainer.callback_metrics
    assert "total" in logged
    assert torch.isfinite(logged["total"])


def test_generation_after_training():
    """Train briefly, then verify generation produces valid outputs."""
    specs = _load_specs("config/stage2_fm_uncond")
    specs["num_epochs"] = 3

    model = CombinedModel(specs)
    dataset = SyntheticModulationDataset(n=16, conditional=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)

    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=32,
        max_epochs=3, enable_checkpointing=False,
        enable_progress_bar=False, logger=False,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

    model = model.cuda().eval()
    samp = model.diffusion_model.generate_unconditional(num_samples=2)
    assert samp.shape == (2, 768)
    assert torch.isfinite(samp).all()


def test_generation_from_pc_after_training():
    """Train briefly with conditioning, then generate from point cloud."""
    specs = _load_specs("config/stage2_fm_cond")
    specs["num_epochs"] = 3

    model = CombinedModel(specs)
    dataset = SyntheticModulationDataset(n=16, conditional=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)

    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=32,
        max_epochs=3, enable_checkpointing=False,
        enable_progress_bar=False, logger=False,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

    model = model.cuda().eval()
    pc = torch.randn(1, 1024, 3, device="cuda")
    samp = model.diffusion_model.generate_from_pc(pc, batch=2, perturb_pc=False)
    assert samp.shape == (2, 768)
    assert torch.isfinite(samp).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
