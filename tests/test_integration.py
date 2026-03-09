"""CHECKPOINT 2: Integration tests.

Verifies CombinedModel instantiates correctly with both flow matching
and DDPM configs, and that training steps produce finite losses.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from models.combined_model import CombinedModel
from models.flow_matching import FlowMatchingModel
from models.diffusion import DiffusionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_specs(config_dir):
    path = os.path.join(ROOT, config_dir, "specs.json")
    with open(path) as f:
        return json.load(f)


# ----- model instantiation -----

def test_combined_model_flow_matching():
    specs = _load_specs("config/stage2_fm_cond")
    model = CombinedModel(specs).to(DEVICE)
    assert isinstance(model.diffusion_model, FlowMatchingModel)


def test_combined_model_flow_matching_uncond():
    specs = _load_specs("config/stage2_fm_uncond")
    model = CombinedModel(specs).to(DEVICE)
    assert isinstance(model.diffusion_model, FlowMatchingModel)


def test_combined_model_ddpm_backward_compat():
    specs = _load_specs("config/stage2_diff_cond")
    model = CombinedModel(specs).to(DEVICE)
    assert isinstance(model.diffusion_model, DiffusionModel)


def test_combined_model_ddpm_uncond_backward_compat():
    specs = _load_specs("config/stage2_diff_uncond")
    model = CombinedModel(specs).to(DEVICE)
    assert isinstance(model.diffusion_model, DiffusionModel)


# ----- training step (flow matching, diffusion-only mode) -----

def test_training_step_flow_matching_cond():
    specs = _load_specs("config/stage2_fm_cond")
    model = CombinedModel(specs).to(DEVICE)
    model.train()
    batch = {
        "point_cloud": torch.randn(2, 1024, 3, device=DEVICE),
        "latent": torch.randn(2, 768, device=DEVICE),
    }
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_training_step_flow_matching_uncond():
    specs = _load_specs("config/stage2_fm_uncond")
    model = CombinedModel(specs).to(DEVICE)
    model.train()
    batch = {
        "point_cloud": False,
        "latent": torch.randn(2, 768, device=DEVICE),
    }
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_training_step_ddpm_cond():
    specs = _load_specs("config/stage2_diff_cond")
    model = CombinedModel(specs).to(DEVICE)
    model.train()
    batch = {
        "point_cloud": torch.randn(2, 1024, 3, device=DEVICE),
        "latent": torch.randn(2, 768, device=DEVICE),
    }
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert loss.dim() == 0
    assert torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
