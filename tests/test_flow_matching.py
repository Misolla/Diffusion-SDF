"""CHECKPOINT 1: FlowMatchingModel unit tests.

Validates CFM interpolation, velocity targets, forward loss, ODE sampling,
classifier-free guidance, and the diffusion_model_from_latent wrapper.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from models.flow_matching import FlowMatchingModel
from models.archs.diffusion_arch import DiffusionNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, D = 4, 768
DEPTH = 2
PC_N = 128
PF_DIM = 128


def _make_model(cond=False, cross_attn=False, cond_dropout=False):
    net = DiffusionNet(
        dim=D, depth=DEPTH, num_timesteps=None, cond=cond,
        cross_attn=cross_attn, cond_dropout=cond_dropout,
        point_feature_dim=PF_DIM,
    ).to(DEVICE)
    fm = FlowMatchingModel(
        model=net, num_sample_steps=5, solver="euler",
        sample_pc_size=PC_N, perturb_pc=None,
    ).to(DEVICE)
    return fm


# ----- interpolation & velocity target -----

def test_interpolation_shapes():
    z_1 = torch.randn(B, D, device=DEVICE)
    eps = torch.randn(B, D, device=DEVICE)
    t = torch.rand(B, device=DEVICE)
    fm = _make_model()
    z_t = fm.interpolate(z_1, t, eps)
    assert z_t.shape == (B, D)
    assert torch.isfinite(z_t).all()


def test_velocity_target():
    z_1 = torch.randn(B, D, device=DEVICE)
    eps = torch.randn(B, D, device=DEVICE)
    u_t = z_1 - eps
    assert u_t.shape == (B, D)
    assert torch.isfinite(u_t).all()


# ----- forward loss (unconditional) -----

def test_forward_loss():
    fm = _make_model(cond=False)
    z_1 = torch.randn(B, D, device=DEVICE)
    t = torch.rand(B, device=DEVICE)
    loss, unreduced = fm(z_1, t)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert unreduced.shape == (B,)


def test_forward_loss_ret_pred_x():
    fm = _make_model(cond=False)
    z_1 = torch.randn(B, D, device=DEVICE)
    t = torch.rand(B, device=DEVICE)
    loss, z_t, target, pred, unreduced = fm(z_1, t, ret_pred_x=True)
    assert loss.dim() == 0
    assert z_t.shape == (B, D)
    assert target.shape == (B, D)
    assert pred.shape == (B, D)
    assert torch.isfinite(loss)


# ----- forward loss (conditional) -----

def test_forward_loss_conditional():
    fm = _make_model(cond=True, cross_attn=True)
    z_1 = torch.randn(B, D, device=DEVICE)
    t = torch.rand(B, device=DEVICE)
    pc = torch.randn(B, PC_N, 3, device=DEVICE)
    loss, unreduced = fm(z_1, t, cond=pc)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


# ----- ODE sampling -----

def test_ode_sample_shapes():
    fm = _make_model(cond=False)
    samp, traj = fm.sample(dim=D, batch_size=2, num_steps=3)
    assert samp.shape == (2, D)
    assert torch.isfinite(samp).all()
    assert len(traj) == 3


def test_ode_sample_conditional():
    fm = _make_model(cond=True, cross_attn=True)
    pc = torch.randn(2, PC_N, 3, device=DEVICE)
    samp, traj = fm.sample(dim=D, batch_size=2, num_steps=3, cond=pc)
    assert samp.shape == (2, D)
    assert torch.isfinite(samp).all()


def test_ode_sample_midpoint():
    fm = _make_model(cond=False)
    fm.solver = "midpoint"
    samp, traj = fm.sample(dim=D, batch_size=2, num_steps=3)
    assert samp.shape == (2, D)
    assert torch.isfinite(samp).all()


# ----- classifier-free guidance -----

def test_classifier_free_guidance():
    fm = _make_model(cond=True, cross_attn=True, cond_dropout=True)
    pc = torch.randn(2, PC_N, 3, device=DEVICE)
    samp, _ = fm.sample_cfg(dim=D, batch_size=2, num_steps=3,
                            cond=pc, guidance_scale=1.0)
    assert samp.shape == (2, D)
    assert torch.isfinite(samp).all()


# ----- wrapper -----

def test_diffusion_model_from_latent_wrapper():
    fm = _make_model(cond=True, cross_attn=True)
    z_start = torch.randn(B, D, device=DEVICE)
    pc = torch.randn(B, 1024, 3, device=DEVICE)
    loss, loss_low, loss_high, pred, perturbed = fm.diffusion_model_from_latent(z_start, cond=pc)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert pred.shape == (B, D)


def test_generate_unconditional():
    fm = _make_model(cond=False)
    samp = fm.generate_unconditional(num_samples=2)
    assert samp.shape == (2, D)
    assert torch.isfinite(samp).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
