"""CHECKPOINT 4: Point-MAE encoder unit tests.

Verifies forward pass, output shapes, FPS/KNN grouping, and that
the output is compatible with cross-attention in DiffusionNet.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from models.archs.encoders.point_mae import (
    PointMAEEncoder, PatchEmbedding, _fps_torch, _knn_torch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B = 4
N = 1024
NUM_GROUP = 64
GROUP_SIZE = 32
OUT_DIM = 128
TRANS_DIM = 384


def _make_encoder(**kwargs):
    defaults = dict(
        num_group=NUM_GROUP, group_size=GROUP_SIZE,
        encoder_dim=TRANS_DIM, trans_dim=TRANS_DIM,
        depth=2, num_heads=6, out_dim=OUT_DIM,
    )
    defaults.update(kwargs)
    return PointMAEEncoder(**defaults).to(DEVICE)


# ---- FPS & KNN -----

def test_fps_returns_correct_count():
    xyz = torch.randn(B, N, 3, device=DEVICE)
    idx = _fps_torch(xyz, NUM_GROUP)
    assert idx.shape == (B, NUM_GROUP)
    assert idx.max() < N
    assert idx.min() >= 0


def test_knn_returns_correct_shape():
    xyz = torch.randn(B, N, 3, device=DEVICE)
    centers = xyz[:, :NUM_GROUP, :]
    idx = _knn_torch(xyz, centers, GROUP_SIZE)
    assert idx.shape == (B, NUM_GROUP, GROUP_SIZE)
    assert idx.max() < N


# ---- Patch Embedding -----

def test_patch_embedding_shape():
    pe = PatchEmbedding(encoder_dim=TRANS_DIM).to(DEVICE)
    patches = torch.randn(B, NUM_GROUP, GROUP_SIZE, 3, device=DEVICE)
    out = pe(patches)
    assert out.shape == (B, NUM_GROUP, TRANS_DIM)
    assert torch.isfinite(out).all()


# ---- Full encoder -----

def test_encoder_forward_shape():
    enc = _make_encoder()
    xyz = torch.randn(B, N, 3, device=DEVICE)
    out = enc(xyz)
    assert out.shape == (B, NUM_GROUP, OUT_DIM)
    assert torch.isfinite(out).all()


def test_encoder_output_dim_matches_point_feature_dim():
    """Output dim must match point_feature_dim used in DiffusionNet cross-attention."""
    for pf_dim in [64, 128, 256]:
        enc = _make_encoder(out_dim=pf_dim, depth=1)
        xyz = torch.randn(2, N, 3, device=DEVICE)
        out = enc(xyz)
        assert out.shape[-1] == pf_dim


def test_encoder_varying_input_size():
    """Should handle different point cloud sizes as long as >= num_group."""
    enc = _make_encoder(depth=1)
    for n_pts in [256, 512, 2048]:
        xyz = torch.randn(2, n_pts, 3, device=DEVICE)
        out = enc(xyz)
        assert out.shape == (2, NUM_GROUP, OUT_DIM)


def test_encoder_gradients_flow():
    enc = _make_encoder(depth=1)
    xyz = torch.randn(2, N, 3, device=DEVICE, requires_grad=True)
    out = enc(xyz)
    loss = out.sum()
    loss.backward()
    assert xyz.grad is not None
    assert torch.isfinite(xyz.grad).all()


def test_encoder_freeze_mode():
    enc = _make_encoder(depth=1, freeze=True)
    xyz = torch.randn(2, N, 3, device=DEVICE)
    out = enc(xyz)
    assert out.shape == (2, NUM_GROUP, OUT_DIM)


def test_encoder_with_cross_attention():
    """Simulate cross-attention: DiffusionNet query attends to Point-MAE tokens."""
    enc = _make_encoder(depth=1)
    xyz = torch.randn(B, N, 3, device=DEVICE)
    tokens = enc(xyz)  # B G out_dim

    query_dim = 768
    cross_attn = torch.nn.MultiheadAttention(
        embed_dim=query_dim, num_heads=8, kdim=OUT_DIM, vdim=OUT_DIM,
        batch_first=True,
    ).to(DEVICE)

    queries = torch.randn(B, 1, query_dim, device=DEVICE)
    out, _ = cross_attn(queries, tokens, tokens)
    assert out.shape == (B, 1, query_dim)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
