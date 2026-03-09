"""Point-MAE encoder for conditioning in DiffusionNet.

Self-contained implementation of the Point-MAE encoder (ECCV 2022)
that takes raw point clouds and produces per-patch token features
for cross-attention conditioning.  Uses pure-PyTorch FPS and KNN
to avoid the knn_cuda dependency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import DropPath, trunc_normal_


# ---- pure-PyTorch FPS & KNN ------------------------------------------------

def _fps_torch(xyz, npoint):
    """Farthest point sampling in pure PyTorch.

    Args:
        xyz: (B, N, 3)
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) indices
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # B 1 3
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)          # B N
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def _knn_torch(xyz, centers, k):
    """K-nearest neighbors in pure PyTorch.

    Args:
        xyz: (B, N, 3)
        centers: (B, G, 3)
        k: int

    Returns:
        idx: (B, G, k) indices into xyz
    """
    dist = torch.cdist(centers, xyz)  # B G N
    _, idx = dist.topk(k, dim=-1, largest=False)
    return idx


# ---- Point-MAE modules -----------------------------------------------------

class PatchEmbedding(nn.Module):
    """Mini-PointNet that embeds each point patch into a feature vector."""

    def __init__(self, encoder_dim=384):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_dim, 1),
        )

    def forward(self, point_groups):
        """
        Args:
            point_groups: (B, G, S, 3) - grouped point patches
        Returns:
            (B, G, C) - per-patch features
        """
        B, G, S, _ = point_groups.shape
        x = point_groups.reshape(B * G, S, 3)
        feat = self.first_conv(x.transpose(2, 1))           # BG 256 S
        feat_global = feat.max(dim=2, keepdim=True)[0]       # BG 256 1
        feat = torch.cat([feat_global.expand(-1, -1, S), feat], dim=1)  # BG 512 S
        feat = self.second_conv(feat)                        # BG C S
        feat_global = feat.max(dim=2)[0]                     # BG C
        return feat_global.reshape(B, G, -1)


class MAETransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0,
                 attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path(h)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PointMAEEncoder(nn.Module):
    """Point-MAE encoder wrapper for conditioning in Diffusion-SDF.

    Takes raw point clouds and produces per-patch token features
    compatible with cross-attention in CausalTransformer.

    Args:
        num_group:  number of point patches (default 64)
        group_size: points per patch (default 32)
        encoder_dim: mini-PointNet output dim (default 384)
        trans_dim:  transformer hidden dim (default 384)
        depth:      number of transformer blocks (default 12)
        num_heads:  attention heads (default 6)
        drop_path_rate: stochastic depth (default 0.1)
        out_dim:    output feature dim per patch (default 128,
                    matching point_feature_dim in DiffusionNet)
        freeze:     whether to freeze pretrained weights (default False)
    """

    def __init__(
        self,
        num_group=64,
        group_size=32,
        encoder_dim=384,
        trans_dim=384,
        depth=12,
        num_heads=6,
        drop_path_rate=0.1,
        out_dim=128,
        freeze=False,
    ):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.trans_dim = trans_dim
        self.out_dim = out_dim

        self.patch_embed = PatchEmbedding(encoder_dim=encoder_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        if encoder_dim != trans_dim:
            self.input_proj = nn.Linear(encoder_dim, trans_dim)
        else:
            self.input_proj = nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MAETransformerBlock(
                dim=trans_dim, num_heads=num_heads,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(trans_dim)
        self.proj_out = nn.Linear(trans_dim, out_dim)

        self.apply(self._init_weights)
        self._freeze = freeze

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_pretrained(self, ckpt_path):
        """Load pretrained Point-MAE encoder weights.

        Handles the official Point-MAE checkpoint format where keys
        are prefixed with 'MAE_encoder.' or 'base_model.'.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("base_model", ckpt.get("state_dict", ckpt))

        new_state = {}
        for k, v in state.items():
            k = k.replace("module.", "")
            if k.startswith("MAE_encoder."):
                k = k[len("MAE_encoder."):]
            elif k.startswith("base_model."):
                k = k[len("base_model."):]
            new_state[k] = v

        mapped = {}
        for k, v in new_state.items():
            if k.startswith("encoder."):
                mapped["patch_embed." + k[len("encoder."):]] = v
            elif k.startswith("blocks."):
                mapped[k] = v
            elif k.startswith("pos_embed."):
                mapped[k] = v
            elif k.startswith("norm."):
                mapped[k] = v

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        return missing, unexpected

    def _group_points(self, xyz):
        """FPS + KNN grouping.

        Args:
            xyz: (B, N, 3)
        Returns:
            neighborhoods: (B, G, S, 3) centered patches
            centers: (B, G, 3)
        """
        B, N, _ = xyz.shape
        G = min(self.num_group, N)
        S = min(self.group_size, N)

        fps_idx = _fps_torch(xyz, G)                         # B G
        batch_idx = torch.arange(B, device=xyz.device).unsqueeze(1)
        centers = xyz[batch_idx, fps_idx]                     # B G 3

        knn_idx = _knn_torch(xyz, centers, S)                 # B G S
        batch_idx_full = torch.arange(B, device=xyz.device).view(B, 1, 1)
        neighborhoods = xyz[batch_idx_full, knn_idx]          # B G S 3
        neighborhoods = neighborhoods - centers.unsqueeze(2)  # center-normalize

        return neighborhoods, centers

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) raw point cloud
        Returns:
            (B, G, out_dim) per-patch features
        """
        if self._freeze:
            with torch.no_grad():
                return self._forward_impl(xyz)
        return self._forward_impl(xyz)

    def _forward_impl(self, xyz):
        neighborhoods, centers = self._group_points(xyz)       # B G S 3, B G 3
        tokens = self.patch_embed(neighborhoods)               # B G encoder_dim
        tokens = self.input_proj(tokens)                       # B G trans_dim
        pos = self.pos_embed(centers)                          # B G trans_dim

        x = tokens + pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return self.proj_out(x)                                # B G out_dim
