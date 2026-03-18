"""Generate meshes from DDPM and Flow Matching checkpoints and render comparison images.

Supports both Stage 3 (combined) and Stage 2 (diffusion-only + S1) checkpoints.
Specs are auto-detected from each checkpoint's parent directory specs.json.

Usage:
  # Stage 3 checkpoints (SDF+VAE+Diffusion in one):
  python scripts/render_comparison.py \
      --ddpm-ckpt config/pipeclean_s3_ddpm/last.ckpt \
      --fm-ckpt config/pipeclean_s3_fm/last.ckpt

  # Stage 2 checkpoints (need S1 for SDF+VAE):
  python scripts/render_comparison.py \
      --ddpm-ckpt config/pipeclean_s2_ddpm/last.ckpt \
      --fm-ckpt config/pipeclean_s2_fm/last.ckpt \
      --s1-ckpt config/pipeclean_s1/last.ckpt

  # With ground truth ShapeNet meshes:
  python scripts/render_comparison.py \
      --ddpm-ckpt config/pipeclean_s3_ddpm/last.ckpt \
      --fm-ckpt config/pipeclean_s3_fm/last.ckpt \
      --gt-dir gt_meshes

  # Custom output and sample count:
  python scripts/render_comparison.py \
      --ddpm-ckpt ... --fm-ckpt ... \
      --output renders/ --num-shapes 5 --samples-per-shape 3
"""
import os
import sys
import json
import time
import warnings
import argparse

os.environ["PYOPENGL_PLATFORM"] = "egl"

import torch
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.combined_model import CombinedModel
from utils import mesh as mesh_utils


def read_specs(ckpt_path):
    """Read specs.json from the checkpoint's parent directory."""
    config_dir = os.path.dirname(ckpt_path)
    specs_path = os.path.join(config_dir, "specs.json")
    if not os.path.isfile(specs_path):
        raise FileNotFoundError(f"No specs.json found at {specs_path}")
    return json.load(open(specs_path))


def build_combined_specs(s2_specs, s1_specs):
    """Merge S1 (modulation) specs with S2 (diffusion) specs to create combined-task specs."""
    merged = dict(s1_specs)
    merged["training_task"] = "combined"
    for key in ("generative_model", "flow_matching_specs", "diffusion_specs",
                "diffusion_model_specs", "diff_lr"):
        if key in s2_specs:
            merged[key] = s2_specs[key]
    return merged


def load_model(ckpt_path, s1_ckpt=None):
    """Load a CombinedModel, auto-detecting specs from the checkpoint directory.

    Stage 3 checkpoints (training_task=combined) are loaded directly.
    Stage 2 checkpoints (training_task=diffusion) require s1_ckpt for SDF+VAE weights.
    """
    specs = read_specs(ckpt_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if specs["training_task"] == "combined":
            model = CombinedModel.load_from_checkpoint(ckpt_path, specs=specs, strict=False)
        elif specs["training_task"] == "diffusion":
            if s1_ckpt is None:
                raise ValueError(
                    f"Checkpoint at {ckpt_path} is Stage 2 (diffusion-only). "
                    "Provide --s1-ckpt for the SDF+VAE weights."
                )
            s1_specs = read_specs(s1_ckpt)
            combined_specs = build_combined_specs(specs, s1_specs)
            model = CombinedModel.load_from_checkpoint(s1_ckpt, specs=combined_specs, strict=False)

            s2_state = torch.load(ckpt_path, map_location="cpu")
            diff_state = {
                k.replace("diffusion_model.", ""): v
                for k, v in s2_state["state_dict"].items()
                if k.startswith("diffusion_model.")
            }
            model.diffusion_model.load_state_dict(diff_state)
        else:
            raise ValueError(f"Unsupported training_task: {specs['training_task']}")

    return model.cuda().eval()


def load_point_cloud(csv_path, pc_size=1024):
    """Load surface points (SDF==0) from an SDF CSV file."""
    data = np.loadtxt(csv_path, delimiter=",")
    surface = data[np.abs(data[:, -1]) < 1e-6][:, :3]
    if len(surface) == 0:
        surface = data[np.argsort(np.abs(data[:, -1]))][:pc_size, :3]
    if surface.shape[0] < pc_size:
        idx = np.random.choice(surface.shape[0], pc_size, replace=True)
    else:
        idx = np.random.choice(surface.shape[0], pc_size, replace=False)
    return torch.from_numpy(surface[idx]).float()


def collect_instances(split_path, data_source, max_count, seed=None):
    """Collect instance (class, id, csv_path) tuples from a split file."""
    split = json.load(open(split_path))
    instances = []
    for dataset in split:
        for cls in split[dataset]:
            for inst in split[dataset][cls]:
                csv = os.path.join(data_source, dataset, cls, inst, "sdf_data.csv")
                if os.path.isfile(csv):
                    instances.append((cls, inst, csv))
    if seed is not None:
        import random
        random.Random(seed).shuffle(instances)
    return instances[:max_count]


@torch.no_grad()
def generate_meshes(model, point_cloud, output_dir, num_samples=1, mesh_res=128):
    """Generate .ply meshes from a point cloud.

    Returns (list_of_ply_paths, diffusion_time_seconds, total_time_seconds).
    diffusion_time covers only the latent sampling; total_time includes decoding + marching cubes.
    """
    pc = point_cloud.unsqueeze(0).cuda()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    samples = model.diffusion_model.generate_from_pc(
        pc, batch=num_samples, return_pc=False, perturb_pc=False,
    )
    torch.cuda.synchronize()
    t_diffusion = time.perf_counter() - t0

    plane_features = model.vae_model.decode(samples)

    paths = []
    for i in range(len(plane_features)):
        pf = plane_features[i].unsqueeze(0)
        stem = os.path.join(output_dir, f"sample_{i}")
        mesh_utils.create_mesh(
            model.sdf_model, pf, stem,
            N=mesh_res, max_batch=2**18, from_plane_features=True,
        )
        ply = stem + ".ply"
        if os.path.exists(ply):
            paths.append(ply)

    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0
    return paths, t_diffusion, t_total


def _rotate_x_90(pts):
    """Rotate points +90 degrees around X axis: (x,y,z) -> (x,-z,y)."""
    rotated = np.empty_like(pts)
    rotated[:, 0] = pts[:, 0]
    rotated[:, 1] = -pts[:, 2]
    rotated[:, 2] = pts[:, 1]
    return rotated


def render_mesh(ply_path, elev=25, azim=135):
    """Render a .ply mesh to a numpy RGB image via matplotlib."""
    try:
        tm = trimesh.load(ply_path)
    except Exception as e:
        print(f"  Warning: could not load {ply_path}: {e}")
        return None

    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    verts = _rotate_x_90(np.asarray(tm.vertices))
    faces = np.asarray(tm.faces)

    if len(faces) > 50000:
        idx = np.random.choice(len(faces), 50000, replace=False)
        faces = faces[idx]

    poly = Poly3DCollection(verts[faces], alpha=0.85, linewidths=0.05, edgecolors="#888888")
    poly.set_facecolor([0.55, 0.70, 0.85])
    ax.add_collection3d(poly)

    center = verts.mean(axis=0)
    extent = max(verts.max(axis=0) - verts.min(axis=0)) * 0.6
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(c - extent, c + extent)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def render_pointcloud(pc_np, elev=25, azim=135):
    """Render a point cloud to a numpy RGB image via matplotlib."""
    pc_np = _rotate_x_90(pc_np)

    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
        s=0.4, c=pc_np[:, 2], cmap="coolwarm", alpha=0.6,
    )

    center = pc_np.mean(axis=0)
    extent = max(pc_np.max(axis=0) - pc_np.min(axis=0)) * 0.6
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(c - extent, c + extent)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def find_gt_mesh(gt_dir, instance_id):
    """Look for a ground truth mesh in gt_dir/{instance_id}/."""
    if gt_dir is None:
        return None
    candidates = [
        os.path.join(gt_dir, instance_id, "model_normalized.ply"),
        os.path.join(gt_dir, instance_id, "model_normalized.obj"),
        os.path.join(gt_dir, instance_id, "model.ply"),
        os.path.join(gt_dir, instance_id, "model.obj"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _fmt_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.1f}s"


def create_comparison_grid(pc_img, gt_img, ddpm_imgs, fm_imgs, output_path, instance_id,
                           ddpm_times=None, fm_times=None):
    """Build a comparison grid: GT | Input PC | DDPM samples | FM samples (two rows).

    ddpm_times / fm_times: optional (diffusion_sec, total_sec) tuples.
    """
    has_gt = gt_img is not None
    left_cols = 2 if has_gt else 1
    sample_cols = max(len(ddpm_imgs), len(fm_imgs))
    n_cols = left_cols + sample_cols

    fig = plt.figure(figsize=(4 * n_cols, 9.0), dpi=150)
    gs = GridSpec(2, n_cols, figure=fig, wspace=0.03, hspace=0.12)

    col = 0
    if has_gt:
        ax_gt = fig.add_subplot(gs[:, col])
        ax_gt.imshow(gt_img)
        ax_gt.set_title("Ground Truth", fontsize=11, fontweight="bold", color="#2e7d32")
        ax_gt.axis("off")
        col += 1

    ax_pc = fig.add_subplot(gs[:, col])
    ax_pc.imshow(pc_img)
    ax_pc.set_title("Input Point Cloud", fontsize=11, fontweight="bold")
    ax_pc.axis("off")
    col += 1

    for i, img in enumerate(ddpm_imgs):
        ax = fig.add_subplot(gs[0, col + i])
        if img is not None:
            ax.imshow(img)
        ax.set_title(f"FM_MAE sample {i}", fontsize=10)
        ax.axis("off")

    if ddpm_imgs:
        x_center = (col + col + len(ddpm_imgs) - 1) / 2 / n_cols
        fig.text(x_center, 0.95, "Flow Matching (Point-MAE)", ha="center", fontsize=13,
                 fontweight="bold", transform=fig.transFigure)
        # fig.text(x_center, 0.49, "Flow Matching (Point-MAE)", ha="center", fontsize=13,
        #          fontweight="bold", transform=fig.transFigure)
        if ddpm_times:
            d_sec, t_sec = ddpm_times
            fig.text(x_center, 0.91, f"sampling: {_fmt_time(d_sec)}  |  total: {_fmt_time(t_sec)}",
                     ha="center", fontsize=10, color="#555555", transform=fig.transFigure)

    for i, img in enumerate(fm_imgs):
        ax = fig.add_subplot(gs[1, col + i])
        if img is not None:
            ax.imshow(img)
        ax.set_title(f"FM sample {i}", fontsize=10)
        ax.axis("off")

    if fm_imgs:
        x_center = (col + col + len(fm_imgs) - 1) / 2 / n_cols
        # fig.text(x_center, 0.49, "Flow Matching (Point-MAE)", ha="center", fontsize=13,
        #          fontweight="bold", transform=fig.transFigure)
        fig.text(x_center, 0.49, "Flow Matching (ConvPointnet)", ha="center", fontsize=13,
                 fontweight="bold", transform=fig.transFigure)
        if fm_times:
            d_sec, t_sec = fm_times
            fig.text(x_center, 0.45, f"sampling: {_fmt_time(d_sec)}  |  total: {_fmt_time(t_sec)}",
                     ha="center", fontsize=10, color="#555555", transform=fig.transFigure)

    fig.suptitle(f"Instance: {instance_id}", fontsize=14, y=1.01)
    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Render DDPM vs Flow Matching mesh comparison images",
    )
    parser.add_argument("--ddpm-ckpt", required=True, help="Path to DDPM checkpoint")
    parser.add_argument("--fm-ckpt", required=True, help="Path to FM checkpoint")
    parser.add_argument("--s1-ckpt", default=None,
                        help="Stage 1 (SDF-VAE) checkpoint; required when using Stage 2 checkpoints")
    parser.add_argument("--gt-dir", default=None,
                        help="Directory with GT meshes: gt_dir/{instance_id}/model_normalized.ply")
    parser.add_argument("--split", default=None,
                        help="Override split file (default: from specs.json)")
    parser.add_argument("--data-source", default=None,
                        help="Override data source directory (default: from specs.json)")
    parser.add_argument("--output", default="renders")
    parser.add_argument("--num-shapes", type=int, default=5)
    parser.add_argument("--samples-per-shape", type=int, default=3)
    parser.add_argument("--mesh-res", type=int, default=128,
                        help="Marching cubes grid resolution (default: 128)")
    parser.add_argument("--fm-steps", type=int, default=None,
                        help="Override FM sampling steps (default: use model config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for shuffling which instances to render")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ddpm_specs = read_specs(args.ddpm_ckpt)
    split_file = args.split or ddpm_specs.get("TestSplit", ddpm_specs.get("TrainSplit"))
    data_source = args.data_source or ddpm_specs.get("DataSource", "data")

    instances = collect_instances(split_file, data_source, args.num_shapes, seed=args.seed)
    if not instances:
        print("ERROR: no instances found. Check --split and --data-source.")
        sys.exit(1)
    print(f"Generating meshes for {len(instances)} instances")

    print(f"\nLoading DDPM model from {args.ddpm_ckpt} ...")
    ddpm_model = load_model(args.ddpm_ckpt, s1_ckpt=args.s1_ckpt)

    print(f"Loading FM model from {args.fm_ckpt} ...")
    fm_model = load_model(args.fm_ckpt, s1_ckpt=args.s1_ckpt)

    if args.fm_steps is not None:
        fm_model.diffusion_model.num_sample_steps = args.fm_steps
        print(f"  Overriding FM sampling steps to {args.fm_steps}")

    for idx, (cls, inst, csv_path) in enumerate(instances, 1):
        print(f"\n[{idx}/{len(instances)}] {cls}/{inst}")
        inst_dir = os.path.join(args.output, inst)
        os.makedirs(inst_dir, exist_ok=True)

        pc = load_point_cloud(csv_path, pc_size=1024)
        pc_img = render_pointcloud(pc.numpy())
        plt.imsave(os.path.join(inst_dir, "input_pc.png"), pc_img)
        print("  Saved input_pc.png")

        gt_img = None
        gt_path = find_gt_mesh(args.gt_dir, inst)
        if gt_path:
            print(f"  Rendering GT mesh: {gt_path}")
            gt_img = render_mesh(gt_path)
            if gt_img is not None:
                plt.imsave(os.path.join(inst_dir, "gt.png"), gt_img)
        elif args.gt_dir:
            print(f"  Warning: no GT mesh found for {inst}")

        print("  Generating DDPM meshes ...")
        ddpm_dir = os.path.join(inst_dir, "ddpm")
        os.makedirs(ddpm_dir, exist_ok=True)
        ddpm_plys, ddpm_diff_t, ddpm_total_t = generate_meshes(
            ddpm_model, pc, ddpm_dir,
            num_samples=args.samples_per_shape, mesh_res=args.mesh_res,
        )
        print(f"  DDPM: {len(ddpm_plys)} meshes  (sampling: {ddpm_diff_t:.2f}s, total: {ddpm_total_t:.2f}s)")

        print("  Generating FM meshes ...")
        fm_dir = os.path.join(inst_dir, "fm")
        os.makedirs(fm_dir, exist_ok=True)
        fm_plys, fm_diff_t, fm_total_t = generate_meshes(
            fm_model, pc, fm_dir,
            num_samples=args.samples_per_shape, mesh_res=args.mesh_res,
        )
        print(f"  FM: {len(fm_plys)} meshes  (sampling: {fm_diff_t:.2f}s, total: {fm_total_t:.2f}s)")

        ddpm_imgs = [render_mesh(p) for p in ddpm_plys]
        fm_imgs = [render_mesh(p) for p in fm_plys]

        for i, img in enumerate(ddpm_imgs):
            if img is not None:
                plt.imsave(os.path.join(inst_dir, f"ddpm_{i}.png"), img)
        for i, img in enumerate(fm_imgs):
            if img is not None:
                plt.imsave(os.path.join(inst_dir, f"fm_{i}.png"), img)

        create_comparison_grid(
            pc_img, gt_img, ddpm_imgs, fm_imgs,
            os.path.join(inst_dir, "comparison.png"), inst,
            ddpm_times=(ddpm_diff_t, ddpm_total_t),
            fm_times=(fm_diff_t, fm_total_t),
        )
        print(f"  -> {inst_dir}/comparison.png")

    print(f"\nDone! All renders saved to {args.output}/")


if __name__ == "__main__":
    main()
