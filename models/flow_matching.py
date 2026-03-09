import torch
from torch import nn
import torch.nn.functional as F

from tqdm.auto import tqdm
import open3d as o3d

from diff_utils.helpers import default, perturb_point_cloud, sample_pc


class FlowMatchingModel(nn.Module):
    """Conditional Flow Matching wrapper around a denoising network.

    Replaces DDPM by learning a velocity field v_theta that transports
    noise eps ~ N(0,I) to data z_1 along the linear path
        z_t = (1 - t) * eps + t * z_1,   t in [0, 1].
    The target velocity is u_t = z_1 - eps.

    At inference the learned field is integrated from t=0 to t=1 with a
    fixed-step ODE solver (Euler or midpoint).
    """

    def __init__(
        self,
        model,
        num_sample_steps=50,
        solver="euler",
        sigma_min=1e-4,
        sample_pc_size=682,
        perturb_pc=None,
        crop_percent=0.25,
        # ignored DDPM-compat kwargs so the same config can be loaded
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.num_sample_steps = num_sample_steps
        self.solver = solver
        self.sigma_min = sigma_min

        self.pc_size = sample_pc_size
        self.perturb_pc = perturb_pc
        self.crop_percent = crop_percent
        assert self.perturb_pc in [None, "partial", "noisy"]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def interpolate(self, z_1, t, eps):
        """Linear interpolation: z_t = (1-t)*eps + t*z_1."""
        t = t.view(-1, 1)
        return (1.0 - t) * eps + t * z_1

    def forward(self, z_1, t, ret_pred_x=False, noise=None, cond=None):
        """Compute the CFM loss for a batch.

        Args:
            z_1: clean latent vectors [B, D]
            t:   continuous time values [B] in [0, 1]
            cond: optional conditioning (point cloud) [B, N, 3]

        Returns:
            (loss, unreduced_loss)  or
            (loss, z_t, target_vel, pred_vel, unreduced_loss) when ret_pred_x
        """
        eps = default(noise, lambda: torch.randn_like(z_1))
        z_t = self.interpolate(z_1, t, eps)
        target_vel = z_1 - eps

        model_in = (z_t, cond) if cond is not None else z_t
        pred_vel = self.model(model_in, t)

        loss = F.mse_loss(pred_vel, target_vel, reduction="none")
        unreduced_loss = loss.detach().clone().mean(dim=1)

        if ret_pred_x:
            return loss.mean(), z_t, target_vel, pred_vel, unreduced_loss
        return loss.mean(), unreduced_loss

    # ------------------------------------------------------------------
    # Sampling (ODE integration)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, dim, batch_size, noise=None, num_steps=None,
               clip_denoised=True, traj=False, cond=None):
        """Generate samples by integrating the velocity ODE."""
        num_steps = num_steps or self.num_sample_steps
        device = next(self.model.parameters()).device
        z = default(noise, torch.randn(batch_size, dim, device=device))

        dt = 1.0 / num_steps
        trajectory = []

        for i in tqdm(range(num_steps), desc="flow sampling"):
            t_val = i / num_steps
            t_batch = torch.full((batch_size,), t_val, device=device)

            model_in = (z, cond) if cond is not None else z

            if self.solver == "euler":
                v = self.model(model_in, t_batch, pass_cond=1)
                z = z + v * dt
            elif self.solver == "midpoint":
                v1 = self.model(model_in, t_batch, pass_cond=1)
                z_mid = z + v1 * (dt / 2.0)
                t_mid = torch.full((batch_size,), t_val + dt / 2.0, device=device)
                model_in_mid = (z_mid, cond) if cond is not None else z_mid
                v2 = self.model(model_in_mid, t_mid, pass_cond=1)
                z = z + v2 * dt
            else:
                raise ValueError(f"Unknown solver: {self.solver}")

            if clip_denoised:
                z = z.clamp(-1.0, 1.0)

            trajectory.append(z.clone())

        if traj:
            return z, trajectory
        return z, trajectory

    @torch.no_grad()
    def sample_cfg(self, dim, batch_size, noise=None, num_steps=None,
                   clip_denoised=True, cond=None, guidance_scale=1.0):
        """Classifier-free guidance sampling.

        v_guided = (1 + w) * v_cond - w * v_uncond
        """
        num_steps = num_steps or self.num_sample_steps
        device = next(self.model.parameters()).device
        z = default(noise, torch.randn(batch_size, dim, device=device))
        dt = 1.0 / num_steps

        for i in tqdm(range(num_steps), desc="flow CFG sampling"):
            t_val = i / num_steps
            t_batch = torch.full((batch_size,), t_val, device=device)

            v_cond = self.model((z, cond), t_batch, pass_cond=1)
            v_uncond = self.model((z, cond), t_batch, pass_cond=0)
            v = (1.0 + guidance_scale) * v_cond - guidance_scale * v_uncond

            z = z + v * dt
            if clip_denoised:
                z = z.clamp(-1.0, 1.0)

        return z, []

    # ------------------------------------------------------------------
    # Convenience wrappers (same interface as DiffusionModel)
    # ------------------------------------------------------------------

    def diffusion_model_from_latent(self, x_start, cond=None):
        """Drop-in replacement for DiffusionModel.diffusion_model_from_latent.

        Samples t ~ U(0,1), perturbs the condition, computes loss.
        Returns (loss, loss_low_t, loss_high_t, pred_vel, perturbed_pc).
        """
        t = torch.rand(x_start.shape[0], device=x_start.device)

        pc = (
            perturb_point_cloud(cond, self.perturb_pc, self.pc_size, self.crop_percent)
            if cond is not None
            else None
        )

        loss, z_t, target_vel, pred_vel, unreduced_loss = self(
            x_start, t, cond=pc, ret_pred_x=True
        )

        loss_low = unreduced_loss[t < 0.1].mean().detach()
        loss_high = unreduced_loss[t >= 0.1].mean().detach()

        return loss, loss_low, loss_high, pred_vel, pc

    def generate_from_pc(self, pc, load_pc=False, batch=5, save_pc=False,
                         return_pc=False, ddim=False, perturb_pc=True):
        self.eval()
        with torch.no_grad():
            if load_pc:
                pc = sample_pc(pc, self.pc_size).cuda().unsqueeze(0)

            if pc is None:
                input_pc = None
                save_pc = False
                full_perturbed_pc = None
            else:
                if perturb_pc:
                    full_perturbed_pc = perturb_point_cloud(pc, self.perturb_pc)
                    perturbed_pc = full_perturbed_pc[
                        :, torch.randperm(full_perturbed_pc.shape[1])[: self.pc_size]
                    ]
                    input_pc = perturbed_pc.repeat(batch, 1, 1)
                else:
                    full_perturbed_pc = pc
                    perturbed_pc = pc
                    input_pc = pc.repeat(batch, 1, 1)

            if save_pc:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    perturbed_pc.cpu().numpy().squeeze()
                )
                o3d.io.write_point_cloud(f"{save_pc}/input_pc.ply", pcd)

            samp, _ = self.sample(
                dim=self.model.dim_in_out, batch_size=batch, cond=input_pc
            )

        if return_pc:
            return samp, perturbed_pc
        return samp

    def generate_unconditional(self, num_samples):
        self.eval()
        with torch.no_grad():
            samp, _ = self.sample(
                dim=self.model.dim_in_out, batch_size=num_samples, cond=None
            )
        return samp
