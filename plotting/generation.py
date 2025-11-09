"""Image generation and sampling plots."""

from typing import TYPE_CHECKING, Sized, cast

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .core import (decode_samples, figure_context, grid_from_images,
                   model_inference)

if TYPE_CHECKING:
    from model import VAE


def save_samples_figure(
    model: "VAE",
    device,
    latent_dim: int,
    out_path: str,
    n: int = 100,
    grid: tuple[int, int] = (10, 10),
):
    """Generate and save random samples from the VAE."""
    with model_inference(model):
        z = torch.randn((n, latent_dim), device=device)
        samples = decode_samples(model, z)

    img = grid_from_images(samples, grid[0], grid[1])

    with figure_context((grid[1], grid[0]), out_path):
        plt.axis("off")
        plt.imshow(img, cmap="gray")


def save_recon_figure(
    model: "VAE",
    data_batch: torch.Tensor,
    device,
    out_path: str,
    n: int = 16,
):
    """Save reconstruction figure showing originals and reconstructions side by side."""
    n = min(n, len(data_batch))

    with model_inference(model):
        data = data_batch[:n].to(device)
        mu, _ = model.encode(data)
        recon = decode_samples(model, mu)

    # Stack originals then reconstructions
    imgs = torch.cat([data.view(-1, 1, 28, 28), recon], dim=0)
    img_grid = grid_from_images(imgs, 2, n)  # 2×n grid

    with figure_context((n * 1.5, 2 * 1.5), out_path):
        plt.axis("off")
        plt.imshow(img_grid, cmap="gray")


def save_interpolation_combined_figure(
    model: "VAE",
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    steps: int = 15,
    method: str = "slerp",
    sweep_steps: int = 15,
):
    """Create combined interpolation figure with data interpolation on top and latent sweep on bottom."""
    # First, get interpolation between two random examples
    ds = loader.dataset  # type: ignore[attr-defined]
    ds_sized = cast(Sized, ds)
    idx = torch.randint(0, len(ds_sized), (2,))
    x0, _ = ds[idx[0].item()]  # type: ignore[index]
    x1, _ = ds[idx[1].item()]  # type: ignore[index]
    x = torch.stack([x0, x1], dim=0).to(device)

    with model_inference(model):
        # Interpolation between examples
        mu, _ = model.encode(x)
        z0, z1 = mu[0], mu[1]
        t = torch.linspace(0, 1, steps, device=device)
        zs = _slerp(z0, z1, t) if method.lower() == "slerp" else _lerp(z0, z1, t)
        interp_imgs = decode_samples(model, zs)

        # Latent space sweep along first dimension
        z1_sweep = torch.linspace(-3, 3, sweep_steps, device=device)
        z_sweep = torch.zeros(sweep_steps, model.config.latent_dim, device=device)
        z_sweep[:, 0] = z1_sweep
        sweep_imgs = decode_samples(model, z_sweep)

    # Create grids
    interp_grid = grid_from_images(interp_imgs, 1, steps)
    sweep_grid = grid_from_images(sweep_imgs, 1, sweep_steps)

    # Create figure with 2 subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(steps, sweep_steps), 3))

    # Top plot: interpolation between examples
    ax1.imshow(interp_grid, cmap="gray")
    ax1.axis("off")
    ax1.set_title("Interpolation Between Two Data Points", fontsize=10, pad=10)

    # Bottom plot: latent sweep
    ax2.imshow(sweep_grid, cmap="gray")
    ax2.axis("off")
    ax2.set_title("Latent Space Sweep (z1 dimension)", fontsize=10, pad=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between a and b for any shape along last dim."""
    t_ = t.view(-1, *([1] * a.dim()))
    return (1 - t_) * a.unsqueeze(0) + t_ * b.unsqueeze(0)


def _slerp(
    a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Spherical linear interpolation that works for any latent dimensionality."""
    # Normalize directions for stable angle computation
    a_norm = torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(eps)
    b_norm = torch.linalg.norm(b, dim=-1, keepdim=True).clamp_min(eps)
    a_dir = a / a_norm
    b_dir = b / b_norm

    dot = (a_dir * b_dir).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    omega = torch.arccos(dot)
    so = torch.sin(omega).clamp_min(eps)

    t_ = t.view(-1, *([1] * a.dim()))
    s0 = torch.sin((1 - t_) * omega) / so
    s1 = torch.sin(t_ * omega) / so

    out = (s0 * a.unsqueeze(0)) + (s1 * b.unsqueeze(0))

    # Fall back to lerp for small angles
    small_angle = (omega < 1e-2).squeeze(-1)
    if small_angle.any():
        out[small_angle] = _lerp(a, b, t)[small_angle]
    return out
