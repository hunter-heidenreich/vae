"""Image generation and sampling plots."""

from typing import Sized, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import VAE

from .core import (
    decode_samples,
    figure_context,
    grid_from_images,
    model_inference,
    save_figure,
    split_plot_path,
)


def save_samples_figure(
    model: VAE,
    device: torch.device,
    latent_dim: int,
    out_path: str,
    grid: tuple[int, int] = (10, 10),
):
    """Generate and save random samples from the VAE."""
    n_samples = grid[0] * grid[1]
    with model_inference(model):
        z = torch.randn((n_samples, latent_dim), device=device)
        samples = decode_samples(model, z)

    img = grid_from_images(samples, grid[0], grid[1])

    with figure_context((grid[1], grid[0]), out_path):
        plt.axis("off")
        plt.imshow(img, cmap="gray")


def save_recon_figure(
    model: VAE,
    data_batch: torch.Tensor,
    device: torch.device,
    out_path: str,
    n: int = 16,
):
    """Save reconstruction figure showing originals and reconstructions side by side."""
    n = min(n, len(data_batch))

    with model_inference(model):
        data = data_batch[:n].to(device)
        mu, _ = model.encode(data)
        recon = decode_samples(model, mu)

    imgs = torch.cat([data.view(-1, *model.config.input_shape), recon], dim=0)
    img_grid = grid_from_images(imgs, 2, n)

    with figure_context((n * 1.5, 2 * 1.5), out_path):
        plt.axis("off")
        plt.imshow(img_grid, cmap="gray")


def save_interpolation_figure(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    steps: int = 15,
    method: str = "slerp",
):
    """Create interpolation figure between two random data points."""
    x, z = _get_interpolation_data(model, loader, device)
    z0, z1 = z[0], z[1]

    method_lower = method.lower()
    if method_lower not in {"slerp", "lerp"}:
        raise ValueError(
            f"Unknown interpolation method: {method!r}. Expected 'slerp' or 'lerp'."
        )

    t = torch.linspace(0, 1, steps, device=device)
    zs = _slerp(z0, z1, t) if method_lower == "slerp" else _lerp(z0, z1, t)
    interp_imgs = decode_samples(model, zs)

    _plot_interpolation_figure(
        x, z, interp_imgs, model.config.input_shape, method, out_path
    )


def save_latent_sweep_figure(
    model: VAE,
    device: torch.device,
    out_path: str,
    sweep_steps: int = 15,
    sweep_range: tuple[float, float] = (-3.0, 3.0),
):
    """Create latent space sweep figure showing each dimension independently."""
    latent_dim = model.config.latent_dim

    with model_inference(model):
        all_sweep_imgs = []
        dim_labels = []

        for dim in range(latent_dim):
            sweep_values = torch.linspace(
                sweep_range[0], sweep_range[1], sweep_steps, device=device
            )
            z_sweep = torch.zeros(sweep_steps, latent_dim, device=device)
            z_sweep[:, dim] = sweep_values

            sweep_imgs = decode_samples(model, z_sweep)
            all_sweep_imgs.append(sweep_imgs)
            dim_labels.append(f"z{dim + 1}")

    fig, axes = plt.subplots(latent_dim, 1, figsize=(sweep_steps, latent_dim * 1.2))

    if latent_dim == 1:
        axes = [axes]

    for ax, sweep_imgs, label in zip(axes, all_sweep_imgs, dim_labels):
        sweep_grid = grid_from_images(sweep_imgs, 1, sweep_steps)

        ax.imshow(sweep_grid, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Latent Space Sweep ({label})", fontsize=10, pad=5)

    save_figure(out_path)
    plt.close()


def save_interpolation_and_sweep_figures(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    steps: int = 15,
    method: str = "slerp",
    sweep_steps: int = 15,
):
    """Create separate interpolation and latent sweep figures."""
    interp_path = split_plot_path(out_path, "interpolation")
    save_interpolation_figure(model, loader, device, interp_path, steps, method)

    sweep_path = split_plot_path(out_path, "latent_sweep")
    save_latent_sweep_figure(model, device, sweep_path, sweep_steps)


def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between a and b for any shape along last dim."""
    t_ = t.view(-1, *([1] * a.dim()))
    return (1 - t_) * a.unsqueeze(0) + t_ * b.unsqueeze(0)


def _slerp(
    a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Spherical linear interpolation that works for any latent dimensionality."""
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

    # Use LERP for very small angles to avoid numerical instability
    small_angle_mask = omega.squeeze(-1) < 1e-3
    if small_angle_mask.any():
        lerp_result = _lerp(a, b, t)
        out[small_angle_mask] = lerp_result[small_angle_mask]

    return out


def _add_latent_vector_visualization(
    ax: "plt.Axes", z0_vals: np.ndarray, z1_vals: np.ndarray
) -> None:
    """Add colored grid visualization of latent vectors below the images."""
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle

    # Determine how many dimensions to show based on latent space size
    total_dims = len(z0_vals)
    if total_dims <= 4:
        # Show all dimensions for small latent spaces (2D, 3D, 4D)
        dims_to_show = total_dims
        show_ellipsis = False
    elif total_dims <= 8:
        # Show first 6 dimensions for medium spaces (5D-8D)
        dims_to_show = 6
        show_ellipsis = True
    else:
        # Show first 8 dimensions for large spaces (9D+)
        dims_to_show = 8
        show_ellipsis = True

    # Extract the dimensions to display
    z0_display = z0_vals[:dims_to_show]
    z1_display = z1_vals[:dims_to_show]

    # Combine both vectors to get consistent color scale
    all_vals = np.concatenate([z0_display, z1_display])
    vmin, vmax = all_vals.min(), all_vals.max()

    # Avoid division by zero if all values are the same
    if vmax == vmin:
        vmin, vmax = vmin - 0.1, vmax + 0.1

    # Create normalization for consistent coloring
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # Parameters for the grid visualization - prevent overlaps
    # Calculate maximum available width and adjust cell size accordingly
    available_width = 0.4  # 40% of figure width for each grid
    total_grid_width = dims_to_show * 0.06 + (dims_to_show - 1) * 0.005  # Estimate

    if total_grid_width > available_width:
        # Scale down if it would exceed available space
        scale_factor = available_width / total_grid_width
        cell_width = 0.06 * scale_factor
        grid_spacing = 0.005 * scale_factor
    else:
        cell_width = 0.06 if dims_to_show > 4 else 0.08
        grid_spacing = 0.005

    cell_height = 0.06
    y_offset = -0.28  # Move further down to prevent overlap

    # Draw z0 grid (left) - constrain to available space
    z0_grid_width = dims_to_show * (cell_width + grid_spacing) - grid_spacing
    z0_x_start = 0.25 - z0_grid_width / 2
    for i, val in enumerate(z0_display):
        x = z0_x_start + i * (cell_width + grid_spacing)
        color = cmap(norm(val))
        rect = Rectangle(
            (x, y_offset),
            cell_width,
            cell_height,
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)

        # Add value text inside cell with size adjustment
        text_color = "black" if norm(val) > 0.5 else "white"
        if dims_to_show > 8:
            fontsize = 6
        elif dims_to_show > 4:
            fontsize = 7
        else:
            fontsize = 9
        ax.text(
            x + cell_width / 2,
            y_offset + cell_height / 2,
            f"{val:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            weight="bold",
            clip_on=False,
        )

    # Add ellipsis if not showing all dimensions
    if show_ellipsis:
        ellipsis_x = z0_x_start + z0_grid_width + 0.01
        ax.text(
            ellipsis_x,
            y_offset + cell_height / 2,
            "...",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            weight="bold",
            clip_on=False,
        )

    # Add z0 label with dimension info
    label_text = "z₀" if not show_ellipsis else f"z₀ [1-{dims_to_show} of {total_dims}]"
    ax.text(
        0.25,
        y_offset - 0.02,
        label_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        weight="bold",
        clip_on=False,
    )

    # Draw z1 grid (right) - constrain to available space
    z1_grid_width = dims_to_show * (cell_width + grid_spacing) - grid_spacing
    z1_x_start = 0.75 - z1_grid_width / 2
    for i, val in enumerate(z1_display):
        x = z1_x_start + i * (cell_width + grid_spacing)
        color = cmap(norm(val))
        rect = Rectangle(
            (x, y_offset),
            cell_width,
            cell_height,
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)

        # Add value text inside cell with size adjustment
        text_color = "black" if norm(val) > 0.5 else "white"
        if dims_to_show > 8:
            fontsize = 6
        elif dims_to_show > 4:
            fontsize = 7
        else:
            fontsize = 9
        ax.text(
            x + cell_width / 2,
            y_offset + cell_height / 2,
            f"{val:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            weight="bold",
            clip_on=False,
        )

    # Add ellipsis if not showing all dimensions
    if show_ellipsis:
        ellipsis_x = z1_x_start + z1_grid_width + 0.01
        ax.text(
            ellipsis_x,
            y_offset + cell_height / 2,
            "...",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            weight="bold",
            clip_on=False,
        )

    # Add z1 label with dimension info
    label_text = "z₁" if not show_ellipsis else f"z₁ [1-{dims_to_show} of {total_dims}]"
    ax.text(
        0.75,
        y_offset - 0.02,
        label_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        weight="bold",
        clip_on=False,
    )

    # Add a simple colorbar legend
    colorbar_x = 0.4
    colorbar_y = y_offset - 0.12
    colorbar_width = 0.2
    colorbar_height = 0.02

    # Create simple colorbar using rectangles
    n_segments = 50
    segment_width = colorbar_width / n_segments
    for i in range(n_segments):
        val = vmin + (vmax - vmin) * i / (n_segments - 1)
        color = cmap(norm(val))
        x = colorbar_x + i * segment_width
        rect = Rectangle(
            (x, colorbar_y),
            segment_width,
            colorbar_height,
            facecolor=color,
            edgecolor="none",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)

    # Add colorbar border
    border = Rectangle(
        (colorbar_x, colorbar_y),
        colorbar_width,
        colorbar_height,
        facecolor="none",
        edgecolor="black",
        linewidth=1,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(border)

    # Add colorbar labels
    ax.text(
        colorbar_x,
        colorbar_y - 0.02,
        f"{vmin:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        clip_on=False,
    )
    ax.text(
        colorbar_x + colorbar_width,
        colorbar_y - 0.02,
        f"{vmax:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        clip_on=False,
    )


def _get_interpolation_data(
    model: VAE, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fetch two random data points and their latent representations."""
    ds = loader.dataset  # type: ignore[attr-defined]
    ds_sized = cast(Sized, ds)
    idx = torch.randint(0, len(ds_sized), (2,))
    x0, _ = ds[idx[0].item()]  # type: ignore[index]
    x1, _ = ds[idx[1].item()]  # type: ignore[index]
    x = torch.stack([x0, x1], dim=0).to(device)

    with model_inference(model):
        mu, _ = model.encode(x)
    return x, mu


def _plot_interpolation_figure(
    original_images: torch.Tensor,
    latents: torch.Tensor,
    interp_imgs: torch.Tensor,
    input_shape: tuple[int, ...],
    method: str,
    out_path: str,
):
    """Handle the plotting logic for the interpolation figure."""
    steps = interp_imgs.shape[0]
    fig_height = 6.5
    fig_width = max(10.0, steps * 0.8)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.6},
    )

    # Top row: Original images and latent vector visualizations
    ax_orig = axes[0]
    orig_imgs_grid = grid_from_images(
        original_images.view(-1, *input_shape).cpu(), 1, 2
    )
    ax_orig.imshow(np.clip(orig_imgs_grid, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax_orig.axis("off")
    ax_orig.set_title(
        "Original Images and Their Latent Embeddings", fontsize=12, pad=15
    )

    z0_vals, z1_vals = latents[0].cpu().numpy(), latents[1].cpu().numpy()
    _add_latent_vector_visualization(ax_orig, z0_vals, z1_vals)

    # Bottom row: Interpolation sequence
    ax_interp = axes[1]
    interp_grid = grid_from_images(interp_imgs, 1, steps)
    ax_interp.imshow(interp_grid, cmap="gray")
    ax_interp.axis("off")
    ax_interp.set_title(f"{method.upper()} Interpolation Sequence", fontsize=12, pad=10)

    save_figure(out_path)
    plt.close()
