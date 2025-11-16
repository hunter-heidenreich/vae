"""Image generation and sampling plots."""

from typing import Sized, cast

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
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
) -> None:
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
) -> None:
    from .constants import CMAP_PROGRESS

    n = min(n, len(data_batch))

    with model_inference(model):
        data = data_batch[:n].to(device)
        mu, _ = model.encode(data)
        recon = decode_samples(model, mu)

    data_view = data.view(-1, *model.config.input_shape)
    delta = data_view - recon

    imgs = torch.cat([data_view, recon, delta], dim=0)
    img_grid_orig_recon = grid_from_images(imgs[: 2 * n], 2, n)
    img_grid_delta = grid_from_images(imgs[2 * n :], 1, n)

    with figure_context((n * 1.5, 3 * 1.5), out_path):
        fig = plt.gcf()
        gs = fig.add_gridspec(3, 1, hspace=0.02, height_ratios=[1, 1, 1])

        ax1 = fig.add_subplot(gs[:2, 0])
        ax1.imshow(img_grid_orig_recon, cmap="gray")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[2, 0])
        vmax = torch.abs(delta).max().item()
        im = ax2.imshow(img_grid_delta, cmap=CMAP_PROGRESS, vmin=-vmax, vmax=vmax)
        ax2.axis("off")

        plt.colorbar(im, ax=ax2, orientation="horizontal", pad=0.02, fraction=0.046)


def save_interpolation_figure(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    steps: int = 15,
    method: str = "slerp",
) -> None:
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
) -> None:
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
) -> None:
    interp_path = split_plot_path(out_path, "interpolation")
    save_interpolation_figure(model, loader, device, interp_path, steps, method)

    sweep_path = split_plot_path(out_path, "latent_sweep")
    save_latent_sweep_figure(model, device, sweep_path, sweep_steps)


def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_ = t.view(-1, *([1] * a.dim()))
    return (1 - t_) * a.unsqueeze(0) + t_ * b.unsqueeze(0)


def _slerp(
    a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
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

    small_angle_mask = omega.squeeze(-1) < 1e-3
    if small_angle_mask.any():
        lerp_result = _lerp(a, b, t)
        out[small_angle_mask] = lerp_result[small_angle_mask]

    return out


def _get_grid_config(total_dims: int) -> tuple[int, bool, float, float, int]:
    if total_dims <= 4:
        dims_to_show, show_ellipsis = total_dims, False
    elif total_dims <= 8:
        dims_to_show, show_ellipsis = 6, True
    else:
        dims_to_show, show_ellipsis = 8, True

    AVAILABLE_WIDTH = 0.4
    BASE_CELL_WIDTH = 0.06
    BASE_SPACING = 0.005
    LARGE_CELL_WIDTH = 0.08

    total_grid_width = (
        dims_to_show * BASE_CELL_WIDTH + (dims_to_show - 1) * BASE_SPACING
    )

    if total_grid_width > AVAILABLE_WIDTH:
        scale_factor = AVAILABLE_WIDTH / total_grid_width
        cell_width = BASE_CELL_WIDTH * scale_factor
        grid_spacing = BASE_SPACING * scale_factor
    else:
        cell_width = LARGE_CELL_WIDTH if dims_to_show <= 4 else BASE_CELL_WIDTH
        grid_spacing = BASE_SPACING

    fontsize = 9 if dims_to_show <= 4 else 7 if dims_to_show <= 8 else 6

    return dims_to_show, show_ellipsis, cell_width, grid_spacing, fontsize


def _draw_latent_grid(
    ax: "plt.Axes",
    values: np.ndarray,
    x_center: float,
    y_offset: float,
    cell_width: float,
    cell_height: float,
    grid_spacing: float,
    norm,
    cmap,
    fontsize: int,
) -> float:
    dims_to_show = len(values)
    grid_width = dims_to_show * (cell_width + grid_spacing) - grid_spacing
    x_start = x_center - grid_width / 2

    for i, val in enumerate(values):
        x = x_start + i * (cell_width + grid_spacing)
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

        text_color = "black" if norm(val) > 0.5 else "white"
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

    return grid_width


def _add_colorbar(
    ax: "plt.Axes", vmin: float, vmax: float, y_offset: float, norm, cmap
) -> None:
    COLORBAR_X = 0.4
    COLORBAR_WIDTH = 0.2
    COLORBAR_HEIGHT = 0.02
    N_SEGMENTS = 50

    colorbar_y = y_offset - 0.12
    segment_width = COLORBAR_WIDTH / N_SEGMENTS

    for i in range(N_SEGMENTS):
        val = vmin + (vmax - vmin) * i / (N_SEGMENTS - 1)
        color = cmap(norm(val))
        x = COLORBAR_X + i * segment_width
        rect = Rectangle(
            (x, colorbar_y),
            segment_width,
            COLORBAR_HEIGHT,
            facecolor=color,
            edgecolor="none",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)

    border = Rectangle(
        (COLORBAR_X, colorbar_y),
        COLORBAR_WIDTH,
        COLORBAR_HEIGHT,
        facecolor="none",
        edgecolor="black",
        linewidth=1,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(border)

    for x_pos, label in [
        (COLORBAR_X, f"{vmin:.2f}"),
        (COLORBAR_X + COLORBAR_WIDTH, f"{vmax:.2f}"),
    ]:
        ax.text(
            x_pos,
            colorbar_y - 0.02,
            label,
            transform=ax.transAxes,
            ha="left" if x_pos == COLORBAR_X else "right",
            va="top",
            fontsize=10,
            clip_on=False,
        )


def _add_latent_vector_visualization(
    ax: "plt.Axes", z0_vals: np.ndarray, z1_vals: np.ndarray
) -> None:
    total_dims = len(z0_vals)
    dims_to_show, show_ellipsis, cell_width, grid_spacing, fontsize = _get_grid_config(
        total_dims
    )
    z0_display = z0_vals[:dims_to_show]
    z1_display = z1_vals[:dims_to_show]
    all_vals = np.concatenate([z0_display, z1_display])
    vmin, vmax = all_vals.min(), all_vals.max()

    if vmax == vmin:
        vmin, vmax = vmin - 0.1, vmax + 0.1

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    cell_height = 0.06
    y_offset = -0.28

    z0_grid_width = _draw_latent_grid(
        ax,
        z0_display,
        0.25,
        y_offset,
        cell_width,
        cell_height,
        grid_spacing,
        norm,
        cmap,
        fontsize,
    )

    z1_grid_width = _draw_latent_grid(
        ax,
        z1_display,
        0.75,
        y_offset,
        cell_width,
        cell_height,
        grid_spacing,
        norm,
        cmap,
        fontsize,
    )

    if show_ellipsis:
        for x_center, grid_width in [(0.25, z0_grid_width), (0.75, z1_grid_width)]:
            x_start = x_center - grid_width / 2
            ellipsis_x = x_start + grid_width + 0.01
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

    label_template = (
        "z₀" if not show_ellipsis else f"z₀ [1-{dims_to_show} of {total_dims}]"
    )
    for x_center, idx in [(0.25, 0), (0.75, 1)]:
        label = label_template.replace("z₀", "z₀" if idx == 0 else "z₁")
        ax.text(
            x_center,
            y_offset - 0.02,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            weight="bold",
            clip_on=False,
        )

    _add_colorbar(ax, vmin, vmax, y_offset, norm, cmap)


def _get_interpolation_data(
    model: VAE, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
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
    steps = interp_imgs.shape[0]
    fig_height = 6.5
    fig_width = max(10.0, steps * 0.8)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.6},
    )

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

    ax_interp = axes[1]
    interp_grid = grid_from_images(interp_imgs, 1, steps)
    ax_interp.imshow(interp_grid, cmap="gray")
    ax_interp.axis("off")
    ax_interp.set_title(f"{method.upper()} Interpolation Sequence", fontsize=12, pad=10)

    save_figure(out_path)
    plt.close()
