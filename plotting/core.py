"""Core plotting utilities and base functionality."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch

if TYPE_CHECKING:
    from model import VAE

# Constants
MNIST_SHAPE = (1, 28, 28)
DEFAULT_DPI = 200
DEFAULT_FORMAT = "webp"
DEFAULT_CMAP = "tab10"
DEFAULT_ALPHA = 0.8


@contextmanager
def model_inference(model: "VAE"):
    """Context manager for model evaluation and no_grad."""
    model.eval()
    with torch.no_grad():
        yield


@contextmanager
def figure_context(
    figsize: tuple[float, float],
    out_path: str,
    dpi: int = DEFAULT_DPI,
    format: str = DEFAULT_FORMAT,
):
    """Context manager for matplotlib figure creation and saving."""
    plt.figure(figsize=figsize)
    try:
        yield
    finally:
        save_figure(out_path, dpi=dpi, format=format)
        plt.close()


def decode_samples(model: "VAE", z: torch.Tensor) -> torch.Tensor:
    """Decode latent vectors and reshape to MNIST format."""
    return torch.sigmoid(model.decode(z)).view(-1, *MNIST_SHAPE)


def grid_from_images(imgs: torch.Tensor, nrow: int, ncol: int) -> torch.Tensor:
    """Create a HxWxC image grid from a batch of images."""
    imgs = imgs.detach().cpu().clamp(0, 1)
    N, C, H, W = imgs.shape
    canvas = torch.zeros((C, nrow * H, ncol * W))

    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            if idx >= N:
                break
            canvas[:, i * H : (i + 1) * H, j * W : (j + 1) * W] = imgs[idx]

    img = canvas.permute(1, 2, 0).numpy()
    return img.squeeze(-1) if C == 1 else img


def save_figure(
    out_path: str,
    dpi: int = DEFAULT_DPI,
    format: str = DEFAULT_FORMAT,
    ensure_dir: bool = True,
    bbox_inches: str = "tight",
) -> None:
    """
    Save the current figure with consistent settings.

    Args:
        out_path: Output file path
        dpi: DPI for the saved image
        format: Image format (webp, png, jpg, etc.)
        ensure_dir: Whether to create output directory if it doesn't exist
        bbox_inches: Bounding box behavior for saving
    """
    # Ensure output path uses consistent format
    path_obj = Path(out_path)
    if format and not path_obj.suffix:
        out_path = str(path_obj.with_suffix(f".{format}"))
    elif format and path_obj.suffix != f".{format}":
        out_path = str(path_obj.with_suffix(f".{format}"))

    # Create directory if needed
    if ensure_dir:
        os.makedirs(Path(out_path).parent, exist_ok=True)

    # Apply tight layout before saving
    plt.tight_layout()

    # Save with consistent settings
    plt.savefig(out_path, dpi=dpi, format=format, bbox_inches=bbox_inches)


@contextmanager
def subplot_context(
    figsize: tuple[float, float],
    out_path: str,
    dpi: int = DEFAULT_DPI,
    format: str = DEFAULT_FORMAT,
):
    """
    Context manager for multi-panel figures with consistent saving.

    Args:
        figsize: Figure size as (width, height)
        out_path: Output file path
        dpi: DPI for the saved image
        format: Image format
    """
    fig = plt.figure(figsize=figsize)
    try:
        yield fig
    finally:
        save_figure(out_path, dpi=dpi, format=format)
        plt.close()


def make_plot_path(
    fig_dir: str,
    base_name: str,
    suffix: str = "",
    format: str = DEFAULT_FORMAT,
) -> str:
    """
    Create a standardized plot file path.

    Args:
        fig_dir: Base directory for figures
        base_name: Base name for the plot (e.g., 'latent_space', 'training_curves')
        suffix: Optional suffix (e.g., 'epochs', 'steps', '2d', 'combined')
        format: File format extension

    Returns:
        Full path to the plot file
    """
    if suffix:
        filename = f"{base_name}_{suffix}.{format}"
    else:
        filename = f"{base_name}.{format}"

    return os.path.join(fig_dir, filename)


def split_plot_path(out_path: str, suffix: str) -> str:
    """
    Create a related plot path by adding a suffix to the base name.

    Args:
        out_path: Original output path
        suffix: Suffix to add to the base name

    Returns:
        New path with suffix added

    Example:
        split_plot_path("/path/figures/interpolation.webp", "latent_sweep")
        -> "/path/figures/interpolation_latent_sweep.webp"
    """
    path_obj = Path(out_path)
    base_name = path_obj.stem
    new_name = f"{base_name}_{suffix}"
    return str(path_obj.with_name(f"{new_name}{path_obj.suffix}"))
