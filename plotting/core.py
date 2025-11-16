"""Core plotting utilities and base functionality."""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import VAE

from .constants import (
    ALPHA_HIGH,
    COLOR_BEST,
    COLOR_BEST_EDGE,
    MARKER_SIZE_HIGHLIGHT,
    ZORDER_HIGHLIGHT,
)

MNIST_SHAPE = (1, 28, 28)
DEFAULT_DPI = 200
DEFAULT_FORMAT = "webp"


def extract_history_data(history: dict, *keys: str) -> tuple:
    return tuple(np.asarray(history[key], dtype=float) for key in keys)


def compute_histogram_bins(data: np.ndarray) -> int:
    """Compute optimal histogram bins using Freedman-Diaconis rule, fallback to Sturges."""
    n = len(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25

    if iqr > 0:
        bin_width = 2 * iqr / (n ** (1 / 3))
        data_range = data.max() - data.min()
        n_bins = int(np.ceil(data_range / bin_width)) if bin_width > 0 else 10
    else:
        n_bins = int(np.ceil(np.log2(n) + 1))

    return max(10, min(n_bins, 100))


@contextmanager
def model_inference(model: VAE):
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
    """Context manager for creating and saving a figure."""
    plt.figure(figsize=figsize)
    try:
        yield
    finally:
        save_figure(out_path, dpi=dpi, format=format)
        plt.close()


@contextmanager
def subplot_context(
    figsize: tuple[float, float],
    out_path: str,
    dpi: int = DEFAULT_DPI,
    format: str = DEFAULT_FORMAT,
):
    """Context manager for creating and saving a figure with subplots."""
    fig = plt.figure(figsize=figsize)
    try:
        yield fig
    finally:
        save_figure(out_path, dpi=dpi, format=format)
        plt.close()


def decode_samples(model: VAE, z: torch.Tensor) -> torch.Tensor:
    """Decode latent samples and reshape to MNIST image format."""
    return torch.sigmoid(model.decode(z)).view(-1, *MNIST_SHAPE)


def grid_from_images(imgs: torch.Tensor, nrow: int, ncol: int) -> torch.Tensor:
    """Arrange images in a grid for visualization."""
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
    """Save the current matplotlib figure to a file."""
    path_obj = Path(out_path)

    if format and path_obj.suffix != f".{format}":
        out_path = str(path_obj.with_suffix(f".{format}"))

    if ensure_dir:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, format=format, bbox_inches=bbox_inches)


def make_plot_path(
    fig_dir: str,
    name: str,
    suffix: str = "",
    group: str = "",
    format: str = DEFAULT_FORMAT,
) -> str:
    """Create a standardized plot file path."""
    filename = f"{name}_{suffix}.{format}" if suffix else f"{name}.{format}"
    path = Path(fig_dir) / group / filename if group else Path(fig_dir) / filename
    return str(path)


def split_plot_path(out_path: str, suffix: str) -> str:
    """Create a related plot path by adding a suffix to the base name."""
    path_obj = Path(out_path)
    return str(path_obj.with_stem(f"{path_obj.stem}_{suffix}"))


def add_best_epoch_marker(
    ax,
    epochs: np.ndarray,
    values: np.ndarray,
    best_epoch: Optional[int],
    label_prefix: str = "Best",
) -> None:
    """Add visual marker at best epoch location."""
    if best_epoch is not None and best_epoch in epochs:
        best_idx = np.where(epochs == best_epoch)[0][0]
        ax.scatter(
            epochs[best_idx],
            values[best_idx],
            marker="*",
            s=MARKER_SIZE_HIGHLIGHT,
            color=COLOR_BEST,
            edgecolor=COLOR_BEST_EDGE,
            linewidth=1.5,
            alpha=ALPHA_HIGH,
            zorder=ZORDER_HIGHLIGHT,
            label=f"{label_prefix} (Epoch {int(best_epoch)})",
        )
