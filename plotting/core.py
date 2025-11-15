"""Core plotting utilities and base functionality."""

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import VAE

# Constants
MNIST_SHAPE = (1, 28, 28)
DEFAULT_DPI = 200
DEFAULT_FORMAT = "webp"
DEFAULT_CMAP = "tab10"
DEFAULT_ALPHA = 0.8


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
    plt.figure(figsize=figsize)
    try:
        yield
    finally:
        save_figure(out_path, dpi=dpi, format=format)
        plt.close()


def decode_samples(model: VAE, z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(model.decode(z)).view(-1, *MNIST_SHAPE)


def grid_from_images(imgs: torch.Tensor, nrow: int, ncol: int) -> torch.Tensor:
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
    path_obj = Path(out_path)
    if format and not path_obj.suffix:
        out_path = str(path_obj.with_suffix(f".{format}"))
    elif format and path_obj.suffix != f".{format}":
        out_path = str(path_obj.with_suffix(f".{format}"))

    if ensure_dir:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, format=format, bbox_inches=bbox_inches)


@contextmanager
def subplot_context(
    figsize: tuple[float, float],
    out_path: str,
    dpi: int = DEFAULT_DPI,
    format: str = DEFAULT_FORMAT,
):
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
    group_dir: str = "",
) -> str:
    """Create a standardized plot file path with optional grouping."""
    if suffix:
        filename = f"{base_name}_{suffix}.{format}"
    else:
        filename = f"{base_name}.{format}"

    base_path = Path(fig_dir)
    if group_dir:
        return str(base_path / group_dir / filename)
    else:
        return str(base_path / filename)


def split_plot_path(out_path: str, suffix: str) -> str:
    """Create a related plot path by adding a suffix to the base name.

    Example:
        split_plot_path("/path/figures/interpolation.webp", "latent_sweep")
        -> "/path/figures/interpolation_latent_sweep.webp"
    """
    path_obj = Path(out_path)
    base_name = path_obj.stem
    new_name = f"{base_name}_{suffix}"
    return str(path_obj.with_name(f"{new_name}{path_obj.suffix}"))


def make_grouped_plot_path(
    fig_dir: str,
    group: str,
    base_name: str,
    suffix: str = "",
    format: str = DEFAULT_FORMAT,
) -> str:
    return make_plot_path(fig_dir, base_name, suffix, format, group_dir=group)
