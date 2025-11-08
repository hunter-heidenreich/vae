"""Core plotting utilities and base functionality."""

from contextlib import contextmanager
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch

if TYPE_CHECKING:
    from model import VAE

# Constants
MNIST_SHAPE = (1, 28, 28)
DEFAULT_DPI = 200
DEFAULT_CMAP = "tab10"
DEFAULT_ALPHA = 0.8


@contextmanager
def model_inference(model: "VAE"):
    """Context manager for model evaluation and no_grad."""
    model.eval()
    with torch.no_grad():
        yield


@contextmanager
def figure_context(figsize, out_path: str, dpi: int = DEFAULT_DPI):
    """Context manager for matplotlib figure creation and saving."""
    plt.figure(figsize=figsize)
    try:
        yield
    finally:
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi)
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