"""Data processing utilities for plotting."""

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from .core import model_inference, DEFAULT_CMAP

if TYPE_CHECKING:
    from model import VAE


def apply_pca_if_needed(
    Z: np.ndarray, target_dims: int = 2
) -> tuple[np.ndarray, PCA | None]:
    """Apply PCA reduction if dimensionality > target_dims."""
    if Z.shape[1] <= target_dims:
        return Z, None

    pca = PCA(n_components=target_dims)
    Z_reduced = pca.fit_transform(Z)
    return Z_reduced, pca


def get_colormap_colors(n_colors: int, cmap_name: str = DEFAULT_CMAP) -> np.ndarray:
    """Get evenly spaced colors from a colormap."""
    return cm.get_cmap(cmap_name)(np.linspace(0, 1, n_colors))


def collect_latents(
    model: "VAE",
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect latent representations and labels from a dataset."""
    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    with model_inference(model):
        for bidx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            mu, _ = model.encode(data)
            zs.append(mu.cpu().numpy())
            ys.append(target.numpy())

            if max_batches is not None and (bidx + 1) >= max_batches:
                break

    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def compute_kl_per_dimension(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence for each latent dimension separately.

    Args:
        mu: Mean parameters [batch_size, latent_dim]
        logvar: Log variance parameters [batch_size, latent_dim]

    Returns:
        KL divergence per dimension [latent_dim]
    """
    # KL_i = 0.5 * (mu_i^2 + exp(logvar_i) - 1 - logvar_i)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
    # Average across batch dimension
    return kl_per_dim.mean(dim=0)