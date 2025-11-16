"""Data processing utilities for plotting."""

import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from model import VAE

from .constants import DEFAULT_CMAP
from .core import model_inference


def apply_pca_if_needed(
    Z: np.ndarray, target_dims: int = 2
) -> tuple[np.ndarray, PCA | None]:
    if Z.shape[1] <= target_dims:
        return Z, None

    pca = PCA(n_components=target_dims)
    Z_reduced = pca.fit_transform(Z)
    return Z_reduced, pca


def get_colormap_colors(n_colors: int, cmap_name: str = DEFAULT_CMAP) -> np.ndarray:
    return cm.get_cmap(cmap_name)(np.linspace(0, 1, n_colors))


def collect_latents(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _, mus, _, ys = collect_all_latent_data(model, dataloader, device, max_batches)
    return mus, ys


def collect_latents_with_std(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, mus, stds, ys = collect_all_latent_data(model, dataloader, device, max_batches)
    return mus, stds, ys


def collect_all_latent_data(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns: (Z_samples, Mu, Std, Y)"""
    z_samples: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    with model_inference(model):
        for bidx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            mu, std = model.encode(data, true_std=True)
            z = model.reparameterize(mu, std)

            z_samples.append(z.cpu().numpy())
            mus.append(mu.cpu().numpy())
            stds.append(std.cpu().numpy())
            ys.append(target.numpy())

            if max_batches is not None and (bidx + 1) >= max_batches:
                break

    return (
        np.concatenate(z_samples, axis=0),
        np.concatenate(mus, axis=0),
        np.concatenate(stds, axis=0),
        np.concatenate(ys, axis=0),
    )


def compute_kl_per_dimension(
    mu: torch.Tensor, variance_param: torch.Tensor, is_logvar: bool = True
) -> torch.Tensor:
    if is_logvar:
        kl_per_dim = 0.5 * (mu.pow(2) + variance_param.exp() - 1 - variance_param)
    else:
        variance = variance_param.pow(2)
        kl_per_dim = 0.5 * (mu.pow(2) + variance - 1 - torch.log(variance))

    return kl_per_dim.mean(dim=0)
