"""Data processing utilities for plotting."""

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from .core import DEFAULT_CMAP, model_inference

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


def get_colormap_colors(n_colors: int, cmap_name: str = DEFAULT_CMAP):
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


def collect_latents_with_logvar(
    model: "VAE",
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect latent representations (mu and logvar) and labels from a dataset."""
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    with model_inference(model):
        for bidx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            mu, log_var = model.encode(data)
            mus.append(mu.cpu().numpy())
            logvars.append(log_var.cpu().numpy())
            ys.append(target.numpy())

            if max_batches is not None and (bidx + 1) >= max_batches:
                break

    return np.concatenate(mus, axis=0), np.concatenate(logvars, axis=0), np.concatenate(ys, axis=0)


def collect_all_latent_data(
    model: "VAE",
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Efficiently collect all latent data in a single pass over the dataloader.
    
    This combines the functionality of collect_latents() and collect_latents_with_logvar()
    to eliminate redundant data loading and model inference passes.

    Args:
        model: The VAE model
        dataloader: DataLoader to iterate over
        device: Device to run inference on
        max_batches: Maximum number of batches to process (None for all)

    Returns:
        Tuple of (Z_samples, Mu, LogVar, Y) where:
        - Z_samples: Sampled latent codes (from reparameterization)
        - Mu: Mean parameters from encoder
        - LogVar: Log variance parameters from encoder (sigma from encode, not log_var)
        - Y: Labels/targets
    """
    z_samples: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []  # Note: sigma from encode, not logvar
    ys: list[np.ndarray] = []

    with model_inference(model):
        for bidx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            
            # Get mu and sigma from encoder
            mu, sigma = model.encode(data)
            
            # Convert sigma to std and sample z using reparameterization
            std = model._sigma_to_std(sigma)
            z = model.reparameterize(mu, std)
            
            # Collect all data
            z_samples.append(z.cpu().numpy())
            mus.append(mu.cpu().numpy())
            sigmas.append(sigma.cpu().numpy())  # Return sigma (log_var equivalent)
            ys.append(target.numpy())

            if max_batches is not None and (bidx + 1) >= max_batches:
                break

    return (
        np.concatenate(z_samples, axis=0),
        np.concatenate(mus, axis=0), 
        np.concatenate(sigmas, axis=0),
        np.concatenate(ys, axis=0)
    )


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
