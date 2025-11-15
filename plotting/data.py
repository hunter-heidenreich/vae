"""Data processing utilities for plotting."""

import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from model import VAE

from .core import DEFAULT_CMAP, model_inference


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
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect latent means and labels from a dataset.

    This is a convenience wrapper around ``collect_all_latent_data`` that
    discards sampled latents and standard deviations.
    """
    z_samples, mus, _, ys = collect_all_latent_data(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
    )

    # ``collect_all_latent_data`` already computes sampled latents via
    # reparameterization. For backwards compatibility with the original
    # implementation of ``collect_latents`` (which returned the encoder
    # means), we intentionally ignore ``z_samples`` here and return ``mus``.
    return mus, ys


def collect_latents_with_std(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect latent means, std deviations, and labels from a dataset.

    This is a convenience wrapper around ``collect_all_latent_data`` that
    discards the sampled latent codes.
    """
    _, mus, stds, ys = collect_all_latent_data(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
    )

    return mus, stds, ys


def collect_all_latent_data(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Efficiently collect all latent data in a single pass over the dataloader.

    This combines the functionality of collect_latents() and collect_latents_with_std()
    to eliminate redundant data loading and model inference passes.

    Args:
        model: The VAE model
        dataloader: DataLoader to iterate over
        device: Device to run inference on
        max_batches: Maximum number of batches to process (None for all)

    Returns:
        Tuple of (Z_samples, Mu, Std, Y) where:
        - Z_samples: Sampled latent codes (from reparameterization)
        - Mu: Mean parameters from encoder
        - Std: Standard deviation parameters from encoder
        - Y: Labels/targets
    """
    z_samples: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    with model_inference(model):
        for bidx, (data, target) in enumerate(dataloader):
            data = data.to(device)

            # Get mu and std from encoder (consistent with collect_latents_with_std)
            mu, std = model.encode(data, true_std=True)

            # Sample z using reparameterization
            z = model.reparameterize(mu, std)

            # Collect all data
            z_samples.append(z.cpu().numpy())
            mus.append(mu.cpu().numpy())
            stds.append(std.cpu().numpy())  # Return std deviation
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
    """
    Compute KL divergence for each latent dimension separately.

    Args:
        mu: Mean parameters [batch_size, latent_dim]
        variance_param: Either log variance parameters or standard deviation [batch_size, latent_dim]
        is_logvar: If True, variance_param is log-variance. If False, variance_param is standard deviation.

    Returns:
        KL divergence per dimension [latent_dim]
    """
    if is_logvar:
        # KL_i = 0.5 * (mu_i^2 + exp(logvar_i) - 1 - logvar_i)
        kl_per_dim = 0.5 * (mu.pow(2) + variance_param.exp() - 1 - variance_param)
    else:
        # variance_param is std, so variance = std^2
        variance = variance_param.pow(2)
        kl_per_dim = 0.5 * (mu.pow(2) + variance - 1 - torch.log(variance))

    # Average across batch dimension
    return kl_per_dim.mean(dim=0)
