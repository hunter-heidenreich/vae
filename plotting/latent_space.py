"""Latent space visualization plots."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .core import DEFAULT_ALPHA, DEFAULT_CMAP, figure_context, save_figure
from .data import apply_pca_if_needed, get_colormap_colors


def save_latent_marginals(Z: np.ndarray, out_path: str):
    """Save 1D marginal KDE plots for all latent dimensions."""
    latent_dim = Z.shape[1]

    with figure_context((8, 5), out_path):
        colors = get_colormap_colors(latent_dim)
        for i in range(latent_dim):
            sns.kdeplot(Z[:, i], label=f"z{i + 1}", color=colors[i], linewidth=2)

        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Latent marginal KDEs ({latent_dim}D)")
        plt.legend()
        plt.grid(True, alpha=0.3)


def save_latent_combined_figure(Z: np.ndarray, Y: np.ndarray, out_path: str):
    """Save a combined figure with both scatter and histogram visualizations."""
    if Z.shape[0] == 0:
        raise ValueError("Empty latent array provided")

    latent_dim = Z.shape[1]

    # Apply PCA if needed for visualization
    Z_plot, pca = apply_pca_if_needed(Z, 2)

    # Create figure with 1x2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: Scatter plot
    ax = axs[0]
    plt.sca(ax)  # Set current axis
    if Z.shape[1] == 1:
        _plot_histogram_1d(Z, Y)
    else:
        _plot_scatter_2d(Z_plot, Y, pca)

    # Right subplot: 2D histogram
    ax = axs[1]
    plt.sca(ax)  # Set current axis
    if latent_dim == 1:
        sns.histplot(Z[:, 0], kde=True, stat="density", alpha=0.7, color="steelblue")
        plt.xlabel("z1")
        plt.ylabel("Density")
        plt.title("Latent 1D distribution")
        plt.grid(True, alpha=0.3)
    else:
        plt.hist2d(Z_plot[:, 0], Z_plot[:, 1], bins=100, cmap="magma")
        plt.colorbar(label="Count")

        if pca is not None:
            plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})")
            plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})")
            plt.title(f"Latent 2D histogram ({latent_dim}D → 2D PCA)")
        else:
            plt.xlabel("z1")
            plt.ylabel("z2")
            plt.title("Latent 2D histogram")

    save_figure(out_path)
    plt.close()


def _plot_scatter_2d(Z: np.ndarray, Y: np.ndarray, pca=None):
    """Plot 2D scatter with appropriate labels."""
    n_classes = len(np.unique(Y))
    scatter = plt.scatter(
        Z[:, 0], Z[:, 1], c=Y, s=5, cmap=DEFAULT_CMAP, alpha=DEFAULT_ALPHA
    )
    plt.colorbar(scatter, ticks=range(n_classes))

    if pca is not None:
        plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})")
        plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})")
        plt.title(f"MNIST latent PCA ({Z.shape[1]}D → 2D)")
    else:
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title("MNIST latent (mu) scatter")


def save_logvar_marginals(LogVar: np.ndarray, out_path: str):
    """Save 1D marginal KDE plots for all log variance dimensions."""
    latent_dim = LogVar.shape[1]

    with figure_context((8, 5), out_path):
        colors = get_colormap_colors(latent_dim)
        for i in range(latent_dim):
            sns.kdeplot(
                LogVar[:, i], label=f"logvar{i + 1}", color=colors[i], linewidth=2
            )

        plt.xlabel("Log Variance Value")
        plt.ylabel("Density")
        plt.title(f"Log Variance marginal KDEs ({latent_dim}D)")
        plt.legend()
        plt.grid(True, alpha=0.3)


def save_logvar_combined_figure(LogVar: np.ndarray, Y: np.ndarray, out_path: str):
    """Save a combined figure with both scatter and histogram visualizations for log variance."""
    if LogVar.shape[0] == 0:
        raise ValueError("Empty log variance array provided")

    latent_dim = LogVar.shape[1]

    # Apply PCA if needed for visualization
    LogVar_plot, pca = apply_pca_if_needed(LogVar, 2)

    # Create figure with 1x2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: Scatter plot
    ax = axs[0]
    plt.sca(ax)  # Set current axis
    if LogVar.shape[1] == 1:
        _plot_logvar_histogram_1d(LogVar, Y)
    else:
        _plot_logvar_scatter_2d(LogVar_plot, Y, pca)

    # Right subplot: 2D histogram
    ax = axs[1]
    plt.sca(ax)  # Set current axis
    if latent_dim == 1:
        sns.histplot(
            LogVar[:, 0], kde=True, stat="density", alpha=0.7, color="steelblue"
        )
        plt.xlabel("logvar1")
        plt.ylabel("Density")
        plt.title("Log Variance 1D distribution")
        plt.grid(True, alpha=0.3)
    else:
        plt.hist2d(LogVar_plot[:, 0], LogVar_plot[:, 1], bins=100, cmap="magma")
        plt.colorbar(label="Count")

        if pca is not None:
            plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})")
            plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})")
            plt.title(f"Log Variance 2D histogram ({latent_dim}D → 2D PCA)")
        else:
            plt.xlabel("logvar1")
            plt.ylabel("logvar2")
            plt.title("Log Variance 2D histogram")

    save_figure(out_path)
    plt.close()


def _plot_logvar_scatter_2d(LogVar: np.ndarray, Y: np.ndarray, pca=None):
    """Plot 2D log variance scatter with appropriate labels."""
    n_classes = len(np.unique(Y))
    scatter = plt.scatter(
        LogVar[:, 0], LogVar[:, 1], c=Y, s=5, cmap=DEFAULT_CMAP, alpha=DEFAULT_ALPHA
    )
    plt.colorbar(scatter, ticks=range(n_classes))

    if pca is not None:
        plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})")
        plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})")
        plt.title(f"MNIST log variance PCA ({LogVar.shape[1]}D → 2D)")
    else:
        plt.xlabel("logvar1")
        plt.ylabel("logvar2")
        plt.title("MNIST latent (logvar) scatter")


def _plot_logvar_histogram_1d(LogVar: np.ndarray, Y: np.ndarray):
    """Plot 1D log variance histogram with class colors."""
    n_classes = len(np.unique(Y))
    colors = get_colormap_colors(n_classes)

    for i, digit in enumerate(np.unique(Y)):
        mask = Y == digit
        plt.hist(
            LogVar[mask, 0], bins=30, alpha=0.7, label=f"Digit {digit}", color=colors[i]
        )

    plt.xlabel("logvar1")
    plt.ylabel("Count")
    plt.title("MNIST latent (logvar) histogram")
    plt.legend()


def _plot_histogram_1d(Z: np.ndarray, Y: np.ndarray):
    """Plot 1D histogram with class colors."""
    n_classes = len(np.unique(Y))
    colors = get_colormap_colors(n_classes)

    for i, digit in enumerate(np.unique(Y)):
        mask = Y == digit
        plt.hist(
            Z[mask, 0], bins=30, alpha=0.7, label=f"Digit {digit}", color=colors[i]
        )

    plt.xlabel("z1")
    plt.ylabel("Count")
    plt.title("MNIST latent (mu) histogram")
    plt.legend()
