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
        _plot_scatter_2d(Z_plot, Y, pca, latent_dim)

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


def _plot_scatter_2d(Z: np.ndarray, Y: np.ndarray, pca=None, original_dim=None):
    """Plot 2D scatter with appropriate labels."""
    n_classes = len(np.unique(Y))
    scatter = plt.scatter(
        Z[:, 0], Z[:, 1], c=Y, s=5, cmap=DEFAULT_CMAP, alpha=DEFAULT_ALPHA
    )
    plt.colorbar(scatter, ticks=range(n_classes))

    if pca is not None:
        plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})")
        plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})")
        if original_dim is not None:
            plt.title(f"MNIST latent PCA ({original_dim}D → 2D)")
        else:
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
        _plot_logvar_scatter_2d(LogVar_plot, Y, pca, latent_dim)

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


def _plot_logvar_scatter_2d(
    LogVar: np.ndarray, Y: np.ndarray, pca=None, original_dim=None
):
    """Plot 2D log variance scatter with appropriate labels."""
    n_classes = len(np.unique(Y))
    scatter = plt.scatter(
        LogVar[:, 0], LogVar[:, 1], c=Y, s=5, cmap=DEFAULT_CMAP, alpha=DEFAULT_ALPHA
    )
    plt.colorbar(scatter, ticks=range(n_classes))

    if pca is not None:
        plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})")
        plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})")
        if original_dim is not None:
            plt.title(f"MNIST log variance PCA ({original_dim}D → 2D)")
        else:
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


def save_latent_evolution_plots(val_history: dict, fig_dir: str):
    """
    Save plots showing how latent dimension statistics evolve during training.
    Creates two separate focused plots: one for mu evolution, one for sigma evolution.

    Features:
    - Individual dimension evolution lines with error bands
    - Overall average across dimensions with spread indicators
    - Adaptive legend formatting for higher dimensional spaces
    - Clean, publication-ready styling

    Args:
        val_history: Validation history dictionary containing latent statistics
        fig_dir: Base figures directory

    Note:
        Handles any number of latent dimensions. For >5 dimensions, uses
        compact legend formatting with multiple columns.
    """
    from .core import make_plot_path

    # Check if we have latent statistics
    if "mu_mean_per_dim" not in val_history or not val_history["mu_mean_per_dim"]:
        print("No latent evolution data found in validation history")
        return

    # Get epochs and latent dimension data
    epochs = val_history.get(
        "epoch", list(range(1, len(val_history["mu_mean_per_dim"]) + 1))
    )

    # Data for mu (mean parameter)
    mu_mean_per_dim_data = np.array(
        val_history["mu_mean_per_dim"]
    )  # Shape: (n_epochs, n_dims)
    mu_std_per_dim_data = np.array(
        val_history["mu_std_per_dim"]
    )  # Shape: (n_epochs, n_dims)

    # Data for sigma (std parameter) - check if available (for backward compatibility)
    has_sigma_data = (
        "sigma_mean_per_dim" in val_history and val_history["sigma_mean_per_dim"]
    )
    if has_sigma_data:
        sigma_mean_per_dim_data = np.array(
            val_history["sigma_mean_per_dim"]
        )  # Shape: (n_epochs, n_dims)
        sigma_std_per_dim_data = np.array(
            val_history["sigma_std_per_dim"]
        )  # Shape: (n_epochs, n_dims)

    n_epochs, n_dims = mu_mean_per_dim_data.shape
    colors = get_colormap_colors(n_dims)

    # === MU (Mean Parameter) Evolution Plot ===
    mu_out_path = make_plot_path(fig_dir, "mu_evolution", group_dir="latent_space")
    _save_mu_evolution_plot(
        epochs, mu_mean_per_dim_data, mu_std_per_dim_data, colors, n_dims, mu_out_path
    )

    # === SIGMA (Std Parameter) Evolution Plot ===
    if has_sigma_data:
        sigma_out_path = make_plot_path(
            fig_dir, "sigma_evolution", group_dir="latent_space"
        )
        _save_sigma_evolution_plot(
            epochs,
            sigma_mean_per_dim_data,
            sigma_std_per_dim_data,
            colors,
            n_dims,
            sigma_out_path,
        )


def _save_mu_evolution_plot(
    epochs, mu_mean_per_dim_data, mu_std_per_dim_data, colors, n_dims, out_path
):
    """Save μ (mean parameter) evolution plot."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot per-dimension means with error bars showing their std deviations
    for dim in range(n_dims):
        mu_means = mu_mean_per_dim_data[:, dim]
        mu_stds = mu_std_per_dim_data[:, dim]

        # Main line for the mean
        ax.plot(
            epochs,
            mu_means,
            label=f"μ{dim + 1}",
            color=colors[dim],
            linewidth=2.5,
            marker="o",
            markersize=4,
        )

        # Error band showing the std deviation around the mean
        ax.fill_between(
            epochs, mu_means - mu_stds, mu_means + mu_stds, color=colors[dim], alpha=0.2
        )

    # Add overall average as a bold line
    overall_mu_means = np.mean(mu_mean_per_dim_data, axis=1)
    overall_mu_spread = np.std(mu_mean_per_dim_data, axis=1)
    ax.plot(
        epochs,
        overall_mu_means,
        "k-",
        linewidth=3,
        marker="s",
        markersize=6,
        label="μ_avg (across dims)",
        zorder=10,
    )
    ax.fill_between(
        epochs,
        overall_mu_means - overall_mu_spread,
        overall_mu_means + overall_mu_spread,
        color="black",
        alpha=0.1,
        label="±1 std across dims",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("μ (Mean Parameter) Value", fontsize=12)
    ax.set_title(
        "Evolution of Latent μ (Mean Parameters)", fontsize=14, fontweight="bold"
    )

    # Adjust legend placement based on number of dimensions
    if n_dims <= 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, ncol=2)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_figure(out_path)
    plt.close()


def _save_sigma_evolution_plot(
    epochs, sigma_mean_per_dim_data, sigma_std_per_dim_data, colors, n_dims, out_path
):
    """Save σ (std parameter) evolution plot."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot per-dimension sigma means with error bars showing their std deviations
    for dim in range(n_dims):
        sigma_means = sigma_mean_per_dim_data[:, dim]
        sigma_stds = sigma_std_per_dim_data[:, dim]

        # Main line for the mean
        ax.plot(
            epochs,
            sigma_means,
            label=f"σ{dim + 1}",
            color=colors[dim],
            linewidth=2.5,
            marker="s",
            markersize=4,
        )

        # Error band showing the std deviation around the mean
        ax.fill_between(
            epochs,
            sigma_means - sigma_stds,
            sigma_means + sigma_stds,
            color=colors[dim],
            alpha=0.2,
        )

    # Add overall average as a bold line
    overall_sigma_means = np.mean(sigma_mean_per_dim_data, axis=1)
    overall_sigma_spread = np.std(sigma_mean_per_dim_data, axis=1)
    ax.plot(
        epochs,
        overall_sigma_means,
        "k-",
        linewidth=3,
        marker="D",
        markersize=6,
        label="σ_avg (across dims)",
        zorder=10,
    )
    ax.fill_between(
        epochs,
        overall_sigma_means - overall_sigma_spread,
        overall_sigma_means + overall_sigma_spread,
        color="black",
        alpha=0.1,
        label="±1 std across dims",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("σ (Std Parameter) Value", fontsize=12)
    ax.set_title(
        "Evolution of Latent σ (Std Parameters)", fontsize=14, fontweight="bold"
    )

    # Adjust legend placement based on number of dimensions
    if n_dims <= 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, ncol=2)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_figure(out_path)
    plt.close()
