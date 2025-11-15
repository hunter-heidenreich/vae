"""Latent space visualization plots."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from .core import DEFAULT_ALPHA, DEFAULT_CMAP, figure_context, save_figure
from .data import apply_pca_if_needed, get_colormap_colors

# Styling constants
MARGINAL_FIGURE_SIZE = (8, 5)
COMBINED_FIGURE_SIZE = (14, 6)
EVOLUTION_FIGURE_SIZE = (12, 8)

# Plot styling
LINE_WIDTH = 2
LINE_WIDTH_BOLD = 2.5
LINE_WIDTH_EXTRA_BOLD = 3
SCATTER_SIZE = 5
MARKER_SIZE = 4
MARKER_SIZE_LARGE = 6
HISTOGRAM_BINS = 30
HISTOGRAM_BINS_2D = 100
GRID_ALPHA = 0.3
FILL_ALPHA = 0.2
HISTOGRAM_ALPHA = 0.7

# Font sizes
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14
SMALL_FONTSIZE = 9

# Legend configuration
MAX_DIMS_SINGLE_COLUMN = 5


def _configure_legend(ax, n_dims: int) -> None:
    """Configure legend based on number of dimensions."""
    if n_dims <= MAX_DIMS_SINGLE_COLUMN:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=SMALL_FONTSIZE, ncol=2
        )


def save_latent_marginals(Z: np.ndarray, out_path: str) -> None:
    """Save 1D marginal KDE plots for all latent dimensions."""
    _save_marginals(Z, out_path, "z", "Latent marginal KDEs")


def save_logvar_marginals(LogVar: np.ndarray, out_path: str) -> None:
    """Save 1D marginal KDE plots for all log variance dimensions."""
    _save_marginals(LogVar, out_path, "logvar", "Log Variance marginal KDEs")


def _save_marginals(
    data: np.ndarray, out_path: str, var_prefix: str, title: str
) -> None:
    """Save 1D marginal KDE plots for dimensions.

    Args:
        data: Data array of shape (n_samples, n_dims)
        out_path: Output file path
        var_prefix: Prefix for dimension labels (e.g., 'z', 'logvar')
        title: Plot title
    """
    latent_dim = data.shape[1]

    with figure_context(MARGINAL_FIGURE_SIZE, out_path):
        colors = get_colormap_colors(latent_dim)
        for i in range(latent_dim):
            sns.kdeplot(
                data[:, i],
                label=f"{var_prefix}{i + 1}",
                color=colors[i],
                linewidth=LINE_WIDTH,
            )

        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"{title} ({latent_dim}D)")
        plt.legend()
        plt.grid(True, alpha=GRID_ALPHA)


def save_latent_combined_figure(Z: np.ndarray, Y: np.ndarray, out_path: str) -> None:
    """Save a combined figure with both scatter and histogram visualizations."""
    _save_combined_figure(Z, Y, out_path, var_name="z", title_prefix="Latent")


def save_logvar_combined_figure(
    LogVar: np.ndarray, Y: np.ndarray, out_path: str
) -> None:
    """Save a combined figure with both scatter and histogram visualizations for log variance."""
    _save_combined_figure(
        LogVar, Y, out_path, var_name="logvar", title_prefix="Log Variance"
    )


def _save_combined_figure(
    data: np.ndarray, Y: np.ndarray, out_path: str, var_name: str, title_prefix: str
) -> None:
    """Save a combined figure with scatter and histogram visualizations.

    Args:
        data: Latent data array of shape (n_samples, n_dims)
        Y: Labels array
        out_path: Output file path
        var_name: Variable name for axis labels (e.g., 'z', 'logvar')
        title_prefix: Prefix for plot titles (e.g., 'Latent', 'Log Variance')
    """
    if data.shape[0] == 0:
        raise ValueError(f"Empty {title_prefix.lower()} array provided")

    latent_dim = data.shape[1]
    data_plot, pca = apply_pca_if_needed(data, 2)

    # Create figure with 1x2 subplots
    fig, axs = plt.subplots(1, 2, figsize=COMBINED_FIGURE_SIZE)

    # Left subplot: Scatter or 1D histogram
    if data.shape[1] == 1:
        _plot_histogram_1d(axs[0], data, Y, var_name, title_prefix)
    else:
        _plot_scatter_2d(axs[0], data_plot, Y, pca, latent_dim, var_name, title_prefix)

    # Right subplot: Density visualization
    _plot_density_visualization(
        axs[1], data, data_plot, pca, latent_dim, var_name, title_prefix
    )

    save_figure(out_path)
    plt.close()


def _plot_scatter_2d(
    ax,
    data: np.ndarray,
    Y: np.ndarray,
    pca: Optional[PCA],
    original_dim: int,
    var_name: str,
    title_prefix: str,
) -> None:
    """Plot 2D scatter with appropriate labels."""
    n_classes = len(np.unique(Y))
    scatter = ax.scatter(
        data[:, 0],
        data[:, 1],
        c=Y,
        s=SCATTER_SIZE,
        cmap=DEFAULT_CMAP,
        alpha=DEFAULT_ALPHA,
    )
    plt.colorbar(scatter, ax=ax, ticks=range(n_classes))

    if pca is not None:
        ax.set_xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})")
        ax.set_ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})")
        ax.set_title(f"{title_prefix} PCA ({original_dim}D → 2D)")
    else:
        ax.set_xlabel(f"{var_name}1")
        ax.set_ylabel(f"{var_name}2")
        ax.set_title(f"{title_prefix} Scatter")


def _plot_density_visualization(
    ax,
    data: np.ndarray,
    data_plot: np.ndarray,
    pca: Optional[PCA],
    latent_dim: int,
    var_name: str,
    title_prefix: str,
) -> None:
    """Plot density visualization (1D or 2D histogram).

    Args:
        ax: Matplotlib axis to plot on
        data: Original data array
        data_plot: PCA-reduced data (if applicable)
        pca: PCA object (if applicable)
        latent_dim: Original dimensionality
        var_name: Variable name for axis labels
        title_prefix: Prefix for plot title
    """
    if latent_dim == 1:
        sns.histplot(
            data[:, 0],
            kde=True,
            stat="density",
            alpha=HISTOGRAM_ALPHA,
            color="steelblue",
            ax=ax,
        )
        ax.set_xlabel(f"{var_name}1")
        ax.set_ylabel("Density")
        ax.set_title(f"{title_prefix} 1D Distribution")
        ax.grid(True, alpha=GRID_ALPHA)
    else:
        hist = ax.hist2d(
            data_plot[:, 0], data_plot[:, 1], bins=HISTOGRAM_BINS_2D, cmap="magma"
        )
        plt.colorbar(hist[3], ax=ax, label="Count")

        if pca is not None:
            ax.set_xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})")
            ax.set_ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})")
            ax.set_title(f"{title_prefix} 2D Histogram ({latent_dim}D → 2D PCA)")
        else:
            ax.set_xlabel(f"{var_name}1")
            ax.set_ylabel(f"{var_name}2")
            ax.set_title(f"{title_prefix} 2D Histogram")


def _plot_histogram_1d(
    ax, data: np.ndarray, Y: np.ndarray, var_name: str, title_prefix: str
) -> None:
    """Plot 1D histogram with class colors."""
    n_classes = len(np.unique(Y))
    colors = get_colormap_colors(n_classes)

    for i, digit in enumerate(np.unique(Y)):
        mask = Y == digit
        ax.hist(
            data[mask, 0],
            bins=HISTOGRAM_BINS,
            alpha=DEFAULT_ALPHA,
            label=f"Digit {digit}",
            color=colors[i],
        )

    ax.set_xlabel(f"{var_name}1")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} Histogram")
    ax.legend()


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
    _save_evolution_plot(
        epochs,
        mu_mean_per_dim_data,
        mu_std_per_dim_data,
        colors,
        n_dims,
        out_path,
        var_symbol="μ",
        marker="o",
        avg_marker="s",
    )


def _save_sigma_evolution_plot(
    epochs, sigma_mean_per_dim_data, sigma_std_per_dim_data, colors, n_dims, out_path
):
    """Save σ (std parameter) evolution plot."""
    _save_evolution_plot(
        epochs,
        sigma_mean_per_dim_data,
        sigma_std_per_dim_data,
        colors,
        n_dims,
        out_path,
        var_symbol="σ",
        marker="s",
        avg_marker="D",
    )


def _save_evolution_plot(
    epochs: list,
    mean_per_dim_data: np.ndarray,
    std_per_dim_data: np.ndarray,
    colors: list,
    n_dims: int,
    out_path: str,
    var_symbol: str,
    marker: str,
    avg_marker: str,
) -> None:
    """Save evolution plot for latent space parameters.

    Args:
        epochs: List of epoch numbers
        mean_per_dim_data: Mean values per dimension over epochs (n_epochs, n_dims)
        std_per_dim_data: Std values per dimension over epochs (n_epochs, n_dims)
        colors: List of colors for each dimension
        n_dims: Number of dimensions
        out_path: Output file path
        var_symbol: Variable symbol for labels (e.g., 'μ', 'σ')
        marker: Marker style for per-dimension lines
        avg_marker: Marker style for average line
    """
    fig, ax = plt.subplots(1, 1, figsize=EVOLUTION_FIGURE_SIZE)

    # Plot per-dimension means with error bars showing their std deviations
    for dim in range(n_dims):
        means = mean_per_dim_data[:, dim]
        stds = std_per_dim_data[:, dim]

        # Main line for the mean
        ax.plot(
            epochs,
            means,
            label=f"{var_symbol}{dim + 1}",
            color=colors[dim],
            linewidth=LINE_WIDTH_BOLD,
            marker=marker,
            markersize=MARKER_SIZE,
        )

        # Error band showing the std deviation around the mean
        ax.fill_between(
            epochs, means - stds, means + stds, color=colors[dim], alpha=FILL_ALPHA
        )

    # Add overall average as a bold line
    overall_means = np.mean(mean_per_dim_data, axis=1)
    overall_spread = np.std(mean_per_dim_data, axis=1)
    ax.plot(
        epochs,
        overall_means,
        "k-",
        linewidth=LINE_WIDTH_EXTRA_BOLD,
        marker=avg_marker,
        markersize=MARKER_SIZE_LARGE,
        label=f"{var_symbol}_avg (across dims)",
        zorder=10,
    )
    ax.fill_between(
        epochs,
        overall_means - overall_spread,
        overall_means + overall_spread,
        color="black",
        alpha=0.1,
        label="±1 std across dims",
    )

    ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(f"{var_symbol} Value", fontsize=LABEL_FONTSIZE)
    ax.set_title(
        f"Evolution of Latent {var_symbol} Parameters",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )

    _configure_legend(ax, n_dims)
    ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    save_figure(out_path)
    plt.close()
