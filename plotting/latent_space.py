"""Latent space visualization plots."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from .constants import (
    ALPHA_HIGH,
    DEFAULT_ALPHA,
    DEFAULT_CMAP,
    FIGURE_SIZE_COMBINED,
    FIGURE_SIZE_NARROW,
    FIGURE_SIZE_STANDARD,
    FILL_ALPHA,
    GRID_ALPHA,
    LABEL_FONTSIZE,
    LINE_WIDTH,
    LINE_WIDTH_BOLD,
    LINE_WIDTH_EXTRA_BOLD,
    MARKER_SIZE_LARGE,
    MARKER_SIZE_STANDARD,
    MAX_DIMS_SINGLE_COLUMN,
    SCATTER_SIZE,
    SMALL_FONTSIZE,
    TITLE_FONTSIZE,
)
from .core import compute_histogram_bins, figure_context, make_plot_path, save_figure
from .data import apply_pca_if_needed, get_colormap_colors


def save_latent_marginals(Z: np.ndarray, out_path: str) -> None:
    _save_marginals(Z, out_path, "z", "Latent marginal KDEs")


def save_logvar_marginals(LogVar: np.ndarray, out_path: str) -> None:
    _save_marginals(LogVar, out_path, "logvar", "Log Variance marginal KDEs")


def _save_marginals(
    data: np.ndarray, out_path: str, var_prefix: str, title: str
) -> None:
    latent_dim = data.shape[1]

    with figure_context(FIGURE_SIZE_NARROW, out_path):
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
    _save_combined_figure(Z, Y, out_path, "z", "Latent")


def save_logvar_combined_figure(
    LogVar: np.ndarray, Y: np.ndarray, out_path: str
) -> None:
    _save_combined_figure(LogVar, Y, out_path, "logvar", "Log Variance")


def _save_combined_figure(
    data: np.ndarray, Y: np.ndarray, out_path: str, var_name: str, title_prefix: str
) -> None:
    if data.shape[0] == 0:
        raise ValueError(f"Empty {title_prefix.lower()} array provided")

    latent_dim = data.shape[1]
    data_plot, pca = apply_pca_if_needed(data, 2)

    fig, axs = plt.subplots(1, 2, figsize=FIGURE_SIZE_COMBINED)

    if data.shape[1] == 1:
        _plot_histogram_1d(axs[0], data, Y, var_name, title_prefix)
    else:
        _plot_scatter_2d(axs[0], data_plot, Y, pca, latent_dim, var_name, title_prefix)

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
    if latent_dim == 1:
        bins = compute_histogram_bins(data[:, 0])
        sns.histplot(
            data[:, 0],
            kde=True,
            stat="density",
            bins=bins,
            alpha=ALPHA_HIGH - 0.2,
            color="steelblue",
            ax=ax,
        )
        ax.set_xlabel(f"{var_name}1")
        ax.set_ylabel("Density")
        ax.set_title(f"{title_prefix} 1D Distribution")
        ax.grid(True, alpha=GRID_ALPHA)
    else:
        bins_x = compute_histogram_bins(data_plot[:, 0])
        bins_y = compute_histogram_bins(data_plot[:, 1])
        hist = ax.hist2d(
            data_plot[:, 0], data_plot[:, 1], bins=[bins_x, bins_y], cmap="magma"
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
    n_classes = len(np.unique(Y))
    colors = get_colormap_colors(n_classes)
    bins = compute_histogram_bins(data[:, 0])

    for i, digit in enumerate(np.unique(Y)):
        mask = Y == digit
        ax.hist(
            data[mask, 0],
            bins=bins,
            alpha=DEFAULT_ALPHA,
            label=f"Digit {digit}",
            color=colors[i],
        )

    ax.set_xlabel(f"{var_name}1")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} Histogram")
    ax.legend()


def save_latent_evolution_plots(val_history: dict, fig_dir: str) -> None:
    if "mu_mean_per_dim" not in val_history or not val_history["mu_mean_per_dim"]:
        return

    epochs = val_history.get(
        "epoch", list(range(1, len(val_history["mu_mean_per_dim"]) + 1))
    )
    mu_mean_per_dim_data = np.array(val_history["mu_mean_per_dim"])
    mu_std_per_dim_data = np.array(val_history["mu_std_per_dim"])

    has_sigma_data = (
        "sigma_mean_per_dim" in val_history and val_history["sigma_mean_per_dim"]
    )
    if has_sigma_data:
        sigma_mean_per_dim_data = np.array(val_history["sigma_mean_per_dim"])
        sigma_std_per_dim_data = np.array(val_history["sigma_std_per_dim"])

    n_epochs, n_dims = mu_mean_per_dim_data.shape
    colors = get_colormap_colors(n_dims)

    mu_out_path = make_plot_path(fig_dir, "mu_evolution", group="latent_space")
    _save_evolution_plot(
        epochs,
        mu_mean_per_dim_data,
        mu_std_per_dim_data,
        colors,
        n_dims,
        mu_out_path,
        "μ",
        "o",
        "s",
    )

    if has_sigma_data:
        sigma_out_path = make_plot_path(
            fig_dir, "sigma_evolution", group="latent_space"
        )
        _save_evolution_plot(
            epochs,
            sigma_mean_per_dim_data,
            sigma_std_per_dim_data,
            colors,
            n_dims,
            sigma_out_path,
            "σ",
            "s",
            "D",
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
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE_STANDARD)
    for dim in range(n_dims):
        means = mean_per_dim_data[:, dim]
        stds = std_per_dim_data[:, dim]

        ax.plot(
            epochs,
            means,
            label=f"{var_symbol}{dim + 1}",
            color=colors[dim],
            linewidth=LINE_WIDTH_BOLD,
            marker=marker,
            markersize=MARKER_SIZE_STANDARD,
        )
        ax.fill_between(
            epochs, means - stds, means + stds, color=colors[dim], alpha=FILL_ALPHA
        )

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

    legend_kwargs = {"bbox_to_anchor": (1.05, 1), "loc": "upper left"}
    if n_dims > MAX_DIMS_SINGLE_COLUMN:
        legend_kwargs.update({"fontsize": SMALL_FONTSIZE, "ncol": 2})
    ax.legend(**legend_kwargs)
    ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    save_figure(out_path)
    plt.close()
