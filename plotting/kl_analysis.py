"""KL divergence per-dimension analysis and diagnostic plots."""

import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    ALPHA_HIGH,
    BAR_LABEL_FONTSIZE,
    COLOR_ACTIVE,
    COLOR_EARLY,
    COLOR_FINAL,
    COLOR_GRAY,
    COLOR_INACTIVE,
    COLOR_MIDDLE,
    COLOR_PURPLE,
    COLOR_THRESHOLD,
    DEFAULT_ALPHA,
    FIGURE_SIZE_STANDARD,
    FIGURE_SIZE_WIDE,
    FILL_ALPHA,
    GRID_ALPHA,
    HEATMAP_TICK_DIVISIONS,
    LINE_WIDTH,
    LINE_WIDTH_EDGE,
    LINE_WIDTH_THIN,
    MARKER_SIZE_SMALL,
    MARKER_SIZE_STANDARD,
    MAX_DIMS_FOR_LABELS,
    MAX_EPOCHS_FULL_TICKS,
)
from .core import compute_histogram_bins, make_plot_path, save_figure


def save_kl_diagnostics_separate(
    test_history: dict, fig_dir: str, active_threshold: float = 0.1
) -> None:
    if not all(test_history.get(k) for k in ["epoch", "kl_per_dim"]):
        return

    epochs = np.asarray(test_history["epoch"], dtype=float)
    kl_per_dim_history = test_history["kl_per_dim"]

    if len(epochs) == 0 or not kl_per_dim_history:
        return

    kl_matrix = np.array(
        [np.asarray(kl_per_dim, dtype=float) for kl_per_dim in kl_per_dim_history]
    )
    latent_dim = kl_matrix.shape[1]
    latest_kl_per_dim = kl_matrix[-1]
    active_units_over_time = np.array(
        [np.sum(kl_array >= active_threshold) for kl_array in kl_matrix]
    )

    # Create separate plots
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_kl_bar_chart(ax, latest_kl_per_dim, active_threshold, latent_dim)
    save_figure(make_plot_path(fig_dir, "per_dimension", "bar", "kl_analysis"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_active_units_over_time(ax, epochs, active_units_over_time, latent_dim)
    save_figure(make_plot_path(fig_dir, "active_units", "over_time", "kl_analysis"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    _plot_kl_heatmap(ax, epochs, kl_matrix, latent_dim)
    save_figure(make_plot_path(fig_dir, "evolution", "heatmap", "kl_analysis"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_kl_statistics_over_time(ax, epochs, kl_matrix)
    save_figure(make_plot_path(fig_dir, "statistics", "over_time", "kl_analysis"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_kl_distribution_histogram(ax, epochs, kl_matrix, active_threshold)
    save_figure(make_plot_path(fig_dir, "distribution", "histogram", "kl_analysis"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_cumulative_kl_contribution(ax, latest_kl_per_dim, latent_dim)
    save_figure(make_plot_path(fig_dir, "cumulative", "contribution", "kl_analysis"))
    plt.close()


def _plot_kl_bar_chart(
    ax, latest_kl_per_dim: np.ndarray, active_threshold: float, latent_dim: int
) -> None:
    dims = np.arange(latent_dim)
    colors = [
        COLOR_ACTIVE if kl >= active_threshold else COLOR_INACTIVE
        for kl in latest_kl_per_dim
    ]
    bars = ax.bar(
        dims,
        latest_kl_per_dim,
        color=colors,
        alpha=DEFAULT_ALPHA,
        edgecolor="black",
        linewidth=LINE_WIDTH_EDGE,
    )
    ax.axhline(
        y=active_threshold,
        color=COLOR_THRESHOLD,
        linestyle="--",
        alpha=ALPHA_HIGH - 0.2,
        label=f"Threshold = {active_threshold}",
    )
    ax.set_xticks(dims)
    ax.set_xticklabels([f"z{i + 1}" for i in dims])

    if latent_dim <= MAX_DIMS_FOR_LABELS:
        max_kl = max(latest_kl_per_dim)
        for bar, val in zip(bars, latest_kl_per_dim):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max_kl * 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_FONTSIZE,
            )

    ax.set_title("Final KL per Dimension")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("KL Divergence")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_active_units_over_time(
    ax, epochs: np.ndarray, active_units_over_time: np.ndarray, latent_dim: int
) -> None:
    ax.plot(
        epochs,
        active_units_over_time,
        "o-",
        color=COLOR_ACTIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE_STANDARD,
        label="Active Units",
    )
    ax.axhline(
        y=latent_dim,
        color=COLOR_GRAY,
        linestyle="--",
        alpha=FILL_ALPHA,
        label=f"Max ({latent_dim})",
    )
    ax.set_ylim((0, latent_dim + 0.5))
    ax.set_title("Active Units Over Time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Units")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_kl_heatmap(
    ax, epochs: np.ndarray, kl_matrix: np.ndarray, latent_dim: int
) -> None:
    im = ax.imshow(kl_matrix.T, aspect="auto", cmap="viridis", origin="lower")

    if len(epochs) <= MAX_EPOCHS_FULL_TICKS:
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels([f"{int(e)}" for e in epochs])
    else:
        step = max(1, len(epochs) // HEATMAP_TICK_DIVISIONS)
        tick_indices = range(0, len(epochs), step)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f"{int(epochs[i])}" for i in tick_indices])
    ax.set_yticks(range(latent_dim))
    ax.set_yticklabels([f"z{i + 1}" for i in range(latent_dim)])

    ax.set_title("KL Evolution Heatmap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Latent Dimension")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("KL Divergence")


def _plot_kl_statistics_over_time(
    ax, epochs: np.ndarray, kl_matrix: np.ndarray
) -> None:
    mean_kl = np.mean(kl_matrix, axis=1)
    std_kl = np.std(kl_matrix, axis=1)
    max_kl = np.max(kl_matrix, axis=1)
    min_kl = np.min(kl_matrix, axis=1)

    ax.plot(
        epochs,
        mean_kl,
        "o-",
        label="Mean KL",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE_SMALL,
    )
    ax.fill_between(
        epochs, mean_kl - std_kl, mean_kl + std_kl, alpha=FILL_ALPHA, label="±1 std"
    )
    ax.plot(
        epochs,
        max_kl,
        "s-",
        label="Max KL",
        alpha=ALPHA_HIGH - 0.2,
        linewidth=LINE_WIDTH_THIN,
        markersize=MARKER_SIZE_SMALL,
    )
    ax.plot(
        epochs,
        min_kl,
        "^-",
        label="Min KL",
        alpha=ALPHA_HIGH - 0.2,
        linewidth=LINE_WIDTH_THIN,
        markersize=MARKER_SIZE_SMALL,
    )

    ax.set_title("KL Statistics Over Time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_kl_distribution_histogram(
    ax, epochs: np.ndarray, kl_matrix: np.ndarray, active_threshold: float
) -> None:
    n_epochs = len(epochs)
    bins = compute_histogram_bins(kl_matrix.flatten())

    if n_epochs >= 3:
        epoch_indices = [0, n_epochs // 2, -1]
        epoch_labels = ["Early", "Middle", "Final"]
        colors = [COLOR_EARLY, COLOR_MIDDLE, COLOR_FINAL]

        for idx, label, color in zip(epoch_indices, epoch_labels, colors):
            ax.hist(
                kl_matrix[idx],
                bins=bins,
                alpha=0.6,
                label=f"{label} (E{int(epochs[idx])})",
                color=color,
                edgecolor="black",
                linewidth=LINE_WIDTH_EDGE,
            )
    else:
        ax.hist(
            kl_matrix[-1],
            bins=bins,
            alpha=ALPHA_HIGH - 0.2,
            label=f"Final (E{int(epochs[-1])})",
            color=COLOR_FINAL,
            edgecolor="black",
            linewidth=LINE_WIDTH_EDGE,
        )

    ax.axvline(
        x=active_threshold,
        color=COLOR_THRESHOLD,
        linestyle="--",
        alpha=DEFAULT_ALPHA,
        label=f"Threshold = {active_threshold}",
    )

    ax.set_title("Distribution of KL Values")
    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_cumulative_kl_contribution(
    ax, latest_kl_per_dim: np.ndarray, latent_dim: int
) -> None:
    final_kl_sorted = np.sort(latest_kl_per_dim)[::-1]
    cumsum_kl = np.cumsum(final_kl_sorted)
    total_kl = np.sum(latest_kl_per_dim)
    cumsum_percentage = 100.0 * cumsum_kl / total_kl

    ax.plot(
        range(1, latent_dim + 1),
        cumsum_percentage,
        "o-",
        color=COLOR_PURPLE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE_STANDARD,
    )
    ax.axhline(
        y=90,
        color=COLOR_GRAY,
        linestyle="--",
        alpha=ALPHA_HIGH - 0.2,
        label="90% of total KL",
    )
    ax.axhline(
        y=95,
        color=COLOR_GRAY,
        linestyle=":",
        alpha=ALPHA_HIGH - 0.2,
        label="95% of total KL",
    )
    ax.set_xticks(range(1, latent_dim + 1))
    ax.set_ylim((0, 105))

    ax.set_title("Cumulative KL Contribution")
    ax.set_xlabel("Number of Top Dimensions")
    ax.set_ylabel("Cumulative KL (%)")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
