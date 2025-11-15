"""KL divergence per-dimension analysis and diagnostic plots."""

import matplotlib.pyplot as plt
import numpy as np

from .core import make_grouped_plot_path, save_figure

# Styling constants
STANDARD_FIGURE_SIZE = (12, 8)
HEATMAP_FIGURE_SIZE = (14, 8)
GRID_ALPHA = 0.3
DEFAULT_ALPHA = 0.8
LINE_ALPHA = 0.7
FILL_ALPHA = 0.3

# Plot styling
LINE_WIDTH = 2
LINE_WIDTH_THIN = 1.5
MARKER_SIZE = 4
MARKER_SIZE_SMALL = 3
EDGE_LINE_WIDTH = 0.5
BAR_LABEL_FONTSIZE = 8

# Histogram settings
HISTOGRAM_BINS = 15
MAX_DIMS_FOR_LABELS = 20

# Tick settings
MAX_EPOCHS_FULL_TICKS = 15
HEATMAP_TICK_DIVISIONS = 8

# Color scheme
COLOR_ACTIVE = "steelblue"
COLOR_INACTIVE = "red"
COLOR_THRESHOLD = "red"
COLOR_GRAY = "gray"
COLOR_PURPLE = "purple"
COLOR_EARLY = "lightblue"
COLOR_MIDDLE = "orange"
COLOR_FINAL = "red"


def _validate_kl_history(test_history: dict) -> bool:
    """Validate that test history contains required KL data.

    Args:
        test_history: History dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    if not test_history.get("kl_per_dim"):
        print("Skipping KL diagnostics: Missing 'kl_per_dim' in test history")
        return False
    if not test_history.get("epoch"):
        print("Skipping KL diagnostics: Missing 'epoch' in test history")
        return False
    return True


def _configure_common_axis(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    add_grid: bool = True,
    add_legend: bool = True,
) -> None:
    """Apply common axis configuration."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if add_grid:
        ax.grid(True, alpha=GRID_ALPHA)
    if add_legend:
        ax.legend()


def save_kl_diagnostics_separate(
    test_history: dict, fig_dir: str, active_threshold: float = 0.1
) -> None:
    """Save separate KL per dimension diagnostic plots."""
    if not _validate_kl_history(test_history):
        return

    epochs = np.asarray(test_history["epoch"], dtype=float)
    kl_per_dim_history = test_history["kl_per_dim"]

    if len(epochs) == 0 or not kl_per_dim_history:
        print("Skipping KL diagnostics: Empty epoch or KL data")
        return

    # Prepare data
    kl_matrix = np.array(
        [np.asarray(kl_per_dim, dtype=float) for kl_per_dim in kl_per_dim_history]
    )
    latent_dim = kl_matrix.shape[1]
    latest_kl_per_dim = kl_matrix[-1]

    # Calculate active units over time
    active_units_over_time = np.array(
        [np.sum(kl_array >= active_threshold) for kl_array in kl_matrix]
    )

    # Create separate plots
    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)
    _plot_kl_bar_chart(ax, latest_kl_per_dim, active_threshold, latent_dim)
    save_figure(make_grouped_plot_path(fig_dir, "kl_analysis", "per_dimension", "bar"))
    plt.close()

    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)
    _plot_active_units_over_time(ax, epochs, active_units_over_time, latent_dim)
    save_figure(
        make_grouped_plot_path(fig_dir, "kl_analysis", "active_units", "over_time")
    )
    plt.close()

    fig, ax = plt.subplots(figsize=HEATMAP_FIGURE_SIZE)
    _plot_kl_heatmap(ax, epochs, kl_matrix, latent_dim)
    save_figure(make_grouped_plot_path(fig_dir, "kl_analysis", "evolution", "heatmap"))
    plt.close()

    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)
    _plot_kl_statistics_over_time(ax, epochs, kl_matrix)
    save_figure(
        make_grouped_plot_path(fig_dir, "kl_analysis", "statistics", "over_time")
    )
    plt.close()

    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)
    _plot_kl_distribution_histogram(ax, epochs, kl_matrix, active_threshold)
    save_figure(
        make_grouped_plot_path(fig_dir, "kl_analysis", "distribution", "histogram")
    )
    plt.close()

    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)
    _plot_cumulative_kl_contribution(ax, latest_kl_per_dim, latent_dim)
    save_figure(
        make_grouped_plot_path(fig_dir, "kl_analysis", "cumulative", "contribution")
    )
    plt.close()


def _plot_kl_bar_chart(
    ax, latest_kl_per_dim: np.ndarray, active_threshold: float, latent_dim: int
) -> None:
    """Plot KL per dimension bar chart."""
    dims = np.arange(latent_dim)
    colors = [
        COLOR_INACTIVE if kl < active_threshold else COLOR_ACTIVE
        for kl in latest_kl_per_dim
    ]
    bars = ax.bar(
        dims,
        latest_kl_per_dim,
        color=colors,
        alpha=DEFAULT_ALPHA,
        edgecolor="black",
        linewidth=EDGE_LINE_WIDTH,
    )
    ax.axhline(
        y=active_threshold,
        color=COLOR_THRESHOLD,
        linestyle="--",
        alpha=LINE_ALPHA,
        label=f"Threshold = {active_threshold}",
    )
    ax.set_xticks(dims)
    ax.set_xticklabels([f"z{i + 1}" for i in dims])

    # Add value labels on bars if not too many dimensions
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

    _configure_common_axis(
        ax, "Final KL per Dimension", "Latent Dimension", "KL Divergence"
    )


def _plot_active_units_over_time(
    ax, epochs: np.ndarray, active_units_over_time: np.ndarray, latent_dim: int
) -> None:
    """Plot active units over time."""
    ax.plot(
        epochs,
        active_units_over_time,
        "o-",
        color=COLOR_ACTIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
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
    _configure_common_axis(ax, "Active Units Over Time", "Epoch", "Active Units")


def _set_heatmap_epoch_ticks(ax, epochs: np.ndarray) -> None:
    """Configure x-axis ticks for heatmap based on number of epochs."""
    if len(epochs) <= MAX_EPOCHS_FULL_TICKS:
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels([f"{int(e)}" for e in epochs])
    else:
        step = max(1, len(epochs) // HEATMAP_TICK_DIVISIONS)
        tick_indices = range(0, len(epochs), step)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f"{int(epochs[i])}" for i in tick_indices])


def _plot_kl_heatmap(
    ax, epochs: np.ndarray, kl_matrix: np.ndarray, latent_dim: int
) -> None:
    """Plot KL evolution heatmap."""
    im = ax.imshow(kl_matrix.T, aspect="auto", cmap="viridis", origin="lower")

    _set_heatmap_epoch_ticks(ax, epochs)
    ax.set_yticks(range(latent_dim))
    ax.set_yticklabels([f"z{i + 1}" for i in range(latent_dim)])

    _configure_common_axis(
        ax,
        "KL Evolution Heatmap",
        "Epoch",
        "Latent Dimension",
        add_grid=False,
        add_legend=False,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("KL Divergence")


def _plot_kl_statistics_over_time(
    ax, epochs: np.ndarray, kl_matrix: np.ndarray
) -> None:
    """Plot KL statistics over time."""
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
        alpha=LINE_ALPHA,
        linewidth=LINE_WIDTH_THIN,
        markersize=MARKER_SIZE_SMALL,
    )
    ax.plot(
        epochs,
        min_kl,
        "^-",
        label="Min KL",
        alpha=LINE_ALPHA,
        linewidth=LINE_WIDTH_THIN,
        markersize=MARKER_SIZE_SMALL,
    )

    _configure_common_axis(ax, "KL Statistics Over Time", "Epoch", "KL Divergence")


def _plot_kl_distribution_histogram(
    ax, epochs: np.ndarray, kl_matrix: np.ndarray, active_threshold: float
) -> None:
    """Plot KL distribution histogram."""
    n_epochs = len(epochs)
    if n_epochs >= 3:
        epoch_indices = [0, n_epochs // 2, -1]
        epoch_labels = ["Early", "Middle", "Final"]
        colors = [COLOR_EARLY, COLOR_MIDDLE, COLOR_FINAL]

        for idx, label, color in zip(epoch_indices, epoch_labels, colors):
            ax.hist(
                kl_matrix[idx],
                bins=HISTOGRAM_BINS,
                alpha=0.6,
                label=f"{label} (E{int(epochs[idx])})",
                color=color,
                edgecolor="black",
                linewidth=EDGE_LINE_WIDTH,
            )
    else:
        ax.hist(
            kl_matrix[-1],
            bins=HISTOGRAM_BINS,
            alpha=LINE_ALPHA,
            label=f"Final (E{int(epochs[-1])})",
            color=COLOR_FINAL,
            edgecolor="black",
            linewidth=EDGE_LINE_WIDTH,
        )

    ax.axvline(
        x=active_threshold,
        color=COLOR_THRESHOLD,
        linestyle="--",
        alpha=DEFAULT_ALPHA,
        label=f"Threshold = {active_threshold}",
    )

    _configure_common_axis(ax, "Distribution of KL Values", "KL Divergence", "Count")


def _plot_cumulative_kl_contribution(
    ax, latest_kl_per_dim: np.ndarray, latent_dim: int
) -> None:
    """Plot cumulative KL contribution."""
    # Sort dimensions by final KL value and compute cumulative contribution
    final_kl_sorted = np.sort(latest_kl_per_dim)[::-1]  # Descending order
    cumsum_kl = np.cumsum(final_kl_sorted)
    total_kl = np.sum(latest_kl_per_dim)
    cumsum_percentage = 100.0 * cumsum_kl / total_kl

    ax.plot(
        range(1, latent_dim + 1),
        cumsum_percentage,
        "o-",
        color=COLOR_PURPLE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
    )
    ax.axhline(
        y=90,
        color=COLOR_GRAY,
        linestyle="--",
        alpha=LINE_ALPHA,
        label="90% of total KL",
    )
    ax.axhline(
        y=95, color=COLOR_GRAY, linestyle=":", alpha=LINE_ALPHA, label="95% of total KL"
    )
    ax.set_xticks(range(1, latent_dim + 1))
    ax.set_ylim((0, 105))

    _configure_common_axis(
        ax,
        "Cumulative KL Contribution",
        "Number of Top Dimensions",
        "Cumulative KL (%)",
    )
