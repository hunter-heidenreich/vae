"""KL divergence per-dimension analysis and diagnostic plots."""

import matplotlib.pyplot as plt
import numpy as np

from .core import DEFAULT_DPI


def save_kl_diagnostics_separate(
    test_history: dict, fig_dir: str, active_threshold: float = 0.1
):
    """Save separate KL per dimension diagnostic plots."""
    if not test_history.get("kl_per_dim") or not test_history.get("epoch"):
        return

    epochs = np.asarray(test_history["epoch"], dtype=float)
    kl_per_dim_history = test_history["kl_per_dim"]

    if len(epochs) == 0 or not kl_per_dim_history:
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

    # 1. KL bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_kl_bar_chart_standalone(ax, kl_matrix, latest_kl_per_dim, active_threshold)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/kl_per_dimension_bar.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 2. Active units over time
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_active_units_over_time_standalone(
        ax, epochs, active_units_over_time, latent_dim
    )
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/kl_active_units_over_time.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 3. KL heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_kl_heatmap_standalone(ax, epochs, kl_matrix, latent_dim)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/kl_evolution_heatmap.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 4. KL statistics over time
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_kl_statistics_over_time_standalone(ax, epochs, kl_matrix)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/kl_statistics_over_time.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 5. KL distribution histogram
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_kl_distribution_histogram_standalone(ax, epochs, kl_matrix, active_threshold)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/kl_distribution_histogram.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 6. Cumulative KL contribution
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_cumulative_kl_contribution_standalone(ax, latest_kl_per_dim, latent_dim)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/kl_cumulative_contribution.webp", dpi=DEFAULT_DPI)
    plt.close()


def _plot_kl_bar_chart_standalone(ax, kl_matrix, latest_kl_per_dim, active_threshold):
    """Plot KL per dimension bar chart as standalone plot."""
    latent_dim = kl_matrix.shape[1]
    dims = np.arange(latent_dim)
    colors = [
        "red" if kl < active_threshold else "steelblue" for kl in latest_kl_per_dim
    ]
    bars = ax.bar(
        dims,
        latest_kl_per_dim,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(
        y=active_threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold = {active_threshold}",
    )
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Final KL per Dimension")
    ax.set_xticks(dims)
    ax.set_xticklabels([f"z{i + 1}" for i in dims])
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    # Add value labels on bars if not too many dimensions
    if latent_dim <= 20:
        for bar, val in zip(bars, latest_kl_per_dim):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(latest_kl_per_dim) * 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def _plot_active_units_over_time_standalone(
    ax, epochs, active_units_over_time, latent_dim
):
    """Plot active units over time as standalone plot."""
    ax.plot(
        epochs,
        active_units_over_time,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=4,
        label="Active Units",
    )
    ax.axhline(
        y=latent_dim,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Max ({latent_dim})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Units")
    ax.set_title("Active Units Over Time")
    ax.set_ylim((0, latent_dim + 0.5))
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_kl_heatmap_standalone(ax, epochs, kl_matrix, latent_dim):
    """Plot KL evolution heatmap as standalone plot."""
    im = ax.imshow(kl_matrix.T, aspect="auto", cmap="viridis", origin="lower")

    # Set ticks for heatmap
    if len(epochs) <= 15:
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels([f"{int(e)}" for e in epochs])
    else:
        step = max(1, len(epochs) // 8)
        tick_indices = range(0, len(epochs), step)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f"{int(epochs[i])}" for i in tick_indices])

    ax.set_yticks(range(latent_dim))
    ax.set_yticklabels([f"z{i + 1}" for i in range(latent_dim)])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Latent Dimension")
    ax.set_title("KL Evolution Heatmap")

    # Add colorbar for heatmap
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("KL Divergence")


def _plot_kl_statistics_over_time_standalone(ax, epochs, kl_matrix):
    """Plot KL statistics over time as standalone plot."""
    mean_kl = np.mean(kl_matrix, axis=1)
    std_kl = np.std(kl_matrix, axis=1)
    max_kl = np.max(kl_matrix, axis=1)
    min_kl = np.min(kl_matrix, axis=1)

    ax.plot(epochs, mean_kl, "o-", label="Mean KL", linewidth=2, markersize=3)
    ax.fill_between(
        epochs, mean_kl - std_kl, mean_kl + std_kl, alpha=0.3, label="±1 std"
    )
    ax.plot(
        epochs, max_kl, "s-", label="Max KL", alpha=0.7, linewidth=1.5, markersize=3
    )
    ax.plot(
        epochs, min_kl, "^-", label="Min KL", alpha=0.7, linewidth=1.5, markersize=3
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Statistics Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_kl_distribution_histogram_standalone(ax, epochs, kl_matrix, active_threshold):
    """Plot KL distribution histogram as standalone plot."""
    # Plot histogram for a few key epochs
    n_epochs = len(epochs)
    if n_epochs >= 3:
        epoch_indices = [0, n_epochs // 2, -1]
        epoch_labels = ["Early", "Middle", "Final"]
        colors_hist = ["lightblue", "orange", "red"]

        for idx, label, color in zip(epoch_indices, epoch_labels, colors_hist):
            ax.hist(
                kl_matrix[idx],
                bins=15,
                alpha=0.6,
                label=f"{label} (E{int(epochs[idx])})",
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
    else:
        ax.hist(
            kl_matrix[-1],
            bins=15,
            alpha=0.7,
            label=f"Final (E{int(epochs[-1])})",
            color="red",
            edgecolor="black",
            linewidth=0.5,
        )

    ax.axvline(
        x=active_threshold,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Threshold = {active_threshold}",
    )
    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of KL Values")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_cumulative_kl_contribution_standalone(ax, latest_kl_per_dim, latent_dim):
    """Plot cumulative KL contribution as standalone plot."""
    # Sort dimensions by final KL value and compute cumulative contribution
    final_kl_sorted_idx = np.argsort(latest_kl_per_dim)[::-1]  # Descending order
    final_kl_sorted = latest_kl_per_dim[final_kl_sorted_idx]
    cumsum_kl = np.cumsum(final_kl_sorted)
    total_kl = np.sum(latest_kl_per_dim)
    cumsum_percentage = 100.0 * cumsum_kl / total_kl

    ax.plot(
        range(1, latent_dim + 1),
        cumsum_percentage,
        "o-",
        color="purple",
        linewidth=2,
        markersize=4,
    )
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.7, label="90% of total KL")
    ax.axhline(y=95, color="gray", linestyle=":", alpha=0.7, label="95% of total KL")
    ax.set_xlabel("Number of Top Dimensions")
    ax.set_ylabel("Cumulative KL (%)")
    ax.set_title("Cumulative KL Contribution")
    ax.set_xticks(range(1, latent_dim + 1))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim((0, 105))
