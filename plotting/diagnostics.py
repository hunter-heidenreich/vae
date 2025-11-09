"""Training diagnostics and analysis plots."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .core import DEFAULT_DPI


def save_training_curves(train_history: dict, test_history: dict, out_path: str):
    """Save training and test loss curves from training history (epoch-based) including loss scatter plot."""
    # Use epochs instead of steps
    train_epochs = np.array([])
    test_epochs = np.array([])

    if train_history and train_history.get("epoch"):
        train_epochs = np.asarray(train_history["epoch"], dtype=float)

    if len(train_epochs) == 0:
        return

    # Convert training histories to numpy arrays
    train_loss_arr = np.asarray(train_history["loss"], dtype=float)
    train_recon_arr = np.asarray(train_history["recon"], dtype=float)
    train_kl_arr = np.asarray(train_history["kl"], dtype=float)

    # Convert test histories to numpy arrays (if available)
    test_loss_arr = np.array([])
    test_recon_arr = np.array([])
    test_kl_arr = np.array([])

    if test_history and test_history.get("epoch"):
        test_epochs = np.asarray(test_history["epoch"], dtype=float)
        test_loss_arr = np.asarray(test_history["loss"], dtype=float)
        test_recon_arr = np.asarray(test_history["recon"], dtype=float)
        test_kl_arr = np.asarray(test_history["kl"], dtype=float)

    # Create figure and axes in a 1x4 grid
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # First plot: Total loss comparison
    axs[0].plot(
        train_epochs, -train_loss_arr, "-", label="Train Loss", alpha=0.8, linewidth=2
    )
    if len(test_epochs) > 0:
        axs[0].plot(
            test_epochs,
            -test_loss_arr,
            "o-",
            label="Test Loss",
            alpha=0.8,
            markersize=4,
        )
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("ELBO")
    axs[0].set_title("ELBO (Higher is Better)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Second plot: Reconstruction loss comparison
    axs[1].plot(
        train_epochs, train_recon_arr, "-", label="Train BCE", alpha=0.8, linewidth=2
    )
    if len(test_epochs) > 0:
        axs[1].plot(
            test_epochs, test_recon_arr, "o-", label="Test BCE", alpha=0.8, markersize=4
        )
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("BCE Loss")
    axs[1].set_title("Reconstruction Loss")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Third plot: KL divergence comparison
    axs[2].plot(
        train_epochs, train_kl_arr, "-", label="Train KL", alpha=0.8, linewidth=2
    )
    if len(test_epochs) > 0:
        axs[2].plot(
            test_epochs, test_kl_arr, "o-", label="Test KL", alpha=0.8, markersize=4
        )
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("KL Loss")
    axs[2].set_title("KL Divergence")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    # Fourth plot: Loss scatter plot (BCE vs KL from test data colored by epoch)
    if len(test_epochs) > 0:
        _plot_loss_scatter(axs[3], test_epochs, test_kl_arr, test_recon_arr)
    else:
        # If no test data, leave the fourth subplot empty or add a placeholder
        axs[3].text(
            0.5,
            0.5,
            "No test data available\nfor loss scatter plot",
            ha="center",
            va="center",
            transform=axs[3].transAxes,
            fontsize=12,
            alpha=0.7,
        )
        axs[3].set_title("Test Loss Scatter")

    # Save and close the figure manually
    plt.tight_layout()
    plt.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close()


def save_gradient_diagnostics(train_history: dict, out_path: str):
    """Save a 2x2 summary of gradient dynamics."""

    required_keys = [
        "epoch",
        "recon_grad_norm",
        "kl_grad_norm",
        "grad_norm_realized",
        "grad_norm_unclipped",
        "recon_kl_cosine",
        "recon_contrib",
        "kl_contrib",
    ]

    if not all(train_history.get(key) for key in required_keys):
        print("Skipping gradient diagnostics plot: Missing required history keys.")
        return

    epochs = np.asarray(train_history["epoch"], dtype=float)
    if len(epochs) == 0:
        return

    # --- Load Data ---
    recon_norm = np.asarray(train_history["recon_grad_norm"])
    kl_norm = np.asarray(train_history["kl_grad_norm"])
    realized_norm = np.asarray(train_history["grad_norm_realized"])
    unclipped_norm = np.asarray(train_history["grad_norm_unclipped"])

    recon_kl_cosine = np.asarray(train_history["recon_kl_cosine"])

    recon_contrib = np.asarray(train_history["recon_contrib"])
    kl_contrib = np.asarray(train_history["kl_contrib"])

    # --- Create Plot ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Gradient Diagnostics", fontsize=16)

    # --- 1. Top-Left: Gradient Norms ---
    _plot_gradient_norms(
        axs[0, 0], epochs, recon_norm, kl_norm, realized_norm, unclipped_norm
    )

    # --- 2. Top-Right: Gradient Alignment (Interference) ---
    _plot_gradient_alignment(axs[0, 1], epochs, recon_kl_cosine)

    # --- 3. Bottom-Left: Relative Contribution to Direction ---
    _plot_gradient_contributions(axs[1, 0], epochs, recon_contrib, kl_contrib)

    # --- 4. Bottom-Right: Effective Realized Magnitudes ---
    _plot_effective_magnitudes(
        axs[1, 1], epochs, realized_norm, recon_contrib, kl_contrib
    )

    # --- Save ---
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))  # Adjust for suptitle
    plt.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close()


def save_kl_diagnostics_combined(
    test_history: dict, out_path: str, active_threshold: float = 0.1
):
    """Save a combined figure with multiple KL per dimension diagnostics."""
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

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # Top row plots
    _plot_kl_bar_chart(fig, kl_matrix, latest_kl_per_dim, active_threshold, 1)
    _plot_active_units_over_time(fig, epochs, active_units_over_time, latent_dim, 2)
    _plot_kl_heatmap(fig, epochs, kl_matrix, latent_dim, 3)

    # Bottom row plots
    _plot_kl_statistics_over_time(fig, epochs, kl_matrix, 4)
    _plot_kl_distribution_histogram(fig, epochs, kl_matrix, active_threshold, 5)
    _plot_cumulative_kl_contribution(fig, latest_kl_per_dim, latent_dim, 6)

    # Add summary text
    active_units = np.sum(latest_kl_per_dim >= active_threshold)
    fig.suptitle(
        f"KL Diagnostics: {active_units}/{latent_dim} Active Units (threshold={active_threshold})",
        fontsize=14,
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    plt.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close()


# Helper functions for cleaner code organization


def _plot_loss_scatter(ax, test_epochs, test_kl_arr, test_recon_arr):
    """Plot loss scatter subplot."""
    # Normalize epochs for color mapping (cool to warm)
    if len(test_epochs) > 1:
        epoch_normalized = (test_epochs - test_epochs.min()) / (
            test_epochs.max() - test_epochs.min()
        )
    else:
        epoch_normalized = np.array([0.5])  # Single point gets middle color

    # Draw connecting line with color gradient to show training path
    if len(test_epochs) > 1:
        points = np.column_stack([test_kl_arr, test_recon_arr])
        segments = []
        for i in range(len(points) - 1):
            segments.append([points[i], points[i + 1]])

        # Create line collection with color gradient
        lc = LineCollection(segments, cmap="coolwarm", alpha=0.6, linewidth=2)
        lc.set_array(epoch_normalized[:-1])  # Color by starting epoch of each segment
        ax.add_collection(lc)

    # Scatter plot with reduced size for cleaner look
    scatter = ax.scatter(
        test_kl_arr,
        test_recon_arr,
        c=epoch_normalized,
        cmap="coolwarm",
        alpha=0.8,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("BCE (Reconstruction)")
    ax.set_title("Test: BCE vs KL (training path)")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Training Progress (cool→warm)")

    # Add special markers for start and end points
    if len(test_epochs) > 0:
        ax.scatter(
            test_kl_arr[0],
            test_recon_arr[0],
            marker="D",
            s=120,
            color="green",
            edgecolor="darkgreen",
            linewidth=2,
            alpha=0.9,
            zorder=10,
            label="Start",
        )

        if len(test_epochs) > 1:
            ax.scatter(
                test_kl_arr[-1],
                test_recon_arr[-1],
                marker="*",
                s=150,
                color="red",
                edgecolor="darkred",
                linewidth=2,
                alpha=0.9,
                zorder=10,
                label="End",
            )

    # Add legend
    if len(test_epochs) > 1:
        ax.legend(loc="upper right", framealpha=0.9)


def _plot_gradient_norms(
    ax, epochs, recon_norm, kl_norm, realized_norm, unclipped_norm
):
    """Plot gradient norms subplot."""
    ax.plot(
        epochs,
        recon_norm,
        "-o",
        label="Recon Norm",
        markersize=3,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        epochs, kl_norm, "-s", label="KL Norm", markersize=3, alpha=0.8, color="red"
    )
    ax.plot(
        epochs,
        realized_norm,
        "-^",
        label="Realized Total Norm",
        markersize=3,
        alpha=0.8,
        color="green",
    )
    # Show unclipped norm only if it's different from realized (i.e., clipping happened)
    if np.any(unclipped_norm != realized_norm):
        ax.plot(
            epochs,
            unclipped_norm,
            "--",
            label="Unclipped Total Norm",
            alpha=0.5,
            color="gray",
        )
    ax.set_title("Gradient L2 Norms")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # Use log scale for norms, they can vary wildly


def _plot_gradient_alignment(ax, epochs, recon_kl_cosine):
    """Plot gradient alignment subplot."""
    ax.plot(
        epochs,
        recon_kl_cosine,
        "-d",
        label="Recon vs KL Cosine",
        markersize=3,
        color="purple",
    )
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.7, label="Orthogonal")
    ax.axhline(
        y=-1,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Opposite (Max Interference)",
    )
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.7, label="Aligned")
    ax.set_title("Gradient Alignment (Interference)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim([-1.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_gradient_contributions(ax, epochs, recon_contrib, kl_contrib):
    """Plot gradient contributions subplot."""
    ax.plot(
        epochs,
        recon_contrib,
        "-o",
        label="Recon Contribution",
        markersize=3,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        epochs,
        kl_contrib,
        "-s",
        label="KL Contribution",
        markersize=3,
        alpha=0.8,
        color="red",
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Balanced")
    ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.5)
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.5)
    ax.set_title("Relative Contribution to Update Direction")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Contribution (Sum = 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_effective_magnitudes(ax, epochs, realized_norm, recon_contrib, kl_contrib):
    """Plot effective magnitudes subplot."""
    effective_recon = realized_norm * recon_contrib
    effective_kl = realized_norm * kl_contrib
    ax.plot(
        epochs,
        effective_recon,
        "-o",
        label="Effective Recon",
        markersize=3,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        epochs,
        effective_kl,
        "-s",
        label="Effective KL",
        markersize=3,
        alpha=0.8,
        color="red",
    )
    ax.set_title("Effective Realized Gradient Magnitudes")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Effective L2 Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_kl_bar_chart(
    fig, kl_matrix, latest_kl_per_dim, active_threshold, subplot_idx
):
    """Plot KL per dimension bar chart."""
    latent_dim = kl_matrix.shape[1]
    ax = plt.subplot(2, 3, subplot_idx)
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


def _plot_active_units_over_time(
    fig, epochs, active_units_over_time, latent_dim, subplot_idx
):
    """Plot active units over time."""
    ax = plt.subplot(2, 3, subplot_idx)
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


def _plot_kl_heatmap(fig, epochs, kl_matrix, latent_dim, subplot_idx):
    """Plot KL evolution heatmap."""
    ax = plt.subplot(2, 3, subplot_idx)
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


def _plot_kl_statistics_over_time(fig, epochs, kl_matrix, subplot_idx):
    """Plot KL statistics over time."""
    ax = plt.subplot(2, 3, subplot_idx)
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


def _plot_kl_distribution_histogram(
    fig, epochs, kl_matrix, active_threshold, subplot_idx
):
    """Plot KL distribution histogram."""
    ax = plt.subplot(2, 3, subplot_idx)

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


def _plot_cumulative_kl_contribution(fig, latest_kl_per_dim, latent_dim, subplot_idx):
    """Plot cumulative KL contribution."""
    ax = plt.subplot(2, 3, subplot_idx)

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
