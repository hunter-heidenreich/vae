"""Training loss curve visualizations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .core import make_grouped_plot_path, save_figure


def save_training_curves(
    train_history: dict, test_history: dict, train_step_history: dict, fig_dir: str
):
    """Save separate training curve plots for epoch-level and step-level data."""

    # Save epoch-level plots
    _save_epoch_training_curves(train_history, test_history, fig_dir)

    # Save step-level plots if available
    if train_step_history and train_step_history.get("step"):
        _save_step_training_curves(train_step_history, fig_dir)


def _save_epoch_training_curves(train_history: dict, test_history: dict, fig_dir: str):
    """Save epoch-level training curves as separate plots."""
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

    # Create separate plots

    # 1. Learning Rate plot (if available)
    if train_history.get("learning_rate"):
        train_lr_arr = np.asarray(train_history["learning_rate"], dtype=float)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            train_epochs,
            train_lr_arr,
            "-",
            label="Learning Rate",
            alpha=0.8,
            linewidth=2,
            color="purple",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")  # Use log scale for better visualization
        save_figure(
            make_grouped_plot_path(fig_dir, "training", "learning_rate", "epochs")
        )
        plt.close()

    # 2. ELBO plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        train_epochs, -train_loss_arr, "-", label="Train ELBO", alpha=0.8, linewidth=2
    )
    if len(test_epochs) > 0:
        ax.plot(
            test_epochs,
            -test_loss_arr,
            "o-",
            label="Test ELBO",
            alpha=0.8,
            markersize=4,
        )

        # Highlight epoch with lowest reconstruction error
        min_recon_idx = np.argmin(test_recon_arr)
        ax.scatter(
            test_epochs[min_recon_idx],
            -test_loss_arr[min_recon_idx],
            marker="*",
            s=200,
            color="gold",
            edgecolor="darkorange",
            linewidth=2,
            alpha=0.9,
            zorder=10,
            label=f"Best Recon (Epoch {int(test_epochs[min_recon_idx])})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO (Higher is Better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(make_grouped_plot_path(fig_dir, "training", "elbo", "epochs"))
    plt.close()

    # 3. Reconstruction loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        train_epochs, train_recon_arr, "-", label="Train BCE", alpha=0.8, linewidth=2
    )
    if len(test_epochs) > 0:
        ax.plot(
            test_epochs, test_recon_arr, "o-", label="Test BCE", alpha=0.8, markersize=4
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Reconstruction Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(
        make_grouped_plot_path(fig_dir, "training", "reconstruction_loss", "epochs")
    )
    plt.close()

    # 4. KL divergence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_epochs, train_kl_arr, "-", label="Train KL", alpha=0.8, linewidth=2)
    if len(test_epochs) > 0:
        ax.plot(
            test_epochs, test_kl_arr, "o-", label="Test KL", alpha=0.8, markersize=4
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Loss")
    ax.set_title("KL Divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(make_grouped_plot_path(fig_dir, "training", "kl_loss", "epochs"))
    plt.close()

    # 5. Loss scatter plot (BCE vs KL from test data colored by epoch)
    if len(test_epochs) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        _plot_loss_scatter(ax, test_epochs, test_kl_arr, test_recon_arr)
        save_figure(
            make_grouped_plot_path(fig_dir, "training", "loss_scatter", "epochs")
        )
        plt.close()


def _save_step_training_curves(train_step_history: dict, fig_dir: str):
    """Save step-level training curves as separate plots."""
    if not train_step_history.get("step"):
        return

    steps = np.asarray(train_step_history["step"], dtype=float)
    loss_arr = np.asarray(train_step_history["loss"], dtype=float)
    recon_arr = np.asarray(train_step_history["recon"], dtype=float)
    kl_arr = np.asarray(train_step_history["kl"], dtype=float)

    # 1. Learning Rate step plot (if available)
    if train_step_history.get("learning_rate"):
        lr_arr = np.asarray(train_step_history["learning_rate"], dtype=float)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            steps, lr_arr, "o-", alpha=0.8, markersize=2, linewidth=1, color="purple"
        )
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule by Training Step")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")  # Use log scale for better visualization
        save_figure(
            make_grouped_plot_path(fig_dir, "training", "learning_rate", "steps")
        )
        plt.close()

    # 2. ELBO step plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, -loss_arr, "o-", alpha=0.8, markersize=3, linewidth=1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO by Training Step (Higher is Better)")
    ax.grid(True, alpha=0.3)
    save_figure(make_grouped_plot_path(fig_dir, "training", "elbo", "steps"))
    plt.close()

    # 3. Reconstruction loss step plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        steps, recon_arr, "o-", alpha=0.8, markersize=3, linewidth=1, color="orange"
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Reconstruction Loss by Training Step")
    ax.grid(True, alpha=0.3)
    save_figure(
        make_grouped_plot_path(fig_dir, "training", "reconstruction_loss", "steps")
    )
    plt.close()

    # 4. KL step plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, kl_arr, "o-", alpha=0.8, markersize=3, linewidth=1, color="red")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("KL Loss")
    ax.set_title("KL Divergence by Training Step")
    ax.grid(True, alpha=0.3)
    save_figure(make_grouped_plot_path(fig_dir, "training", "kl_loss", "steps"))
    plt.close()


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

        # Highlight point with lowest reconstruction error
        min_recon_idx = np.argmin(test_recon_arr)
        ax.scatter(
            test_kl_arr[min_recon_idx],
            test_recon_arr[min_recon_idx],
            marker="*",
            s=200,
            color="gold",
            edgecolor="darkorange",
            linewidth=2,
            alpha=0.9,
            zorder=15,
            label=f"Best Recon (Epoch {int(test_epochs[min_recon_idx])})",
        )

    # Add legend
    if len(test_epochs) > 1:
        ax.legend(loc="upper right", framealpha=0.9)
