"""Training loss curve visualizations."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .core import make_grouped_plot_path, save_figure

# Figure sizes
EPOCH_FIGURE_SIZE = (10, 6)
STEP_FIGURE_SIZE = (12, 6)
SCATTER_FIGURE_SIZE = (10, 8)

# Line styling
LINE_WIDTH = 2
LINE_WIDTH_THIN = 1
LINE_WIDTH_EDGE = 0.5
LINE_WIDTH_GRADIENT = 2

# Marker sizes
MARKER_SIZE_SMALL = 2
MARKER_SIZE_MEDIUM = 3
MARKER_SIZE_STANDARD = 4
MARKER_SIZE_LARGE = 6
MARKER_SIZE_SCATTER = 50
MARKER_SIZE_HIGHLIGHT_SMALL = 120
MARKER_SIZE_HIGHLIGHT_MEDIUM = 150
MARKER_SIZE_HIGHLIGHT_LARGE = 200

# Alpha values
ALPHA_DEFAULT = 0.8
ALPHA_HIGH = 0.9
ALPHA_MEDIUM = 0.6
ALPHA_GRID = 0.3

# Z-order for layering
ZORDER_HIGHLIGHT = 10
ZORDER_BEST = 15

# Colors
COLOR_LEARNING_RATE = "purple"
COLOR_RECONSTRUCTION = "orange"
COLOR_KL = "red"
COLOR_BEST = "gold"
COLOR_BEST_EDGE = "darkorange"
COLOR_START = "green"
COLOR_START_EDGE = "darkgreen"
COLOR_END = "red"
COLOR_END_EDGE = "darkred"

# Colormaps
CMAP_PROGRESS = "coolwarm"


def _extract_array(history: dict, key: str) -> np.ndarray:
    """Extract and convert history value to float array."""
    return np.asarray(history[key], dtype=float)


def _has_data(history: Optional[dict], key: str) -> bool:
    """Check if history has data for given key."""
    return history is not None and history.get(key) is not None


def _plot_metric(
    ax,
    x_data: np.ndarray,
    y_data: np.ndarray,
    label: str,
    color: Optional[str] = None,
    is_step: bool = False,
) -> None:
    """Plot a single metric line.

    Args:
        ax: Matplotlib axis
        x_data: X-axis data (epochs or steps)
        y_data: Y-axis data (metric values)
        label: Line label
        color: Line color (optional)
        is_step: Whether this is step-level data (affects marker size)
    """
    marker_size = MARKER_SIZE_MEDIUM if is_step else MARKER_SIZE_STANDARD
    line_width = LINE_WIDTH_THIN if is_step else LINE_WIDTH

    plot_kwargs = {"alpha": ALPHA_DEFAULT, "linewidth": line_width, "label": label}

    if color is not None:
        plot_kwargs["color"] = color

    if is_step:
        ax.plot(x_data, y_data, "o-", markersize=marker_size, **plot_kwargs)
    else:
        ax.plot(x_data, y_data, "-", **plot_kwargs)


def _add_best_epoch_marker(
    ax,
    test_epochs: np.ndarray,
    test_loss_arr: np.ndarray,
    test_recon_arr: np.ndarray,
    best_epoch: Optional[int],
) -> None:
    """Add marker for best epoch on ELBO plot.

    Args:
        ax: Matplotlib axis
        test_epochs: Test epoch array
        test_loss_arr: Test loss array (for ELBO calculation)
        test_recon_arr: Test reconstruction array (fallback)
        best_epoch: Best epoch number (if known)
    """
    # Highlight best epoch (lowest validation loss) if available
    if best_epoch is not None and best_epoch in test_epochs:
        best_epoch_idx = np.where(test_epochs == best_epoch)[0][0]
        ax.scatter(
            test_epochs[best_epoch_idx],
            -test_loss_arr[best_epoch_idx],
            marker="*",
            s=MARKER_SIZE_HIGHLIGHT_LARGE,
            color=COLOR_BEST,
            edgecolor=COLOR_BEST_EDGE,
            linewidth=LINE_WIDTH,
            alpha=ALPHA_HIGH,
            zorder=ZORDER_HIGHLIGHT,
            label=f"Best Model (Epoch {int(best_epoch)})",
        )
    else:
        # Fallback: Highlight epoch with lowest reconstruction error
        min_recon_idx = np.argmin(test_recon_arr)
        ax.scatter(
            test_epochs[min_recon_idx],
            -test_loss_arr[min_recon_idx],
            marker="*",
            s=MARKER_SIZE_HIGHLIGHT_LARGE,
            color=COLOR_BEST,
            edgecolor=COLOR_BEST_EDGE,
            linewidth=LINE_WIDTH,
            alpha=ALPHA_HIGH,
            zorder=ZORDER_HIGHLIGHT,
            label=f"Best Recon (Epoch {int(test_epochs[min_recon_idx])})",
        )


def _create_elbo_plot(
    train_epochs: np.ndarray,
    train_loss_arr: np.ndarray,
    test_epochs: Optional[np.ndarray],
    test_loss_arr: Optional[np.ndarray],
    test_recon_arr: Optional[np.ndarray],
    fig_dir: str,
    best_epoch: Optional[int],
) -> None:
    """Create ELBO plot with best epoch highlighting.

    Args:
        train_epochs: Training epoch array
        train_loss_arr: Training loss array
        test_epochs: Test epoch array (optional)
        test_loss_arr: Test loss array (optional)
        test_recon_arr: Test reconstruction array (optional, for fallback)
        fig_dir: Base directory for figures
        best_epoch: Best epoch number (optional)
    """
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)

    ax.plot(
        train_epochs,
        -train_loss_arr,
        "-",
        label="Train ELBO",
        alpha=ALPHA_DEFAULT,
        linewidth=LINE_WIDTH,
    )

    if test_epochs is not None and len(test_epochs) > 0:
        ax.plot(
            test_epochs,
            -test_loss_arr,
            "o-",
            label="Test ELBO",
            alpha=ALPHA_DEFAULT,
            markersize=MARKER_SIZE_STANDARD,
        )
        _add_best_epoch_marker(
            ax, test_epochs, test_loss_arr, test_recon_arr, best_epoch
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO (Higher is Better)")
    ax.legend()
    ax.grid(True, alpha=ALPHA_GRID)
    save_figure(make_grouped_plot_path(fig_dir, "training", "elbo", "epochs"))
    plt.close()


def _create_metric_plot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    train_label: str,
    test_label: str,
    fig_dir: str,
    filename: str,
    suffix: str,
    color: Optional[str] = None,
    use_log_scale: bool = False,
    is_step: bool = False,
) -> None:
    """Create and save a metric plot.

    Args:
        x_train: Training x-axis data
        y_train: Training y-axis data
        x_test: Test x-axis data (optional)
        y_test: Test y-axis data (optional)
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        train_label: Training data label
        test_label: Test data label
        fig_dir: Base directory for figures
        filename: Base filename
        suffix: Filename suffix (e.g., 'epochs' or 'steps')
        color: Line color (optional)
        use_log_scale: Whether to use log scale for y-axis
        is_step: Whether this is step-level data
    """
    figsize = STEP_FIGURE_SIZE if is_step else EPOCH_FIGURE_SIZE
    fig, ax = plt.subplots(figsize=figsize)

    # Plot training data
    _plot_metric(ax, x_train, y_train, train_label, color, is_step)

    # Plot test data if available
    if x_test is not None and len(x_test) > 0:
        marker_size = MARKER_SIZE_MEDIUM if is_step else MARKER_SIZE_STANDARD
        ax.plot(
            x_test,
            y_test,
            "o-",
            label=test_label,
            alpha=ALPHA_DEFAULT,
            markersize=marker_size,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=ALPHA_GRID)

    if use_log_scale:
        ax.set_yscale("log")

    save_figure(make_grouped_plot_path(fig_dir, "training", filename, suffix))
    plt.close()


def save_training_curves(
    train_history: dict,
    test_history: dict,
    train_step_history: dict,
    fig_dir: str,
    best_epoch: Optional[int] = None,
) -> None:
    """Save separate training curve plots for epoch-level and step-level data.

    Args:
        train_history: Training metrics by epoch
        test_history: Test/validation metrics by epoch
        train_step_history: Training metrics by step
        fig_dir: Base directory for saving figures
        best_epoch: Epoch with best validation performance (if available)
    """
    # Save epoch-level plots
    _save_epoch_training_curves(train_history, test_history, fig_dir, best_epoch)

    # Save step-level plots if available
    if _has_data(train_step_history, "step"):
        _save_step_training_curves(train_step_history, fig_dir)


def _save_epoch_training_curves(
    train_history: dict,
    test_history: dict,
    fig_dir: str,
    best_epoch: Optional[int] = None,
) -> None:
    """Save epoch-level training curves as separate plots.

    Args:
        train_history: Training metrics by epoch
        test_history: Test/validation metrics by epoch
        fig_dir: Base directory for figures
        best_epoch: Epoch with best validation performance (optional)
    """
    if not _has_data(train_history, "epoch"):
        return

    train_epochs = _extract_array(train_history, "epoch")
    train_loss_arr = _extract_array(train_history, "loss")
    train_recon_arr = _extract_array(train_history, "recon")
    train_kl_arr = _extract_array(train_history, "kl")

    # Extract test data if available
    test_epochs = None
    test_loss_arr = None
    test_recon_arr = None
    test_kl_arr = None

    if _has_data(test_history, "epoch"):
        test_epochs = _extract_array(test_history, "epoch")
        test_loss_arr = _extract_array(test_history, "loss")
        test_recon_arr = _extract_array(test_history, "recon")
        test_kl_arr = _extract_array(test_history, "kl")

    # Learning Rate plot (if available)
    if _has_data(train_history, "learning_rate"):
        train_lr_arr = _extract_array(train_history, "learning_rate")
        _create_metric_plot(
            train_epochs,
            train_lr_arr,
            None,
            None,
            "Epoch",
            "Learning Rate",
            "Learning Rate Schedule",
            "Learning Rate",
            "Test LR",
            fig_dir,
            "learning_rate",
            "epochs",
            color=COLOR_LEARNING_RATE,
            use_log_scale=True,
        )

    # ELBO plot with best epoch highlighting
    _create_elbo_plot(
        train_epochs,
        train_loss_arr,
        test_epochs,
        test_loss_arr,
        test_recon_arr,
        fig_dir,
        best_epoch,
    )

    # Reconstruction loss plot
    _create_metric_plot(
        train_epochs,
        train_recon_arr,
        test_epochs,
        test_recon_arr,
        "Epoch",
        "BCE Loss",
        "Reconstruction Loss",
        "Train BCE",
        "Test BCE",
        fig_dir,
        "reconstruction_loss",
        "epochs",
        color=COLOR_RECONSTRUCTION,
    )

    # KL divergence plot
    _create_metric_plot(
        train_epochs,
        train_kl_arr,
        test_epochs,
        test_kl_arr,
        "Epoch",
        "KL Loss",
        "KL Divergence",
        "Train KL",
        "Test KL",
        fig_dir,
        "kl_loss",
        "epochs",
        color=COLOR_KL,
    )

    # Loss scatter plot (BCE vs KL from test data colored by epoch)
    if test_epochs is not None and len(test_epochs) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        _plot_loss_scatter(ax, test_epochs, test_kl_arr, test_recon_arr)
        save_figure(
            make_grouped_plot_path(fig_dir, "training", "loss_scatter", "epochs")
        )
        plt.close()


def _save_step_training_curves(train_step_history: dict, fig_dir: str) -> None:
    """Save step-level training curves as separate plots.

    Args:
        train_step_history: Training metrics by step
        fig_dir: Base directory for figures
    """
    steps = _extract_array(train_step_history, "step")
    loss_arr = _extract_array(train_step_history, "loss")
    recon_arr = _extract_array(train_step_history, "recon")
    kl_arr = _extract_array(train_step_history, "kl")

    # Learning Rate step plot (if available)
    if _has_data(train_step_history, "learning_rate"):
        lr_arr = _extract_array(train_step_history, "learning_rate")
        _create_metric_plot(
            steps,
            lr_arr,
            None,
            None,
            "Training Step",
            "Learning Rate",
            "Learning Rate Schedule by Training Step",
            "Learning Rate",
            "Test LR",
            fig_dir,
            "learning_rate",
            "steps",
            color=COLOR_LEARNING_RATE,
            use_log_scale=True,
            is_step=True,
        )

    # ELBO step plot
    _create_metric_plot(
        steps,
        -loss_arr,
        None,
        None,
        "Training Step",
        "ELBO",
        "ELBO by Training Step (Higher is Better)",
        "Train ELBO",
        "Test ELBO",
        fig_dir,
        "elbo",
        "steps",
        is_step=True,
    )

    # Reconstruction loss step plot
    _create_metric_plot(
        steps,
        recon_arr,
        None,
        None,
        "Training Step",
        "BCE Loss",
        "Reconstruction Loss by Training Step",
        "Train BCE",
        "Test BCE",
        fig_dir,
        "reconstruction_loss",
        "steps",
        color=COLOR_RECONSTRUCTION,
        is_step=True,
    )

    # KL step plot
    _create_metric_plot(
        steps,
        kl_arr,
        None,
        None,
        "Training Step",
        "KL Loss",
        "KL Divergence by Training Step",
        "Train KL",
        "Test KL",
        fig_dir,
        "kl_loss",
        "steps",
        color=COLOR_KL,
        is_step=True,
    )


def _plot_loss_scatter(
    ax, test_epochs: np.ndarray, test_kl_arr: np.ndarray, test_recon_arr: np.ndarray
) -> None:
    """Plot loss scatter showing training trajectory in KL vs reconstruction space.

    Args:
        ax: Matplotlib axis
        test_epochs: Test epoch array
        test_kl_arr: Test KL divergence array
        test_recon_arr: Test reconstruction loss array
    """
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
        segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]

        lc = LineCollection(
            segments,
            cmap=CMAP_PROGRESS,
            alpha=ALPHA_MEDIUM,
            linewidth=LINE_WIDTH_GRADIENT,
        )
        lc.set_array(epoch_normalized[:-1])
        ax.add_collection(lc)

    # Scatter plot
    scatter = ax.scatter(
        test_kl_arr,
        test_recon_arr,
        c=epoch_normalized,
        cmap=CMAP_PROGRESS,
        alpha=ALPHA_DEFAULT,
        s=MARKER_SIZE_SCATTER,
        edgecolors="black",
        linewidth=LINE_WIDTH_EDGE,
    )
    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("BCE (Reconstruction)")
    ax.set_title("Test: BCE vs KL (training path)")
    ax.grid(True, alpha=ALPHA_GRID)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Training Progress (cool→warm)")

    # Add start marker
    ax.scatter(
        test_kl_arr[0],
        test_recon_arr[0],
        marker="D",
        s=MARKER_SIZE_HIGHLIGHT_SMALL,
        color=COLOR_START,
        edgecolor=COLOR_START_EDGE,
        linewidth=LINE_WIDTH,
        alpha=ALPHA_HIGH,
        zorder=ZORDER_HIGHLIGHT,
        label="Start",
    )

    # Add end marker if multiple epochs
    if len(test_epochs) > 1:
        ax.scatter(
            test_kl_arr[-1],
            test_recon_arr[-1],
            marker="*",
            s=MARKER_SIZE_HIGHLIGHT_MEDIUM,
            color=COLOR_END,
            edgecolor=COLOR_END_EDGE,
            linewidth=LINE_WIDTH,
            alpha=ALPHA_HIGH,
            zorder=ZORDER_HIGHLIGHT,
            label="End",
        )

    # Highlight point with lowest reconstruction error
    min_recon_idx = np.argmin(test_recon_arr)
    ax.scatter(
        test_kl_arr[min_recon_idx],
        test_recon_arr[min_recon_idx],
        marker="*",
        s=MARKER_SIZE_HIGHLIGHT_LARGE,
        color=COLOR_BEST,
        edgecolor=COLOR_BEST_EDGE,
        linewidth=LINE_WIDTH,
        alpha=ALPHA_HIGH,
        zorder=ZORDER_BEST,
        label=f"Best Recon (Epoch {int(test_epochs[min_recon_idx])})",
    )

    # Add legend
    if len(test_epochs) > 1:
        ax.legend(loc="upper right", framealpha=ALPHA_HIGH)
