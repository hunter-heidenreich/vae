"""Training loss curve visualizations."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .constants import (
    ALPHA_HIGH,
    ALPHA_MEDIUM,
    CMAP_PROGRESS,
    COLOR_BEST,
    COLOR_BEST_EDGE,
    COLOR_END,
    COLOR_END_EDGE,
    COLOR_KL,
    COLOR_LEARNING_RATE,
    COLOR_RECONSTRUCTION,
    COLOR_START,
    COLOR_START_EDGE,
    DEFAULT_ALPHA,
    FIGURE_SIZE_STANDARD,
    FIGURE_SIZE_WIDE,
    GRID_ALPHA,
    LINE_WIDTH,
    LINE_WIDTH_EDGE,
    MARKER_SIZE_HIGHLIGHT,
    MARKER_SIZE_MEDIUM,
    MARKER_SIZE_STANDARD,
    ZORDER_BEST,
    ZORDER_HIGHLIGHT,
)
from .core import (
    add_best_epoch_marker,
    extract_history_data,
    make_plot_path,
    save_figure,
)


def _has_data(history: Optional[dict], key: str) -> bool:
    return history is not None and history.get(key) is not None


def _plot_metric(
    ax,
    x_data: np.ndarray,
    y_data: np.ndarray,
    label: str,
    color: Optional[str] = None,
    is_step: bool = False,
) -> None:
    marker_size = MARKER_SIZE_MEDIUM if is_step else MARKER_SIZE_STANDARD
    plot_kwargs = {"alpha": DEFAULT_ALPHA, "linewidth": LINE_WIDTH, "label": label}
    if color:
        plot_kwargs["color"] = color

    if is_step:
        ax.plot(x_data, y_data, "o-", markersize=marker_size, **plot_kwargs)
    else:
        ax.plot(x_data, y_data, "-", **plot_kwargs)


def _add_elbo_best_marker(
    ax,
    test_epochs: np.ndarray,
    test_loss_arr: np.ndarray,
    test_recon_arr: np.ndarray,
    best_epoch: Optional[int],
) -> None:
    if best_epoch is not None and best_epoch in test_epochs:
        add_best_epoch_marker(ax, test_epochs, -test_loss_arr, best_epoch, "Best Model")
    else:
        min_recon_idx = np.argmin(test_recon_arr)
        add_best_epoch_marker(
            ax, test_epochs, -test_loss_arr, test_epochs[min_recon_idx], "Best Recon"
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
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

    ax.plot(
        train_epochs,
        -train_loss_arr,
        "-",
        label="Train ELBO",
        alpha=DEFAULT_ALPHA,
        linewidth=LINE_WIDTH,
    )

    if test_epochs is not None and len(test_epochs) > 0:
        ax.plot(
            test_epochs,
            -test_loss_arr,
            "o-",
            label="Test ELBO",
            alpha=DEFAULT_ALPHA,
            markersize=MARKER_SIZE_STANDARD,
        )
        _add_elbo_best_marker(
            ax, test_epochs, test_loss_arr, test_recon_arr, best_epoch
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO (Higher is Better)")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    save_figure(make_plot_path(fig_dir, "elbo", "epochs", "training"))
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
    figsize = FIGURE_SIZE_WIDE if is_step else FIGURE_SIZE_STANDARD
    fig, ax = plt.subplots(figsize=figsize)

    _plot_metric(ax, x_train, y_train, train_label, color, is_step)

    if x_test is not None and len(x_test) > 0:
        marker_size = MARKER_SIZE_MEDIUM if is_step else MARKER_SIZE_STANDARD
        ax.plot(
            x_test,
            y_test,
            "o-",
            label=test_label,
            alpha=DEFAULT_ALPHA,
            markersize=marker_size,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)

    if use_log_scale:
        ax.set_yscale("log")

    save_figure(make_plot_path(fig_dir, filename, suffix, "training"))
    plt.close()


def save_training_curves(
    train_history: dict,
    test_history: dict,
    train_step_history: dict,
    fig_dir: str,
    best_epoch: Optional[int] = None,
) -> None:
    _save_epoch_training_curves(train_history, test_history, fig_dir, best_epoch)

    if _has_data(train_step_history, "step"):
        _save_step_training_curves(train_step_history, fig_dir)


def _save_epoch_training_curves(
    train_history: dict,
    test_history: dict,
    fig_dir: str,
    best_epoch: Optional[int] = None,
) -> None:
    if not _has_data(train_history, "epoch"):
        return

    train_epochs, train_loss_arr, train_recon_arr, train_kl_arr = extract_history_data(
        train_history, "epoch", "loss", "recon", "kl"
    )

    test_epochs = test_loss_arr = test_recon_arr = test_kl_arr = None
    if _has_data(test_history, "epoch"):
        test_epochs, test_loss_arr, test_recon_arr, test_kl_arr = extract_history_data(
            test_history, "epoch", "loss", "recon", "kl"
        )

    if _has_data(train_history, "learning_rate"):
        (train_lr_arr,) = extract_history_data(train_history, "learning_rate")
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

    _create_elbo_plot(
        train_epochs,
        train_loss_arr,
        test_epochs,
        test_loss_arr,
        test_recon_arr,
        fig_dir,
        best_epoch,
    )

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

    if test_epochs is not None and len(test_epochs) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        _plot_loss_scatter(ax, test_epochs, test_kl_arr, test_recon_arr)
        save_figure(make_plot_path(fig_dir, "loss_scatter", "epochs", "training"))
        plt.close()


def _save_step_training_curves(train_step_history: dict, fig_dir: str) -> None:
    steps, loss_arr, recon_arr, kl_arr = extract_history_data(
        train_step_history, "step", "loss", "recon", "kl"
    )

    if _has_data(train_step_history, "learning_rate"):
        (lr_arr,) = extract_history_data(train_step_history, "learning_rate")
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
    if len(test_epochs) > 1:
        epoch_normalized = (test_epochs - test_epochs.min()) / (
            test_epochs.max() - test_epochs.min()
        )
    else:
        epoch_normalized = np.array([0.5])

    if len(test_epochs) > 1:
        points = np.column_stack([test_kl_arr, test_recon_arr])
        segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
        lc = LineCollection(
            segments, cmap=CMAP_PROGRESS, alpha=ALPHA_MEDIUM, linewidth=LINE_WIDTH
        )
        lc.set_array(epoch_normalized[:-1])
        ax.add_collection(lc)

    scatter = ax.scatter(
        test_kl_arr,
        test_recon_arr,
        c=epoch_normalized,
        cmap=CMAP_PROGRESS,
        alpha=DEFAULT_ALPHA,
        s=50,
        edgecolors="black",
        linewidth=LINE_WIDTH_EDGE,
    )
    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("BCE (Reconstruction)")
    ax.set_title("Test: BCE vs KL (training path)")
    ax.grid(True, alpha=GRID_ALPHA)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Training Progress (cool→warm)")

    ax.scatter(
        test_kl_arr[0],
        test_recon_arr[0],
        marker="D",
        s=120,
        color=COLOR_START,
        edgecolor=COLOR_START_EDGE,
        linewidth=LINE_WIDTH,
        alpha=ALPHA_HIGH,
        zorder=ZORDER_HIGHLIGHT,
        label="Start",
    )

    if len(test_epochs) > 1:
        ax.scatter(
            test_kl_arr[-1],
            test_recon_arr[-1],
            marker="*",
            s=MARKER_SIZE_HIGHLIGHT,
            color=COLOR_END,
            edgecolor=COLOR_END_EDGE,
            linewidth=LINE_WIDTH,
            alpha=ALPHA_HIGH,
            zorder=ZORDER_HIGHLIGHT,
            label="End",
        )

    min_recon_idx = np.argmin(test_recon_arr)
    ax.scatter(
        test_kl_arr[min_recon_idx],
        test_recon_arr[min_recon_idx],
        marker="*",
        s=200,
        color=COLOR_BEST,
        edgecolor=COLOR_BEST_EDGE,
        linewidth=LINE_WIDTH,
        alpha=ALPHA_HIGH,
        zorder=ZORDER_BEST,
        label=f"Best Recon (Epoch {int(test_epochs[min_recon_idx])})",
    )

    if len(test_epochs) > 1:
        ax.legend(loc="upper right", framealpha=ALPHA_HIGH)
