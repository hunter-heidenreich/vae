"""Parameter magnitude and change diagnostic plots."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .core import make_grouped_plot_path, save_figure

# Styling constants
EPOCH_FIGURE_SIZE = (12, 8)
EPOCH_MARKER_SIZE = 3
GRID_ALPHA = 0.3
BEST_EPOCH_MARKER_SIZE = 150
BEST_EPOCH_MARKER_ALPHA = 0.9

# Color scheme
COLOR_ENCODER = "blue"
COLOR_DECODER = "orange"
COLOR_TOTAL = "purple"


def _validate_history_keys(history: dict, required_keys: list[str], name: str) -> bool:
    """Validate that history contains all required keys with non-empty data.

    Args:
        history: History dictionary to validate
        required_keys: List of required key names
        name: Name of the diagnostic for error message

    Returns:
        True if valid, False otherwise
    """
    missing_keys = [key for key in required_keys if not history.get(key)]
    if missing_keys:
        print(f"Skipping {name}: Missing keys {missing_keys}")
        return False
    return True


def _configure_common_axis(
    ax, title: str, xlabel: str, ylabel: str, use_log: bool = False
):
    """Apply common axis configuration."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    if use_log:
        ax.set_yscale("log")


def _add_best_epoch_marker(
    ax,
    epochs: np.ndarray,
    values: np.ndarray,
    best_epoch: Optional[int],
    label_prefix: str = "Best",
):
    """Add a marker for the best epoch on a plot."""
    if best_epoch is not None and best_epoch in epochs:
        best_idx = np.where(epochs == best_epoch)[0][0]
        ax.scatter(
            epochs[best_idx],
            values[best_idx],
            marker="*",
            s=BEST_EPOCH_MARKER_SIZE,
            color="gold",
            edgecolor="darkorange",
            linewidth=1.5,
            alpha=BEST_EPOCH_MARKER_ALPHA,
            zorder=10,
            label=f"{label_prefix} (Epoch {int(best_epoch)})",
        )


def save_parameter_diagnostics(
    train_history: dict,
    val_history: dict,
    fig_dir: str,
    best_epoch: Optional[int] = None,
):
    """Save parameter magnitude and change diagnostics."""
    _save_parameter_norm_diagnostics(train_history, val_history, fig_dir, best_epoch)
    _save_parameter_change_diagnostics(train_history, fig_dir, best_epoch)


def _save_parameter_norm_diagnostics(
    train_history: dict,
    val_history: dict,
    fig_dir: str,
    best_epoch: Optional[int] = None,
):
    """Save parameter norm diagnostics using validation data if available."""
    required_keys = [
        "epoch",
        "encoder_param_norm",
        "decoder_param_norm",
        "total_param_norm",
    ]

    # Check both histories
    train_has_params = all(train_history.get(key) for key in required_keys)
    val_has_params = all(val_history.get(key) for key in required_keys)

    if not (train_has_params or val_has_params):
        print(f"Skipping parameter norm diagnostics: Missing keys {required_keys}")
        return

    # Prefer validation data (cleaner, no training noise)
    history_to_use = val_history if val_has_params else train_history
    data_source = "validation" if val_has_params else "training"

    x_values = np.asarray(history_to_use["epoch"], dtype=float)
    if len(x_values) == 0:
        return

    data = {
        "encoder_norm": np.asarray(history_to_use["encoder_param_norm"]),
        "decoder_norm": np.asarray(history_to_use["decoder_param_norm"]),
        "total_norm": np.asarray(history_to_use["total_param_norm"]),
    }

    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_parameter_norms(
        ax,
        x_values,
        data["encoder_norm"],
        data["decoder_norm"],
        data["total_norm"],
        data_source,
        EPOCH_MARKER_SIZE,
    )
    _add_best_epoch_marker(ax, x_values, data["total_norm"], best_epoch, "Best Model")
    save_figure(make_grouped_plot_path(fig_dir, "parameters", "norms", "epochs"))
    plt.close()


def _save_parameter_change_diagnostics(
    train_history: dict, fig_dir: str, best_epoch: Optional[int] = None
):
    """Save parameter change diagnostics (training data only)."""
    required_keys = [
        "epoch",
        "encoder_param_change_norm",
        "decoder_param_change_norm",
        "total_param_change_norm",
        "encoder_param_change_rel",
        "decoder_param_change_rel",
        "total_param_change_rel",
    ]

    if not _validate_history_keys(
        train_history, required_keys, "parameter change diagnostics"
    ):
        return

    x_values = np.asarray(train_history["epoch"], dtype=float)
    if len(x_values) == 0:
        return

    # Load absolute and relative changes
    data = {
        "encoder_change_norm": np.asarray(train_history["encoder_param_change_norm"]),
        "decoder_change_norm": np.asarray(train_history["decoder_param_change_norm"]),
        "total_change_norm": np.asarray(train_history["total_param_change_norm"]),
        "encoder_change_rel": np.asarray(train_history["encoder_param_change_rel"]),
        "decoder_change_rel": np.asarray(train_history["decoder_param_change_rel"]),
        "total_change_rel": np.asarray(train_history["total_param_change_rel"]),
    }

    # 1. Absolute Parameter Changes
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_parameter_changes_absolute(
        ax,
        x_values,
        data["encoder_change_norm"],
        data["decoder_change_norm"],
        data["total_change_norm"],
        EPOCH_MARKER_SIZE,
    )
    save_figure(
        make_grouped_plot_path(fig_dir, "parameters", "changes_absolute", "epochs")
    )
    plt.close()

    # 2. Relative Parameter Changes
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_parameter_changes_relative(
        ax,
        x_values,
        data["encoder_change_rel"],
        data["decoder_change_rel"],
        data["total_change_rel"],
        EPOCH_MARKER_SIZE,
    )
    save_figure(
        make_grouped_plot_path(fig_dir, "parameters", "changes_relative", "epochs")
    )
    plt.close()


def _plot_parameter_norms(
    ax,
    x_values: np.ndarray,
    encoder_norm: np.ndarray,
    decoder_norm: np.ndarray,
    total_norm: np.ndarray,
    data_source: str,
    marker_size: int,
):
    """Plot parameter norms over time."""
    ax.plot(
        x_values,
        encoder_norm,
        "-o",
        label="Encoder Parameters",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_ENCODER,
    )
    ax.plot(
        x_values,
        decoder_norm,
        "-s",
        label="Decoder Parameters",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_DECODER,
    )
    ax.plot(
        x_values,
        total_norm,
        "-^",
        label="Total Parameters",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_TOTAL,
    )
    _configure_common_axis(
        ax, f"Parameter L2 Norms ({data_source})", "Epoch", "L2 Norm", use_log=True
    )


def _plot_parameter_changes_absolute(
    ax,
    x_values: np.ndarray,
    encoder_change_norm: np.ndarray,
    decoder_change_norm: np.ndarray,
    total_change_norm: np.ndarray,
    marker_size: int,
):
    """Plot absolute parameter changes over epochs."""
    ax.plot(
        x_values,
        encoder_change_norm,
        "-o",
        label="Encoder Change",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_ENCODER,
    )
    ax.plot(
        x_values,
        decoder_change_norm,
        "-s",
        label="Decoder Change",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_DECODER,
    )
    ax.plot(
        x_values,
        total_change_norm,
        "-^",
        label="Total Change",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_TOTAL,
    )
    _configure_common_axis(
        ax, "Absolute Parameter Changes", "Epoch", "L2 Norm of Change", use_log=True
    )


def _plot_parameter_changes_relative(
    ax,
    x_values: np.ndarray,
    encoder_change_rel: np.ndarray,
    decoder_change_rel: np.ndarray,
    total_change_rel: np.ndarray,
    marker_size: int,
):
    """Plot relative parameter changes over epochs."""
    ax.plot(
        x_values,
        encoder_change_rel,
        "-o",
        label="Encoder Relative Change",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_ENCODER,
    )
    ax.plot(
        x_values,
        decoder_change_rel,
        "-s",
        label="Decoder Relative Change",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_DECODER,
    )
    ax.plot(
        x_values,
        total_change_rel,
        "-^",
        label="Total Relative Change",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_TOTAL,
    )
    _configure_common_axis(
        ax, "Relative Parameter Changes", "Epoch", "Relative Change (Δ / Current)"
    )
