"""Parameter magnitude and change diagnostic plots."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    COLOR_DECODER,
    COLOR_ENCODER,
    COLOR_TOTAL,
    FIGURE_SIZE_STANDARD,
    GRID_ALPHA,
    MARKER_SIZE_MEDIUM,
)
from .core import add_best_epoch_marker, make_plot_path, save_figure


def save_parameter_diagnostics(
    train_history: dict,
    val_history: dict,
    fig_dir: str,
    best_epoch: Optional[int] = None,
) -> None:
    _save_parameter_norm_diagnostics(train_history, val_history, fig_dir, best_epoch)
    _save_parameter_change_diagnostics(train_history, fig_dir, best_epoch)


def _save_parameter_norm_diagnostics(
    train_history: dict,
    val_history: dict,
    fig_dir: str,
    best_epoch: Optional[int] = None,
) -> None:
    required_keys = [
        "epoch",
        "encoder_param_norm",
        "decoder_param_norm",
        "total_param_norm",
    ]

    train_has_params = all(train_history.get(key) for key in required_keys)
    val_has_params = all(val_history.get(key) for key in required_keys)

    if not (train_has_params or val_has_params):
        return

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

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_parameter_norms(
        ax,
        x_values,
        data["encoder_norm"],
        data["decoder_norm"],
        data["total_norm"],
        data_source,
        MARKER_SIZE_MEDIUM,
    )
    add_best_epoch_marker(ax, x_values, data["total_norm"], best_epoch, "Best Model")
    save_figure(make_plot_path(fig_dir, "norms", "epochs", "parameters"))
    plt.close()


def _save_parameter_change_diagnostics(
    train_history: dict, fig_dir: str, best_epoch: Optional[int] = None
) -> None:
    required_keys = [
        "epoch",
        "encoder_param_change_norm",
        "decoder_param_change_norm",
        "total_param_change_norm",
        "encoder_param_change_rel",
        "decoder_param_change_rel",
        "total_param_change_rel",
    ]

    if not all(train_history.get(k) for k in required_keys):
        return

    x_values = np.asarray(train_history["epoch"], dtype=float)
    if len(x_values) == 0:
        return

    data = {
        "encoder_change_norm": np.asarray(train_history["encoder_param_change_norm"]),
        "decoder_change_norm": np.asarray(train_history["decoder_param_change_norm"]),
        "total_change_norm": np.asarray(train_history["total_param_change_norm"]),
        "encoder_change_rel": np.asarray(train_history["encoder_param_change_rel"]),
        "decoder_change_rel": np.asarray(train_history["decoder_param_change_rel"]),
        "total_change_rel": np.asarray(train_history["total_param_change_rel"]),
    }

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_parameter_changes_absolute(
        ax,
        x_values,
        data["encoder_change_norm"],
        data["decoder_change_norm"],
        data["total_change_norm"],
        MARKER_SIZE_MEDIUM,
    )
    save_figure(make_plot_path(fig_dir, "changes_absolute", "epochs", "parameters"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_parameter_changes_relative(
        ax,
        x_values,
        data["encoder_change_rel"],
        data["decoder_change_rel"],
        data["total_change_rel"],
        MARKER_SIZE_MEDIUM,
    )
    save_figure(make_plot_path(fig_dir, "changes_relative", "epochs", "parameters"))
    plt.close()


def _plot_parameter_norms(
    ax,
    x_values: np.ndarray,
    encoder_norm: np.ndarray,
    decoder_norm: np.ndarray,
    total_norm: np.ndarray,
    data_source: str,
    marker_size: int,
) -> None:
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
    ax.set_title(f"Parameter L2 Norms ({data_source})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_parameter_changes_absolute(
    ax,
    x_values: np.ndarray,
    encoder_change_norm: np.ndarray,
    decoder_change_norm: np.ndarray,
    total_change_norm: np.ndarray,
    marker_size: int,
) -> None:
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
    ax.set_title("Absolute Parameter Changes")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm of Change")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_parameter_changes_relative(
    ax,
    x_values: np.ndarray,
    encoder_change_rel: np.ndarray,
    decoder_change_rel: np.ndarray,
    total_change_rel: np.ndarray,
    marker_size: int,
) -> None:
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
    ax.set_title("Relative Parameter Changes")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative Change (Δ / Current)")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
