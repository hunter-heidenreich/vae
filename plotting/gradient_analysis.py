"""Gradient analysis and diagnostic plots."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    COLOR_COSINE,
    COLOR_DECODER,
    COLOR_ENCODER,
    COLOR_KL,
    COLOR_REALIZED,
    COLOR_TOTAL_DECODER,
    COLOR_TOTAL_ENCODER,
    COLOR_UNCLIPPED,
    FIGURE_SIZE_STANDARD,
    FIGURE_SIZE_WIDE,
    GRID_ALPHA,
    MARKER_SIZE_MEDIUM,
    MARKER_SIZE_SMALL,
)
from .core import (
    add_best_epoch_marker,
    extract_history_data,
    make_plot_path,
    save_figure,
)


def save_gradient_diagnostics(
    train_history: dict,
    train_step_history: dict,
    fig_dir: str,
    best_epoch: Optional[int] = None,
) -> None:
    _save_epoch_gradient_diagnostics(train_history, fig_dir, best_epoch)
    _save_encoder_decoder_gradient_diagnostics(train_history, fig_dir, best_epoch)

    if train_step_history and train_step_history.get("step"):
        _save_step_gradient_diagnostics(train_step_history, fig_dir)
        _save_step_encoder_decoder_gradient_diagnostics(train_step_history, fig_dir)


def _save_epoch_gradient_diagnostics(
    train_history: dict, fig_dir: str, best_epoch: Optional[int] = None
) -> None:
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

    if not all(train_history.get(k) for k in required_keys):
        return

    (
        x_values,
        recon_norm,
        kl_norm,
        realized_norm,
        unclipped_norm,
        recon_kl_cosine,
        recon_contrib,
        kl_contrib,
    ) = extract_history_data(
        train_history,
        "epoch",
        "recon_grad_norm",
        "kl_grad_norm",
        "grad_norm_realized",
        "grad_norm_unclipped",
        "recon_kl_cosine",
        "recon_contrib",
        "kl_contrib",
    )

    if len(x_values) == 0:
        return

    data = {
        "recon_norm": recon_norm,
        "kl_norm": kl_norm,
        "realized_norm": realized_norm,
        "unclipped_norm": unclipped_norm,
        "recon_kl_cosine": recon_kl_cosine,
        "recon_contrib": recon_contrib,
        "kl_contrib": kl_contrib,
    }

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_gradient_norms(
        ax,
        x_values,
        data["recon_norm"],
        data["kl_norm"],
        data["realized_norm"],
        data["unclipped_norm"],
        "Epoch",
        MARKER_SIZE_MEDIUM,
    )
    add_best_epoch_marker(ax, x_values, data["realized_norm"], best_epoch, "Best Model")
    save_figure(make_plot_path(fig_dir, "norms", "epochs", "gradients"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_gradient_alignment(
        ax, x_values, data["recon_kl_cosine"], "Epoch", MARKER_SIZE_MEDIUM
    )
    add_best_epoch_marker(
        ax, x_values, data["recon_kl_cosine"], best_epoch, "Best Model"
    )
    save_figure(make_plot_path(fig_dir, "alignment", "epochs", "gradients"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_gradient_contributions(
        ax,
        x_values,
        data["recon_contrib"],
        data["kl_contrib"],
        "Epoch",
        MARKER_SIZE_MEDIUM,
    )
    save_figure(make_plot_path(fig_dir, "contributions", "epochs", "gradients"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_effective_magnitudes(
        ax,
        x_values,
        data["realized_norm"],
        data["recon_contrib"],
        data["kl_contrib"],
        "Epoch",
        MARKER_SIZE_MEDIUM,
    )
    save_figure(make_plot_path(fig_dir, "effective_magnitudes", "epochs", "gradients"))
    plt.close()


def _save_step_gradient_diagnostics(train_step_history: dict, fig_dir: str) -> None:
    required_keys = [
        "step",
        "recon_grad_norm",
        "kl_grad_norm",
        "grad_norm_realized",
        "grad_norm_unclipped",
        "recon_kl_cosine",
        "recon_contrib",
        "kl_contrib",
    ]

    if not all(train_step_history.get(k) for k in required_keys):
        return

    x_values = np.asarray(train_step_history["step"], dtype=float)
    if len(x_values) == 0:
        return

    data = {
        "recon_norm": np.asarray(train_step_history["recon_grad_norm"]),
        "kl_norm": np.asarray(train_step_history["kl_grad_norm"]),
        "realized_norm": np.asarray(train_step_history["grad_norm_realized"]),
        "unclipped_norm": np.asarray(train_step_history["grad_norm_unclipped"]),
        "recon_kl_cosine": np.asarray(train_step_history["recon_kl_cosine"]),
        "recon_contrib": np.asarray(train_step_history["recon_contrib"]),
        "kl_contrib": np.asarray(train_step_history["kl_contrib"]),
    }

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    _plot_gradient_norms(
        ax,
        x_values,
        data["recon_norm"],
        data["kl_norm"],
        data["realized_norm"],
        data["unclipped_norm"],
        "Training Step",
        MARKER_SIZE_SMALL,
    )
    save_figure(make_plot_path(fig_dir, "norms", "steps", "gradients"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    _plot_gradient_alignment(
        ax, x_values, data["recon_kl_cosine"], "Training Step", MARKER_SIZE_SMALL
    )
    save_figure(make_plot_path(fig_dir, "alignment", "steps", "gradients"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    _plot_gradient_contributions(
        ax,
        x_values,
        data["recon_contrib"],
        data["kl_contrib"],
        "Training Step",
        MARKER_SIZE_SMALL,
    )
    save_figure(make_plot_path(fig_dir, "contributions", "steps", "gradients"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    _plot_effective_magnitudes(
        ax,
        x_values,
        data["realized_norm"],
        data["recon_contrib"],
        data["kl_contrib"],
        "Training Step",
        MARKER_SIZE_SMALL,
    )
    save_figure(make_plot_path(fig_dir, "effective_magnitudes", "steps", "gradients"))
    plt.close()


def _plot_gradient_norms(
    ax,
    x_values: np.ndarray,
    recon_norm: np.ndarray,
    kl_norm: np.ndarray,
    realized_norm: np.ndarray,
    unclipped_norm: np.ndarray,
    xlabel: str,
    marker_size: int,
) -> None:
    ax.plot(
        x_values,
        recon_norm,
        "-o",
        label="Recon Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_ENCODER,
    )
    ax.plot(
        x_values,
        kl_norm,
        "-s",
        label="KL Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_KL,
    )
    ax.plot(
        x_values,
        realized_norm,
        "-^",
        label="Realized Total Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_REALIZED,
    )

    if np.any(unclipped_norm != realized_norm):
        ax.plot(
            x_values,
            unclipped_norm,
            "--",
            label="Unclipped Total Norm",
            alpha=0.5,
            color=COLOR_UNCLIPPED,
        )

    ax.set_title("Gradient L2 Norms")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("L2 Norm")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_gradient_alignment(
    ax, x_values: np.ndarray, recon_kl_cosine: np.ndarray, xlabel: str, marker_size: int
) -> None:
    ax.plot(
        x_values,
        recon_kl_cosine,
        "-d",
        label="Recon vs KL Cosine",
        markersize=marker_size,
        color=COLOR_COSINE,
    )
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.7, label="Orthogonal")
    ax.axhline(y=-1, color="gray", linestyle="--", alpha=0.7, label="Opposite")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.7, label="Aligned")
    ax.set_ylim([-1.1, 1.1])
    ax.set_title("Gradient Alignment")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cosine Similarity")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_gradient_contributions(
    ax,
    x_values: np.ndarray,
    recon_contrib: np.ndarray,
    kl_contrib: np.ndarray,
    xlabel: str,
    marker_size: int,
) -> None:
    ax.plot(
        x_values,
        recon_contrib,
        "-o",
        label="Recon Contribution",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_ENCODER,
    )
    ax.plot(
        x_values,
        kl_contrib,
        "-s",
        label="KL Contribution",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_KL,
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Balanced")
    ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.5)
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.5)
    ax.set_title("Gradient Contributions")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized Contribution")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_effective_magnitudes(
    ax,
    x_values: np.ndarray,
    realized_norm: np.ndarray,
    recon_contrib: np.ndarray,
    kl_contrib: np.ndarray,
    xlabel: str,
    marker_size: int,
) -> None:
    effective_recon = realized_norm * recon_contrib
    effective_kl = realized_norm * kl_contrib
    ax.plot(
        x_values,
        effective_recon,
        "-o",
        label="Effective Recon",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_ENCODER,
    )
    ax.plot(
        x_values,
        effective_kl,
        "-s",
        label="Effective KL",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_KL,
    )
    ax.set_title("Effective Gradient Magnitudes")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Effective L2 Magnitude")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _save_encoder_decoder_gradient_diagnostics(
    train_history: dict, fig_dir: str, best_epoch: Optional[int] = None
) -> None:
    required_keys = [
        "epoch",
        "recon_encoder_grad_norm",
        "recon_decoder_grad_norm",
        "kl_encoder_grad_norm",
        "kl_decoder_grad_norm",
        "total_encoder_grad_norm",
        "total_decoder_grad_norm",
        "recon_kl_encoder_cosine",
        "recon_encoder_contrib",
        "kl_encoder_contrib",
    ]

    if not all(train_history.get(k) for k in required_keys):
        return

    x_values = np.asarray(train_history["epoch"], dtype=float)
    if len(x_values) == 0:
        return

    data = {
        "recon_encoder": np.asarray(train_history["recon_encoder_grad_norm"]),
        "recon_decoder": np.asarray(train_history["recon_decoder_grad_norm"]),
        "kl_encoder": np.asarray(train_history["kl_encoder_grad_norm"]),
        "kl_decoder": np.asarray(train_history["kl_decoder_grad_norm"]),
        "total_encoder": np.asarray(train_history["total_encoder_grad_norm"]),
        "total_decoder": np.asarray(train_history["total_decoder_grad_norm"]),
        "recon_kl_encoder_cosine": np.asarray(train_history["recon_kl_encoder_cosine"]),
        "recon_encoder_contrib": np.asarray(train_history["recon_encoder_contrib"]),
        "kl_encoder_contrib": np.asarray(train_history["kl_encoder_contrib"]),
    }

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_encoder_decoder_recon_gradients(
        ax,
        x_values,
        data["recon_encoder"],
        data["recon_decoder"],
        "Epoch",
        MARKER_SIZE_MEDIUM,
    )
    save_figure(make_plot_path(fig_dir, "encoder_decoder_recon", "epochs", "gradients"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_kl_gradient_distribution(
        ax,
        x_values,
        data["kl_encoder"],
        data["kl_decoder"],
        "Epoch",
        MARKER_SIZE_MEDIUM,
    )
    save_figure(make_plot_path(fig_dir, "kl_distribution", "epochs", "gradients"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_total_gradients_by_component(
        ax,
        x_values,
        data["total_encoder"],
        data["total_decoder"],
        "Epoch",
        MARKER_SIZE_MEDIUM,
    )
    save_figure(make_plot_path(fig_dir, "total_by_component", "epochs", "gradients"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    _plot_encoder_gradient_analysis(
        ax,
        x_values,
        data["recon_encoder"],
        data["kl_encoder"],
        data["recon_kl_encoder_cosine"],
        data["recon_encoder_contrib"],
        data["kl_encoder_contrib"],
        "Epoch",
        MARKER_SIZE_MEDIUM,
    )
    save_figure(make_plot_path(fig_dir, "encoder_analysis", "epochs", "gradients"))
    plt.close()


def _save_step_encoder_decoder_gradient_diagnostics(
    train_step_history: dict, fig_dir: str
) -> None:
    required_keys = [
        "step",
        "recon_encoder_grad_norm",
        "recon_decoder_grad_norm",
        "total_encoder_grad_norm",
        "total_decoder_grad_norm",
    ]

    if not all(train_step_history.get(k) for k in required_keys):
        return

    x_values = np.asarray(train_step_history["step"], dtype=float)
    if len(x_values) == 0:
        return

    data = {
        "recon_encoder": np.asarray(train_step_history["recon_encoder_grad_norm"]),
        "recon_decoder": np.asarray(train_step_history["recon_decoder_grad_norm"]),
        "total_encoder": np.asarray(train_step_history["total_encoder_grad_norm"]),
        "total_decoder": np.asarray(train_step_history["total_decoder_grad_norm"]),
    }

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    _plot_encoder_decoder_recon_gradients(
        ax,
        x_values,
        data["recon_encoder"],
        data["recon_decoder"],
        "Training Step",
        MARKER_SIZE_SMALL,
    )
    save_figure(make_plot_path(fig_dir, "encoder_decoder_recon", "steps", "gradients"))
    plt.close()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    _plot_total_gradients_by_component(
        ax,
        x_values,
        data["total_encoder"],
        data["total_decoder"],
        "Training Step",
        MARKER_SIZE_SMALL,
    )
    save_figure(make_plot_path(fig_dir, "total_by_component", "steps", "gradients"))
    plt.close()


def _plot_encoder_decoder_recon_gradients(
    ax,
    x_values: np.ndarray,
    recon_encoder: np.ndarray,
    recon_decoder: np.ndarray,
    xlabel: str,
    marker_size: int,
) -> None:
    ax.plot(
        x_values,
        recon_encoder,
        "-o",
        label="Encoder Recon Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_ENCODER,
    )
    ax.plot(
        x_values,
        recon_decoder,
        "-s",
        label="Decoder Recon Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_DECODER,
    )
    ax.set_title("Reconstruction Gradients: Encoder vs Decoder")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("L2 Norm")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_kl_gradient_distribution(
    ax,
    x_values: np.ndarray,
    kl_encoder: np.ndarray,
    kl_decoder: np.ndarray,
    xlabel: str,
    marker_size: int,
) -> None:
    ax.plot(
        x_values,
        kl_encoder,
        "-o",
        label="Encoder KL Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_KL,
    )
    ax.plot(
        x_values,
        kl_decoder,
        "-s",
        label="Decoder KL Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_UNCLIPPED,
    )
    ax.set_title("KL Gradient Distribution")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("L2 Norm")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_total_gradients_by_component(
    ax,
    x_values: np.ndarray,
    total_encoder: np.ndarray,
    total_decoder: np.ndarray,
    xlabel: str,
    marker_size: int,
) -> None:
    ax.plot(
        x_values,
        total_encoder,
        "-o",
        label="Total Encoder Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_TOTAL_ENCODER,
    )
    ax.plot(
        x_values,
        total_decoder,
        "-s",
        label="Total Decoder Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_TOTAL_DECODER,
    )
    ax.set_title("Total Gradients by Component")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("L2 Norm")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_encoder_gradient_analysis(
    ax,
    x_values: np.ndarray,
    recon_encoder: np.ndarray,
    kl_encoder: np.ndarray,
    recon_kl_encoder_cosine: np.ndarray,
    recon_encoder_contrib: np.ndarray,
    kl_encoder_contrib: np.ndarray,
    xlabel: str,
    marker_size: int,
) -> None:
    ax2 = ax.twinx()
    ax.plot(
        x_values,
        recon_encoder,
        "-o",
        label="Encoder Recon Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_ENCODER,
    )
    ax.plot(
        x_values,
        kl_encoder,
        "-s",
        label="Encoder KL Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_KL,
    )
    ax.set_ylabel("L2 Norm", color="k")
    ax.set_yscale("log")
    ax.legend(loc="upper left")

    ax2.plot(
        x_values,
        recon_kl_encoder_cosine,
        "-d",
        label="Recon-KL Cosine",
        markersize=marker_size,
        color=COLOR_COSINE,
        alpha=0.7,
    )
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Cosine Similarity", color=COLOR_COSINE)
    ax2.set_ylim([-1.1, 1.1])
    ax2.legend(loc="upper right")

    ax.set_title("Encoder Gradient Analysis")
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=GRID_ALPHA)
