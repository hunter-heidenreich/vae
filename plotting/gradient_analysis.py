"""Gradient analysis and diagnostic plots."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .core import make_grouped_plot_path, save_figure

# Styling constants
EPOCH_FIGURE_SIZE = (12, 8)
STEP_FIGURE_SIZE = (14, 8)
EPOCH_MARKER_SIZE = 3
STEP_MARKER_SIZE = 2
GRID_ALPHA = 0.3
BEST_EPOCH_MARKER_SIZE = 150
BEST_EPOCH_MARKER_ALPHA = 0.9

# Color scheme
COLOR_RECON = "blue"
COLOR_KL = "red"
COLOR_REALIZED = "green"
COLOR_UNCLIPPED = "gray"
COLOR_COSINE = "purple"
COLOR_ENCODER = "blue"
COLOR_DECODER = "orange"
COLOR_TOTAL_ENCODER = "purple"
COLOR_TOTAL_DECODER = "green"


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


def save_gradient_diagnostics(
    train_history: dict,
    train_step_history: dict,
    fig_dir: str,
    best_epoch: Optional[int] = None,
):
    """Save separate gradient diagnostic plots for epoch and step level data."""
    # Save epoch-level gradient diagnostics
    _save_epoch_gradient_diagnostics(train_history, fig_dir, best_epoch)

    # Save enhanced encoder/decoder gradient diagnostics
    _save_encoder_decoder_gradient_diagnostics(train_history, fig_dir, best_epoch)

    # Save step-level gradient diagnostics if available
    if train_step_history and train_step_history.get("step"):
        _save_step_gradient_diagnostics(train_step_history, fig_dir)
        _save_step_encoder_decoder_gradient_diagnostics(train_step_history, fig_dir)


def _save_epoch_gradient_diagnostics(
    train_history: dict, fig_dir: str, best_epoch: Optional[int] = None
):
    """Save epoch-level gradient diagnostics as separate plots."""
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

    if not _validate_history_keys(
        train_history, required_keys, "epoch gradient diagnostics"
    ):
        return

    x_values = np.asarray(train_history["epoch"], dtype=float)
    if len(x_values) == 0:
        return

    # Load data
    data = {
        "recon_norm": np.asarray(train_history["recon_grad_norm"]),
        "kl_norm": np.asarray(train_history["kl_grad_norm"]),
        "realized_norm": np.asarray(train_history["grad_norm_realized"]),
        "unclipped_norm": np.asarray(train_history["grad_norm_unclipped"]),
        "recon_kl_cosine": np.asarray(train_history["recon_kl_cosine"]),
        "recon_contrib": np.asarray(train_history["recon_contrib"]),
        "kl_contrib": np.asarray(train_history["kl_contrib"]),
    }

    # 1. Gradient Norms
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_gradient_norms(
        ax,
        x_values,
        data["recon_norm"],
        data["kl_norm"],
        data["realized_norm"],
        data["unclipped_norm"],
        "Epoch",
        EPOCH_MARKER_SIZE,
    )
    _add_best_epoch_marker(
        ax, x_values, data["realized_norm"], best_epoch, "Best Model"
    )
    save_figure(make_grouped_plot_path(fig_dir, "gradients", "norms", "epochs"))
    plt.close()

    # 2. Gradient Alignment
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_gradient_alignment(
        ax, x_values, data["recon_kl_cosine"], "Epoch", EPOCH_MARKER_SIZE
    )
    _add_best_epoch_marker(
        ax, x_values, data["recon_kl_cosine"], best_epoch, "Best Model"
    )
    save_figure(make_grouped_plot_path(fig_dir, "gradients", "alignment", "epochs"))
    plt.close()

    # 3. Gradient Contributions
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_gradient_contributions(
        ax,
        x_values,
        data["recon_contrib"],
        data["kl_contrib"],
        "Epoch",
        EPOCH_MARKER_SIZE,
    )
    save_figure(make_grouped_plot_path(fig_dir, "gradients", "contributions", "epochs"))
    plt.close()

    # 4. Effective Magnitudes
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_effective_magnitudes(
        ax,
        x_values,
        data["realized_norm"],
        data["recon_contrib"],
        data["kl_contrib"],
        "Epoch",
        EPOCH_MARKER_SIZE,
    )
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "effective_magnitudes", "epochs")
    )
    plt.close()


def _save_step_gradient_diagnostics(train_step_history: dict, fig_dir: str):
    """Save step-level gradient diagnostics as separate plots."""
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

    if not _validate_history_keys(
        train_step_history, required_keys, "step gradient diagnostics"
    ):
        return

    x_values = np.asarray(train_step_history["step"], dtype=float)
    if len(x_values) == 0:
        return

    # Load data
    data = {
        "recon_norm": np.asarray(train_step_history["recon_grad_norm"]),
        "kl_norm": np.asarray(train_step_history["kl_grad_norm"]),
        "realized_norm": np.asarray(train_step_history["grad_norm_realized"]),
        "unclipped_norm": np.asarray(train_step_history["grad_norm_unclipped"]),
        "recon_kl_cosine": np.asarray(train_step_history["recon_kl_cosine"]),
        "recon_contrib": np.asarray(train_step_history["recon_contrib"]),
        "kl_contrib": np.asarray(train_step_history["kl_contrib"]),
    }

    # 1. Step Gradient Norms
    fig, ax = plt.subplots(figsize=STEP_FIGURE_SIZE)
    _plot_gradient_norms(
        ax,
        x_values,
        data["recon_norm"],
        data["kl_norm"],
        data["realized_norm"],
        data["unclipped_norm"],
        "Training Step",
        STEP_MARKER_SIZE,
    )
    save_figure(make_grouped_plot_path(fig_dir, "gradients", "norms", "steps"))
    plt.close()

    # 2. Step Gradient Alignment
    fig, ax = plt.subplots(figsize=STEP_FIGURE_SIZE)
    _plot_gradient_alignment(
        ax, x_values, data["recon_kl_cosine"], "Training Step", STEP_MARKER_SIZE
    )
    save_figure(make_grouped_plot_path(fig_dir, "gradients", "alignment", "steps"))
    plt.close()

    # 3. Step Gradient Contributions
    fig, ax = plt.subplots(figsize=STEP_FIGURE_SIZE)
    _plot_gradient_contributions(
        ax,
        x_values,
        data["recon_contrib"],
        data["kl_contrib"],
        "Training Step",
        STEP_MARKER_SIZE,
    )
    save_figure(make_grouped_plot_path(fig_dir, "gradients", "contributions", "steps"))
    plt.close()

    # 4. Step Effective Magnitudes
    fig, ax = plt.subplots(figsize=STEP_FIGURE_SIZE)
    _plot_effective_magnitudes(
        ax,
        x_values,
        data["realized_norm"],
        data["recon_contrib"],
        data["kl_contrib"],
        "Training Step",
        STEP_MARKER_SIZE,
    )
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "effective_magnitudes", "steps")
    )
    plt.close()


# Consolidated plotting functions (work for both epoch and step level)
def _plot_gradient_norms(
    ax,
    x_values: np.ndarray,
    recon_norm: np.ndarray,
    kl_norm: np.ndarray,
    realized_norm: np.ndarray,
    unclipped_norm: np.ndarray,
    xlabel: str,
    marker_size: int,
):
    """Plot gradient norms."""
    ax.plot(
        x_values,
        recon_norm,
        "-o",
        label="Recon Norm",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_RECON,
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

    # Show unclipped norm only if clipping occurred
    if np.any(unclipped_norm != realized_norm):
        ax.plot(
            x_values,
            unclipped_norm,
            "--",
            label="Unclipped Total Norm",
            alpha=0.5,
            color=COLOR_UNCLIPPED,
        )

    _configure_common_axis(ax, "Gradient L2 Norms", xlabel, "L2 Norm", use_log=True)


def _plot_gradient_alignment(
    ax, x_values: np.ndarray, recon_kl_cosine: np.ndarray, xlabel: str, marker_size: int
):
    """Plot gradient alignment (cosine similarity)."""
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
    _configure_common_axis(ax, "Gradient Alignment", xlabel, "Cosine Similarity")


def _plot_gradient_contributions(
    ax,
    x_values: np.ndarray,
    recon_contrib: np.ndarray,
    kl_contrib: np.ndarray,
    xlabel: str,
    marker_size: int,
):
    """Plot relative gradient contributions."""
    ax.plot(
        x_values,
        recon_contrib,
        "-o",
        label="Recon Contribution",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_RECON,
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
    _configure_common_axis(
        ax, "Gradient Contributions", xlabel, "Normalized Contribution"
    )


def _plot_effective_magnitudes(
    ax,
    x_values: np.ndarray,
    realized_norm: np.ndarray,
    recon_contrib: np.ndarray,
    kl_contrib: np.ndarray,
    xlabel: str,
    marker_size: int,
):
    """Plot effective gradient magnitudes."""
    effective_recon = realized_norm * recon_contrib
    effective_kl = realized_norm * kl_contrib
    ax.plot(
        x_values,
        effective_recon,
        "-o",
        label="Effective Recon",
        markersize=marker_size,
        alpha=0.8,
        color=COLOR_RECON,
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
    _configure_common_axis(
        ax, "Effective Gradient Magnitudes", xlabel, "Effective L2 Magnitude"
    )


def _save_encoder_decoder_gradient_diagnostics(
    train_history: dict, fig_dir: str, best_epoch: Optional[int] = None
):
    """Save encoder/decoder-specific gradient diagnostics."""
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

    if not _validate_history_keys(
        train_history, required_keys, "encoder/decoder gradient diagnostics"
    ):
        return

    x_values = np.asarray(train_history["epoch"], dtype=float)
    if len(x_values) == 0:
        return

    # Load encoder/decoder data
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

    # 1. Encoder vs Decoder Reconstruction Gradients
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_encoder_decoder_recon_gradients(
        ax,
        x_values,
        data["recon_encoder"],
        data["recon_decoder"],
        "Epoch",
        EPOCH_MARKER_SIZE,
    )
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "encoder_decoder_recon", "epochs")
    )
    plt.close()

    # 2. KL Gradients (encoder-only in VAE)
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_kl_gradient_distribution(
        ax, x_values, data["kl_encoder"], data["kl_decoder"], "Epoch", EPOCH_MARKER_SIZE
    )
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "kl_distribution", "epochs")
    )
    plt.close()

    # 3. Total Gradients by Component
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_total_gradients_by_component(
        ax,
        x_values,
        data["total_encoder"],
        data["total_decoder"],
        "Epoch",
        EPOCH_MARKER_SIZE,
    )
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "total_by_component", "epochs")
    )
    plt.close()

    # 4. Encoder-specific Analysis
    fig, ax = plt.subplots(figsize=EPOCH_FIGURE_SIZE)
    _plot_encoder_gradient_analysis(
        ax,
        x_values,
        data["recon_encoder"],
        data["kl_encoder"],
        data["recon_kl_encoder_cosine"],
        data["recon_encoder_contrib"],
        data["kl_encoder_contrib"],
        "Epoch",
        EPOCH_MARKER_SIZE,
    )
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "encoder_analysis", "epochs")
    )
    plt.close()


def _save_step_encoder_decoder_gradient_diagnostics(
    train_step_history: dict, fig_dir: str
):
    """Save step-level encoder/decoder-specific gradient diagnostics."""
    required_keys = [
        "step",
        "recon_encoder_grad_norm",
        "recon_decoder_grad_norm",
        "total_encoder_grad_norm",
        "total_decoder_grad_norm",
    ]

    if not _validate_history_keys(
        train_step_history,
        required_keys,
        "step-level encoder/decoder gradient diagnostics",
    ):
        return

    x_values = np.asarray(train_step_history["step"], dtype=float)
    if len(x_values) == 0:
        return

    # Load encoder/decoder data
    data = {
        "recon_encoder": np.asarray(train_step_history["recon_encoder_grad_norm"]),
        "recon_decoder": np.asarray(train_step_history["recon_decoder_grad_norm"]),
        "total_encoder": np.asarray(train_step_history["total_encoder_grad_norm"]),
        "total_decoder": np.asarray(train_step_history["total_decoder_grad_norm"]),
    }

    # 1. Step-level Encoder vs Decoder Reconstruction Gradients
    fig, ax = plt.subplots(figsize=STEP_FIGURE_SIZE)
    _plot_encoder_decoder_recon_gradients(
        ax,
        x_values,
        data["recon_encoder"],
        data["recon_decoder"],
        "Training Step",
        STEP_MARKER_SIZE,
    )
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "encoder_decoder_recon", "steps")
    )
    plt.close()

    # 2. Step-level Total Gradients by Component
    fig, ax = plt.subplots(figsize=STEP_FIGURE_SIZE)
    _plot_total_gradients_by_component(
        ax,
        x_values,
        data["total_encoder"],
        data["total_decoder"],
        "Training Step",
        STEP_MARKER_SIZE,
    )
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "total_by_component", "steps")
    )
    plt.close()


# Encoder/Decoder plotting functions
def _plot_encoder_decoder_recon_gradients(
    ax,
    x_values: np.ndarray,
    recon_encoder: np.ndarray,
    recon_decoder: np.ndarray,
    xlabel: str,
    marker_size: int,
):
    """Plot reconstruction gradients for encoder vs decoder."""
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
    _configure_common_axis(
        ax,
        "Reconstruction Gradients: Encoder vs Decoder",
        xlabel,
        "L2 Norm",
        use_log=True,
    )


def _plot_kl_gradient_distribution(
    ax,
    x_values: np.ndarray,
    kl_encoder: np.ndarray,
    kl_decoder: np.ndarray,
    xlabel: str,
    marker_size: int,
):
    """Plot KL gradient distribution (decoder should be near zero)."""
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
    _configure_common_axis(
        ax, "KL Gradient Distribution", xlabel, "L2 Norm", use_log=True
    )


def _plot_total_gradients_by_component(
    ax,
    x_values: np.ndarray,
    total_encoder: np.ndarray,
    total_decoder: np.ndarray,
    xlabel: str,
    marker_size: int,
):
    """Plot total gradients by encoder/decoder component."""
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
    _configure_common_axis(
        ax, "Total Gradients by Component", xlabel, "L2 Norm", use_log=True
    )


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
):
    """Plot encoder-specific gradient analysis with dual y-axes."""
    ax2 = ax.twinx()

    # Plot gradient norms on main axis
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

    # Plot cosine similarity on secondary axis
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
