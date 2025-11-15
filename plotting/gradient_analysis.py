"""Gradient analysis and diagnostic plots."""

import matplotlib.pyplot as plt
import numpy as np

from .core import DEFAULT_DPI


def save_gradient_diagnostics(
    train_history: dict, train_step_history: dict, fig_dir: str
):
    """Save separate gradient diagnostic plots for epoch and step level data."""

    # Save epoch-level gradient diagnostics
    _save_epoch_gradient_diagnostics(train_history, fig_dir)

    # Save step-level gradient diagnostics if available
    if train_step_history and train_step_history.get("step"):
        _save_step_gradient_diagnostics(train_step_history, fig_dir)


def _save_epoch_gradient_diagnostics(train_history: dict, fig_dir: str):
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

    if not all(train_history.get(key) for key in required_keys):
        print("Skipping epoch gradient diagnostics: Missing required history keys.")
        return

    epochs = np.asarray(train_history["epoch"], dtype=float)
    if len(epochs) == 0:
        return

    # Load data
    recon_norm = np.asarray(train_history["recon_grad_norm"])
    kl_norm = np.asarray(train_history["kl_grad_norm"])
    realized_norm = np.asarray(train_history["grad_norm_realized"])
    unclipped_norm = np.asarray(train_history["grad_norm_unclipped"])
    recon_kl_cosine = np.asarray(train_history["recon_kl_cosine"])
    recon_contrib = np.asarray(train_history["recon_contrib"])
    kl_contrib = np.asarray(train_history["kl_contrib"])

    # 1. Gradient Norms
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_gradient_norms(ax, epochs, recon_norm, kl_norm, realized_norm, unclipped_norm)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gradient_norms_epochs.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 2. Gradient Alignment
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_gradient_alignment(ax, epochs, recon_kl_cosine)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gradient_alignment_epochs.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 3. Gradient Contributions
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_gradient_contributions(ax, epochs, recon_contrib, kl_contrib)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gradient_contributions_epochs.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 4. Effective Magnitudes
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_effective_magnitudes(ax, epochs, realized_norm, recon_contrib, kl_contrib)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gradient_effective_magnitudes_epochs.webp", dpi=DEFAULT_DPI)
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

    if not all(train_step_history.get(key) for key in required_keys):
        print("Skipping step gradient diagnostics: Missing required history keys.")
        return

    steps = np.asarray(train_step_history["step"], dtype=float)
    if len(steps) == 0:
        return

    # Load data
    recon_norm = np.asarray(train_step_history["recon_grad_norm"])
    kl_norm = np.asarray(train_step_history["kl_grad_norm"])
    realized_norm = np.asarray(train_step_history["grad_norm_realized"])
    unclipped_norm = np.asarray(train_step_history["grad_norm_unclipped"])
    recon_kl_cosine = np.asarray(train_step_history["recon_kl_cosine"])
    recon_contrib = np.asarray(train_step_history["recon_contrib"])
    kl_contrib = np.asarray(train_step_history["kl_contrib"])

    # 1. Step Gradient Norms
    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_gradient_norms_steps(
        ax, steps, recon_norm, kl_norm, realized_norm, unclipped_norm
    )
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gradient_norms_steps.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 2. Step Gradient Alignment
    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_gradient_alignment_steps(ax, steps, recon_kl_cosine)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gradient_alignment_steps.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 3. Step Gradient Contributions
    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_gradient_contributions_steps(ax, steps, recon_contrib, kl_contrib)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gradient_contributions_steps.webp", dpi=DEFAULT_DPI)
    plt.close()

    # 4. Step Effective Magnitudes
    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_effective_magnitudes_steps(
        ax, steps, realized_norm, recon_contrib, kl_contrib
    )
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gradient_effective_magnitudes_steps.webp", dpi=DEFAULT_DPI)
    plt.close()


# Epoch-level plotting functions
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


# Step-level plotting functions
def _plot_gradient_norms_steps(
    ax, steps, recon_norm, kl_norm, realized_norm, unclipped_norm
):
    """Plot gradient norms for step-level data."""
    ax.plot(
        steps,
        recon_norm,
        "o-",
        label="Recon Norm",
        markersize=2,
        alpha=0.8,
        color="blue",
    )
    ax.plot(steps, kl_norm, "s-", label="KL Norm", markersize=2, alpha=0.8, color="red")
    ax.plot(
        steps,
        realized_norm,
        "^-",
        label="Realized Total Norm",
        markersize=2,
        alpha=0.8,
        color="green",
    )
    if np.any(unclipped_norm != realized_norm):
        ax.plot(
            steps,
            unclipped_norm,
            "--",
            label="Unclipped Total Norm",
            alpha=0.5,
            color="gray",
        )
    ax.set_title("Gradient L2 Norms by Training Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def _plot_gradient_alignment_steps(ax, steps, recon_kl_cosine):
    """Plot gradient alignment for step-level data."""
    ax.plot(
        steps,
        recon_kl_cosine,
        "o-",
        label="Recon vs KL Cosine",
        markersize=2,
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
    ax.set_title("Gradient Alignment by Training Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim([-1.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_gradient_contributions_steps(ax, steps, recon_contrib, kl_contrib):
    """Plot gradient contributions for step-level data."""
    ax.plot(
        steps,
        recon_contrib,
        "o-",
        label="Recon Contribution",
        markersize=2,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        steps,
        kl_contrib,
        "s-",
        label="KL Contribution",
        markersize=2,
        alpha=0.8,
        color="red",
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Balanced")
    ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.5)
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.5)
    ax.set_title("Relative Contribution to Update Direction by Training Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Normalized Contribution (Sum = 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_effective_magnitudes_steps(
    ax, steps, realized_norm, recon_contrib, kl_contrib
):
    """Plot effective magnitudes for step-level data."""
    effective_recon = realized_norm * recon_contrib
    effective_kl = realized_norm * kl_contrib
    ax.plot(
        steps,
        effective_recon,
        "o-",
        label="Effective Recon",
        markersize=2,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        steps,
        effective_kl,
        "s-",
        label="Effective KL",
        markersize=2,
        alpha=0.8,
        color="red",
    )
    ax.set_title("Effective Realized Gradient Magnitudes by Training Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Effective L2 Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
