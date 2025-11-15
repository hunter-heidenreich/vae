"""Gradient analysis and diagnostic plots."""

import matplotlib.pyplot as plt
import numpy as np

from .core import make_grouped_plot_path, save_figure


def _add_best_epoch_marker(ax, epochs, values, best_epoch, label_prefix="Best"):
    """Add a marker for the best epoch on a plot."""
    if best_epoch is not None and best_epoch in epochs:
        best_idx = np.where(epochs == best_epoch)[0][0]
        ax.scatter(
            epochs[best_idx],
            values[best_idx],
            marker="*",
            s=150,
            color="gold",
            edgecolor="darkorange",
            linewidth=1.5,
            alpha=0.9,
            zorder=10,
            label=f"{label_prefix} (Epoch {int(best_epoch)})",
        )


def save_gradient_diagnostics(
    train_history: dict, train_step_history: dict, fig_dir: str, best_epoch: int = None
):
    """Save separate gradient diagnostic plots for epoch and step level data."""

    # Save epoch-level gradient diagnostics
    _save_epoch_gradient_diagnostics(train_history, fig_dir, best_epoch)

    # Save enhanced encoder/decoder gradient diagnostics
    _save_encoder_decoder_gradient_diagnostics(train_history, fig_dir, best_epoch)

    # Save step-level gradient diagnostics if available
    if train_step_history and train_step_history.get("step"):
        _save_step_gradient_diagnostics(train_step_history, fig_dir)

        # Save step-level encoder/decoder diagnostics
        _save_step_encoder_decoder_gradient_diagnostics(train_step_history, fig_dir)


def _save_epoch_gradient_diagnostics(
    train_history: dict, fig_dir: str, best_epoch: int = None
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
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "gradient_norms", "epochs")
    )
    plt.close()

    # 2. Gradient Alignment
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_gradient_alignment(ax, epochs, recon_kl_cosine)
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "gradient_alignment", "epochs")
    )
    plt.close()

    # 3. Gradient Contributions
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_gradient_contributions(ax, epochs, recon_contrib, kl_contrib)
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "gradient_contributions", "epochs")
    )
    plt.close()

    # 4. Effective Magnitudes
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_effective_magnitudes(ax, epochs, realized_norm, recon_contrib, kl_contrib)
    save_figure(
        make_grouped_plot_path(
            fig_dir, "gradients", "gradient_effective_magnitudes", "epochs"
        )
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
    save_figure(make_grouped_plot_path(fig_dir, "gradients", "gradient_norms", "steps"))
    plt.close()

    # 2. Step Gradient Alignment
    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_gradient_alignment_steps(ax, steps, recon_kl_cosine)
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "gradient_alignment", "steps")
    )
    plt.close()

    # 3. Step Gradient Contributions
    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_gradient_contributions_steps(ax, steps, recon_contrib, kl_contrib)
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "gradient_contributions", "steps")
    )
    plt.close()

    # 4. Step Effective Magnitudes
    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_effective_magnitudes_steps(
        ax, steps, realized_norm, recon_contrib, kl_contrib
    )
    save_figure(
        make_grouped_plot_path(
            fig_dir, "gradients", "gradient_effective_magnitudes", "steps"
        )
    )
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


def _save_encoder_decoder_gradient_diagnostics(
    train_history: dict, fig_dir: str, best_epoch: int = None
):
    """Save encoder/decoder-specific gradient diagnostics."""
    encoder_decoder_keys = [
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

    if not all(train_history.get(key) for key in encoder_decoder_keys):
        print(
            "Skipping encoder/decoder gradient diagnostics: Missing required history keys."
        )
        return

    epochs = np.asarray(train_history["epoch"], dtype=float)
    if len(epochs) == 0:
        return

    # Load encoder/decoder data
    recon_enc = np.asarray(train_history["recon_encoder_grad_norm"])
    recon_dec = np.asarray(train_history["recon_decoder_grad_norm"])
    kl_enc = np.asarray(train_history["kl_encoder_grad_norm"])
    kl_dec = np.asarray(train_history["kl_decoder_grad_norm"])
    total_enc = np.asarray(train_history["total_encoder_grad_norm"])
    total_dec = np.asarray(train_history["total_decoder_grad_norm"])
    recon_kl_enc_cosine = np.asarray(train_history["recon_kl_encoder_cosine"])
    recon_enc_contrib = np.asarray(train_history["recon_encoder_contrib"])
    kl_enc_contrib = np.asarray(train_history["kl_encoder_contrib"])

    # 1. Encoder vs Decoder Reconstruction Gradients
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_encoder_decoder_recon_gradients(ax, epochs, recon_enc, recon_dec)
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "encoder_decoder_recon", "epochs")
    )
    plt.close()

    # 2. KL Gradients (should be encoder-only)
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_kl_gradient_distribution(ax, epochs, kl_enc, kl_dec)
    save_figure(
        make_grouped_plot_path(
            fig_dir, "gradients", "kl_gradient_distribution", "epochs"
        )
    )
    plt.close()

    # 3. Total Gradients by Component
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_total_gradients_by_component(ax, epochs, total_enc, total_dec)
    save_figure(
        make_grouped_plot_path(
            fig_dir, "gradients", "total_gradients_by_component", "epochs"
        )
    )
    plt.close()

    # 4. Encoder-specific Analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_encoder_gradient_analysis(
        ax,
        epochs,
        recon_enc,
        kl_enc,
        recon_kl_enc_cosine,
        recon_enc_contrib,
        kl_enc_contrib,
    )
    save_figure(
        make_grouped_plot_path(
            fig_dir, "gradients", "encoder_gradient_analysis", "epochs"
        )
    )
    plt.close()


def _save_step_encoder_decoder_gradient_diagnostics(
    train_step_history: dict, fig_dir: str
):
    """Save step-level encoder/decoder-specific gradient diagnostics."""
    encoder_decoder_keys = [
        "step",
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

    if not all(train_step_history.get(key) for key in encoder_decoder_keys):
        print(
            "Skipping step-level encoder/decoder gradient diagnostics: Missing required history keys."
        )
        return

    steps = np.asarray(train_step_history["step"], dtype=float)
    if len(steps) == 0:
        return

    # Load encoder/decoder data
    recon_enc = np.asarray(train_step_history["recon_encoder_grad_norm"])
    recon_dec = np.asarray(train_step_history["recon_decoder_grad_norm"])
    total_enc = np.asarray(train_step_history["total_encoder_grad_norm"])
    total_dec = np.asarray(train_step_history["total_decoder_grad_norm"])

    # 1. Step-level Encoder vs Decoder Reconstruction Gradients
    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_encoder_decoder_recon_gradients_steps(ax, steps, recon_enc, recon_dec)
    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "encoder_decoder_recon", "steps")
    )
    plt.close()

    # 2. Step-level Total Gradients by Component
    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_total_gradients_by_component_steps(ax, steps, total_enc, total_dec)
    save_figure(
        make_grouped_plot_path(
            fig_dir, "gradients", "total_gradients_by_component", "steps"
        )
    )
    plt.close()


# Encoder/Decoder plotting functions
def _plot_encoder_decoder_recon_gradients(ax, epochs, recon_enc, recon_dec):
    """Plot reconstruction gradients for encoder vs decoder."""
    ax.plot(
        epochs,
        recon_enc,
        "-o",
        label="Encoder Recon Norm",
        markersize=3,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        epochs,
        recon_dec,
        "-s",
        label="Decoder Recon Norm",
        markersize=3,
        alpha=0.8,
        color="orange",
    )
    ax.set_title("Reconstruction Gradients: Encoder vs Decoder")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def _plot_kl_gradient_distribution(ax, epochs, kl_enc, kl_dec):
    """Plot KL gradient distribution (decoder should be ~0)."""
    ax.plot(
        epochs,
        kl_enc,
        "-o",
        label="Encoder KL Norm",
        markersize=3,
        alpha=0.8,
        color="red",
    )
    ax.plot(
        epochs,
        kl_dec,
        "-s",
        label="Decoder KL Norm (should be ~0)",
        markersize=3,
        alpha=0.8,
        color="gray",
    )
    ax.set_title("KL Gradient Distribution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def _plot_total_gradients_by_component(ax, epochs, total_enc, total_dec):
    """Plot total gradients by encoder/decoder component."""
    ax.plot(
        epochs,
        total_enc,
        "-o",
        label="Total Encoder Norm",
        markersize=3,
        alpha=0.8,
        color="purple",
    )
    ax.plot(
        epochs,
        total_dec,
        "-s",
        label="Total Decoder Norm",
        markersize=3,
        alpha=0.8,
        color="green",
    )
    ax.set_title("Total Gradients by Network Component")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def _plot_encoder_gradient_analysis(
    ax,
    epochs,
    recon_enc,
    kl_enc,
    recon_kl_enc_cosine,
    recon_enc_contrib,
    kl_enc_contrib,
):
    """Plot encoder-specific gradient analysis."""
    ax2 = ax.twinx()

    # Plot gradient norms on main axis
    ax.plot(
        epochs,
        recon_enc,
        "-o",
        label="Encoder Recon Norm",
        markersize=3,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        epochs,
        kl_enc,
        "-s",
        label="Encoder KL Norm",
        markersize=3,
        alpha=0.8,
        color="red",
    )
    ax.set_ylabel("L2 Norm", color="k")
    ax.set_yscale("log")
    ax.legend(loc="upper left")

    # Plot cosine similarity on secondary axis
    ax2.plot(
        epochs,
        recon_kl_enc_cosine,
        "-d",
        label="Recon-KL Cosine (Encoder)",
        markersize=3,
        color="purple",
        alpha=0.7,
    )
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Cosine Similarity", color="purple")
    ax2.set_ylim([-1.1, 1.1])
    ax2.legend(loc="upper right")

    ax.set_title("Encoder Gradient Analysis")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)


def _plot_encoder_decoder_recon_gradients_steps(ax, steps, recon_enc, recon_dec):
    """Plot step-level reconstruction gradients for encoder vs decoder."""
    ax.plot(
        steps,
        recon_enc,
        "o-",
        label="Encoder Recon Norm",
        markersize=2,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        steps,
        recon_dec,
        "s-",
        label="Decoder Recon Norm",
        markersize=2,
        alpha=0.8,
        color="orange",
    )
    ax.set_title("Reconstruction Gradients by Training Step: Encoder vs Decoder")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def _plot_total_gradients_by_component_steps(ax, steps, total_enc, total_dec):
    """Plot step-level total gradients by encoder/decoder component."""
    ax.plot(
        steps,
        total_enc,
        "o-",
        label="Total Encoder Norm",
        markersize=2,
        alpha=0.8,
        color="purple",
    )
    ax.plot(
        steps,
        total_dec,
        "s-",
        label="Total Decoder Norm",
        markersize=2,
        alpha=0.8,
        color="green",
    )
    ax.set_title("Total Gradients by Network Component (Training Steps)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def save_parameter_diagnostics(
    train_history: dict, val_history: dict, fig_dir: str, best_epoch: int = None
):
    """Save parameter magnitude and change diagnostics."""

    # Parameter norms over time
    _save_parameter_norm_diagnostics(train_history, val_history, fig_dir, best_epoch)

    # Parameter changes over time (training only)
    _save_parameter_change_diagnostics(train_history, fig_dir, best_epoch)


def _save_parameter_norm_diagnostics(
    train_history: dict, val_history: dict, fig_dir: str, best_epoch: int = None
):
    """Save parameter norm diagnostics using both training and validation data."""
    param_norm_keys = [
        "epoch",
        "encoder_param_norm",
        "decoder_param_norm",
        "total_param_norm",
    ]

    # Check if we have the required keys in either history
    train_has_params = all(train_history.get(key) for key in param_norm_keys)
    val_has_params = all(val_history.get(key) for key in param_norm_keys)

    if not (train_has_params or val_has_params):
        print("Skipping parameter norm diagnostics: Missing required history keys.")
        return

    # Use validation data if available (cleaner), otherwise training data
    history_to_use = val_history if val_has_params else train_history
    data_source = "validation" if val_has_params else "training"

    epochs = np.asarray(history_to_use["epoch"], dtype=float)
    if len(epochs) == 0:
        return

    encoder_norm = np.asarray(history_to_use["encoder_param_norm"])
    decoder_norm = np.asarray(history_to_use["decoder_param_norm"])
    total_norm = np.asarray(history_to_use["total_param_norm"])

    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_parameter_norms(
        ax, epochs, encoder_norm, decoder_norm, total_norm, data_source
    )

    # Add best epoch marker if available
    if best_epoch is not None:
        _add_best_epoch_marker(ax, epochs, total_norm, best_epoch, "Best Model")

    save_figure(
        make_grouped_plot_path(fig_dir, "gradients", "parameter_norms", "epochs")
    )
    plt.close()


def _save_parameter_change_diagnostics(
    train_history: dict, fig_dir: str, best_epoch: int = None
):
    """Save parameter change diagnostics (training data only)."""
    param_change_keys = [
        "epoch",
        "encoder_param_change_norm",
        "decoder_param_change_norm",
        "total_param_change_norm",
        "encoder_param_change_rel",
        "decoder_param_change_rel",
        "total_param_change_rel",
    ]

    if not all(train_history.get(key) for key in param_change_keys):
        print("Skipping parameter change diagnostics: Missing required history keys.")
        return

    epochs = np.asarray(train_history["epoch"], dtype=float)
    if len(epochs) == 0:
        return

    # Absolute changes
    enc_change_norm = np.asarray(train_history["encoder_param_change_norm"])
    dec_change_norm = np.asarray(train_history["decoder_param_change_norm"])
    total_change_norm = np.asarray(train_history["total_param_change_norm"])

    # Relative changes
    enc_change_rel = np.asarray(train_history["encoder_param_change_rel"])
    dec_change_rel = np.asarray(train_history["decoder_param_change_rel"])
    total_change_rel = np.asarray(train_history["total_param_change_rel"])

    # 1. Absolute Parameter Changes
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_parameter_changes_absolute(
        ax, epochs, enc_change_norm, dec_change_norm, total_change_norm
    )
    save_figure(
        make_grouped_plot_path(
            fig_dir, "gradients", "parameter_changes_absolute", "epochs"
        )
    )
    plt.close()

    # 2. Relative Parameter Changes
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_parameter_changes_relative(
        ax, epochs, enc_change_rel, dec_change_rel, total_change_rel
    )
    save_figure(
        make_grouped_plot_path(
            fig_dir, "gradients", "parameter_changes_relative", "epochs"
        )
    )
    plt.close()


def _plot_parameter_norms(
    ax, epochs, encoder_norm, decoder_norm, total_norm, data_source
):
    """Plot parameter norms over epochs."""
    ax.plot(
        epochs,
        encoder_norm,
        "-o",
        label="Encoder Parameters",
        markersize=3,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        epochs,
        decoder_norm,
        "-s",
        label="Decoder Parameters",
        markersize=3,
        alpha=0.8,
        color="orange",
    )
    ax.plot(
        epochs,
        total_norm,
        "-^",
        label="Total Parameters",
        markersize=3,
        alpha=0.8,
        color="purple",
    )
    ax.set_title(f"Parameter L2 Norms Over Training ({data_source} data)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def _plot_parameter_changes_absolute(
    ax, epochs, enc_change_norm, dec_change_norm, total_change_norm
):
    """Plot absolute parameter changes over epochs."""
    ax.plot(
        epochs,
        enc_change_norm,
        "-o",
        label="Encoder Change Norm",
        markersize=3,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        epochs,
        dec_change_norm,
        "-s",
        label="Decoder Change Norm",
        markersize=3,
        alpha=0.8,
        color="orange",
    )
    ax.plot(
        epochs,
        total_change_norm,
        "-^",
        label="Total Change Norm",
        markersize=3,
        alpha=0.8,
        color="purple",
    )
    ax.set_title("Absolute Parameter Changes Between Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm of Parameter Change")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def _plot_parameter_changes_relative(
    ax, epochs, enc_change_rel, dec_change_rel, total_change_rel
):
    """Plot relative parameter changes over epochs."""
    ax.plot(
        epochs,
        enc_change_rel,
        "-o",
        label="Encoder Relative Change",
        markersize=3,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        epochs,
        dec_change_rel,
        "-s",
        label="Decoder Relative Change",
        markersize=3,
        alpha=0.8,
        color="orange",
    )
    ax.plot(
        epochs,
        total_change_rel,
        "-^",
        label="Total Relative Change",
        markersize=3,
        alpha=0.8,
        color="purple",
    )
    ax.set_title("Relative Parameter Changes Between Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative Change (Change Norm / Current Norm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
