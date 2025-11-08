from contextlib import contextmanager
from typing import Sized, cast

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from model import VAE

# Constants
MNIST_SHAPE = (1, 28, 28)
DEFAULT_DPI = 200
DEFAULT_CMAP = "tab10"
DEFAULT_ALPHA = 0.8


@contextmanager
def model_inference(model: VAE):
    """Context manager for model evaluation and no_grad."""
    model.eval()
    with torch.no_grad():
        yield


@contextmanager
def figure_context(figsize, out_path: str, dpi: int = DEFAULT_DPI):
    """Context manager for matplotlib figure creation and saving."""
    plt.figure(figsize=figsize)
    try:
        yield
    finally:
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi)
        plt.close()


def apply_pca_if_needed(
    Z: np.ndarray, target_dims: int = 2
) -> tuple[np.ndarray, PCA | None]:
    """Apply PCA reduction if dimensionality > target_dims."""
    if Z.shape[1] <= target_dims:
        return Z, None

    pca = PCA(n_components=target_dims)
    Z_reduced = pca.fit_transform(Z)
    return Z_reduced, pca


def get_colormap_colors(n_colors: int, cmap_name: str = DEFAULT_CMAP) -> np.ndarray:
    """Get evenly spaced colors from a colormap."""
    return cm.get_cmap(cmap_name)(np.linspace(0, 1, n_colors))


def decode_samples(model: VAE, z: torch.Tensor) -> torch.Tensor:
    """Decode latent vectors and reshape to MNIST format."""
    return torch.sigmoid(model.decode(z)).view(-1, *MNIST_SHAPE)


def grid_from_images(imgs: torch.Tensor, nrow: int, ncol: int) -> np.ndarray:
    """Create a HxWxC image grid from a batch of images."""
    imgs = imgs.detach().cpu().clamp(0, 1)
    N, C, H, W = imgs.shape
    canvas = torch.zeros((C, nrow * H, ncol * W))

    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            if idx >= N:
                break
            canvas[:, i * H : (i + 1) * H, j * W : (j + 1) * W] = imgs[idx]

    img = canvas.permute(1, 2, 0).numpy()
    return img.squeeze(-1) if C == 1 else img


def save_samples_figure(
    model: VAE,
    device,
    latent_dim: int,
    out_path: str,
    n: int = 100,
    grid: tuple[int, int] = (10, 10),
):
    """Generate and save random samples from the VAE."""
    with model_inference(model):
        z = torch.randn((n, latent_dim), device=device)
        samples = decode_samples(model, z)

    img = grid_from_images(samples, grid[0], grid[1])

    with figure_context((grid[1], grid[0]), out_path):
        plt.axis("off")
        plt.imshow(img, cmap="gray")


def save_recon_figure(
    model: VAE,
    data_batch: torch.Tensor,
    device,
    out_path: str,
    n: int = 16,
):
    """Save reconstruction figure showing originals and reconstructions side by side."""
    n = min(n, len(data_batch))

    with model_inference(model):
        data = data_batch[:n].to(device)
        mu, _ = model.encode(data)
        recon = decode_samples(model, mu)

    # Stack originals then reconstructions
    imgs = torch.cat([data.view(-1, *MNIST_SHAPE), recon], dim=0)
    img_grid = grid_from_images(imgs, 2, n)  # 2×n grid

    with figure_context((n * 1.5, 2 * 1.5), out_path):
        plt.axis("off")
        plt.imshow(img_grid, cmap="gray")


def collect_latents(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect latent representations and labels from a dataset."""
    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    with model_inference(model):
        for bidx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            mu, _ = model.encode(data)
            zs.append(mu.cpu().numpy())
            ys.append(target.numpy())

            if max_batches is not None and (bidx + 1) >= max_batches:
                break

    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def _plot_scatter_2d(Z: np.ndarray, Y: np.ndarray, pca: PCA | None = None):
    """Plot 2D scatter with appropriate labels."""
    n_classes = len(np.unique(Y))
    scatter = plt.scatter(
        Z[:, 0], Z[:, 1], c=Y, s=5, cmap=DEFAULT_CMAP, alpha=DEFAULT_ALPHA
    )
    plt.colorbar(scatter, ticks=range(n_classes))

    if pca is not None:
        plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})")
        plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})")
        plt.title(f"MNIST latent PCA ({Z.shape[1]}D → 2D)")
    else:
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title("MNIST latent (mu) scatter")


def _plot_histogram_1d(Z: np.ndarray, Y: np.ndarray):
    """Plot 1D histogram with class colors."""
    n_classes = len(np.unique(Y))
    colors = get_colormap_colors(n_classes)

    for i, digit in enumerate(np.unique(Y)):
        mask = Y == digit
        plt.hist(
            Z[mask, 0], bins=30, alpha=0.7, label=f"Digit {digit}", color=colors[i]
        )

    plt.xlabel("z1")
    plt.ylabel("Count")
    plt.title("MNIST latent (mu) histogram")
    plt.legend()





def save_latent_marginals(Z: np.ndarray, out_path: str):
    """Save 1D marginal KDE plots for all latent dimensions."""
    latent_dim = Z.shape[1]

    with figure_context((8, 5), out_path):
        colors = get_colormap_colors(latent_dim)
        for i in range(latent_dim):
            sns.kdeplot(Z[:, i], label=f"z{i + 1}", color=colors[i], linewidth=2)

        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Latent marginal KDEs ({latent_dim}D)")
        plt.legend()
        plt.grid(True, alpha=0.3)


def save_latent_combined_figure(Z: np.ndarray, Y: np.ndarray, out_path: str):
    """Save a combined figure with both scatter and histogram visualizations."""
    if Z.shape[0] == 0:
        raise ValueError("Empty latent array provided")

    latent_dim = Z.shape[1]

    # Apply PCA if needed for visualization
    Z_plot, pca = apply_pca_if_needed(Z, 2)

    # Create figure with 1x2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: Scatter plot
    ax = axs[0]
    plt.sca(ax)  # Set current axis
    if Z.shape[1] == 1:
        _plot_histogram_1d(Z, Y)
    else:
        _plot_scatter_2d(Z_plot, Y, pca)

    # Right subplot: 2D histogram
    ax = axs[1]
    plt.sca(ax)  # Set current axis
    if latent_dim == 1:
        sns.histplot(Z[:, 0], kde=True, stat="density", alpha=0.7, color="steelblue")
        plt.xlabel("z1")
        plt.ylabel("Density")
        plt.title("Latent 1D distribution")
        plt.grid(True, alpha=0.3)
    else:
        plt.hist2d(Z_plot[:, 0], Z_plot[:, 1], bins=100, cmap="magma")
        plt.colorbar(label="Count")

        if pca is not None:
            plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})")
            plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})")
            plt.title(f"Latent 2D histogram ({latent_dim}D → 2D PCA)")
        else:
            plt.xlabel("z1")
            plt.ylabel("z2")
            plt.title("Latent 2D histogram")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close()




def save_interpolation_combined_figure(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    steps: int = 15,
    method: str = "slerp",
    sweep_steps: int = 15,
):
    """Create combined interpolation figure with data interpolation on top and latent sweep on bottom."""
    # First, get interpolation between two random examples
    ds = loader.dataset  # type: ignore[attr-defined]
    ds_sized = cast(Sized, ds)
    idx = torch.randint(0, len(ds_sized), (2,))
    x0, _ = ds[idx[0].item()]  # type: ignore[index]
    x1, _ = ds[idx[1].item()]  # type: ignore[index]
    x = torch.stack([x0, x1], dim=0).to(device)

    with model_inference(model):
        # Interpolation between examples
        mu, _ = model.encode(x)
        z0, z1 = mu[0], mu[1]
        t = torch.linspace(0, 1, steps, device=device)
        zs = _slerp(z0, z1, t) if method.lower() == "slerp" else _lerp(z0, z1, t)
        interp_imgs = decode_samples(model, zs)
        
        # Latent space sweep along first dimension
        z1_sweep = torch.linspace(-3, 3, sweep_steps, device=device)
        z_sweep = torch.zeros(sweep_steps, model.latent_dim, device=device)
        z_sweep[:, 0] = z1_sweep
        sweep_imgs = decode_samples(model, z_sweep)

    # Create grids
    interp_grid = grid_from_images(interp_imgs, 1, steps)
    sweep_grid = grid_from_images(sweep_imgs, 1, sweep_steps)
    
    # Create figure with 2 subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(steps, sweep_steps), 3))
    
    # Top plot: interpolation between examples
    ax1.imshow(interp_grid, cmap="gray")
    ax1.axis("off")
    ax1.set_title("Interpolation Between Two Data Points", fontsize=10, pad=10)
    
    # Bottom plot: latent sweep
    ax2.imshow(sweep_grid, cmap="gray")
    ax2.axis("off")
    ax2.set_title("Latent Space Sweep (z1 dimension)", fontsize=10, pad=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close()


def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between a and b for any shape along last dim."""
    t_ = t.view(-1, *([1] * a.dim()))
    return (1 - t_) * a.unsqueeze(0) + t_ * b.unsqueeze(0)


def _slerp(
    a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Spherical linear interpolation that works for any latent dimensionality."""
    # Normalize directions for stable angle computation
    a_norm = torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(eps)
    b_norm = torch.linalg.norm(b, dim=-1, keepdim=True).clamp_min(eps)
    a_dir = a / a_norm
    b_dir = b / b_norm

    dot = (a_dir * b_dir).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    omega = torch.arccos(dot)
    so = torch.sin(omega).clamp_min(eps)

    t_ = t.view(-1, *([1] * a.dim()))
    s0 = torch.sin((1 - t_) * omega) / so
    s1 = torch.sin(t_ * omega) / so

    out = (s0 * a.unsqueeze(0)) + (s1 * b.unsqueeze(0))

    # Fall back to lerp for small angles
    small_angle = (omega < 1e-2).squeeze(-1)
    if small_angle.any():
        out[small_angle] = _lerp(a, b, t)[small_angle]
    return out





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
        ax = axs[3]

        # Normalize epochs for color mapping (cool to warm)
        if len(test_epochs) > 1:
            epoch_normalized = (test_epochs - test_epochs.min()) / (
                test_epochs.max() - test_epochs.min()
            )
        else:
            epoch_normalized = np.array([0.5])  # Single point gets middle color

        # Draw connecting line with color gradient to show training path
        if len(test_epochs) > 1:
            # Create segments for the line with color gradient
            from matplotlib.collections import LineCollection

            points = np.column_stack([test_kl_arr, test_recon_arr])
            segments = []
            for i in range(len(points) - 1):
                segments.append([points[i], points[i + 1]])

            # Create line collection with color gradient
            lc = LineCollection(segments, cmap="coolwarm", alpha=0.6, linewidth=2)
            lc.set_array(
                epoch_normalized[:-1]
            )  # Color by starting epoch of each segment
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

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Training Progress (cool→warm)")

        # Add special markers for start and end points
        if len(test_epochs) > 0:
            # Mark start point with a green diamond
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

            # Mark end point with a red star (only if there's more than one point)
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

        # Add legend to explain the special markers
        if len(test_epochs) > 1:
            ax.legend(loc="upper right", framealpha=0.9)
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
    ax = axs[0, 0]
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

    # --- 2. Top-Right: Gradient Alignment (Interference) ---
    ax = axs[0, 1]
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

    # --- 3. Bottom-Left: Relative Contribution to Direction ---
    # This directly answers "what degree comes from where"
    ax = axs[1, 0]
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

    # --- 4. Bottom-Right: Effective Realized Magnitudes ---
    # This shows the magnitude of the *actual* update from each component
    ax = axs[1, 1]
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

    # --- Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust for suptitle
    plt.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close()





def compute_kl_per_dimension(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence for each latent dimension separately.

    Args:
        mu: Mean parameters [batch_size, latent_dim]
        logvar: Log variance parameters [batch_size, latent_dim]

    Returns:
        KL divergence per dimension [latent_dim]
    """
    # KL_i = 0.5 * (mu_i^2 + exp(logvar_i) - 1 - logvar_i)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
    # Average across batch dimension
    return kl_per_dim.mean(dim=0)

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

    # Top left: KL per dimension bar chart (final epoch)
    ax1 = plt.subplot(2, 3, 1)
    dims = np.arange(latent_dim)
    colors = [
        "red" if kl < active_threshold else "steelblue" for kl in latest_kl_per_dim
    ]
    bars = ax1.bar(
        dims,
        latest_kl_per_dim,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.axhline(
        y=active_threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold = {active_threshold}",
    )
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("KL Divergence")
    ax1.set_title("Final KL per Dimension")
    ax1.set_xticks(dims)
    ax1.set_xticklabels([f"z{i + 1}" for i in dims])
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend()

    # Add value labels on bars if not too many dimensions
    if latent_dim <= 20:
        for bar, val in zip(bars, latest_kl_per_dim):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(latest_kl_per_dim) * 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Top middle: Active units over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(
        epochs,
        active_units_over_time,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=4,
        label="Active Units",
    )
    ax2.axhline(
        y=latent_dim,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Max ({latent_dim})",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Active Units")
    ax2.set_title("Active Units Over Time")
    ax2.set_ylim([0, latent_dim + 0.5])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Top right: KL dimension heatmap
    ax3 = plt.subplot(2, 3, 3)
    im = ax3.imshow(kl_matrix.T, aspect="auto", cmap="viridis", origin="lower")

    # Set ticks for heatmap
    if len(epochs) <= 15:
        ax3.set_xticks(range(len(epochs)))
        ax3.set_xticklabels([f"{int(e)}" for e in epochs])
    else:
        step = max(1, len(epochs) // 8)
        tick_indices = range(0, len(epochs), step)
        ax3.set_xticks(tick_indices)
        ax3.set_xticklabels([f"{int(epochs[i])}" for i in tick_indices])

    ax3.set_yticks(range(latent_dim))
    ax3.set_yticklabels([f"z{i + 1}" for i in range(latent_dim)])
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Latent Dimension")
    ax3.set_title("KL Evolution Heatmap")

    # Add colorbar for heatmap
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label("KL Divergence")

    # Bottom left: KL statistics over time
    ax4 = plt.subplot(2, 3, 4)
    mean_kl = np.mean(kl_matrix, axis=1)
    std_kl = np.std(kl_matrix, axis=1)
    max_kl = np.max(kl_matrix, axis=1)
    min_kl = np.min(kl_matrix, axis=1)

    ax4.plot(epochs, mean_kl, "o-", label="Mean KL", linewidth=2, markersize=3)
    ax4.fill_between(
        epochs, mean_kl - std_kl, mean_kl + std_kl, alpha=0.3, label="±1 std"
    )
    ax4.plot(
        epochs, max_kl, "s-", label="Max KL", alpha=0.7, linewidth=1.5, markersize=3
    )
    ax4.plot(
        epochs, min_kl, "^-", label="Min KL", alpha=0.7, linewidth=1.5, markersize=3
    )
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("KL Divergence")
    ax4.set_title("KL Statistics Over Time")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Bottom middle: Distribution of KL values (histogram)
    ax5 = plt.subplot(2, 3, 5)

    # Plot histogram for a few key epochs
    n_epochs = len(epochs)
    if n_epochs >= 3:
        # Show early, middle, and final epochs
        epoch_indices = [0, n_epochs // 2, -1]
        epoch_labels = ["Early", "Middle", "Final"]
        colors_hist = ["lightblue", "orange", "red"]

        for idx, label, color in zip(epoch_indices, epoch_labels, colors_hist):
            ax5.hist(
                kl_matrix[idx],
                bins=15,
                alpha=0.6,
                label=f"{label} (E{int(epochs[idx])})",
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
    else:
        # Just show the final epoch
        ax5.hist(
            kl_matrix[-1],
            bins=15,
            alpha=0.7,
            label=f"Final (E{int(epochs[-1])})",
            color="red",
            edgecolor="black",
            linewidth=0.5,
        )

    ax5.axvline(
        x=active_threshold,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Threshold = {active_threshold}",
    )
    ax5.set_xlabel("KL Divergence")
    ax5.set_ylabel("Count")
    ax5.set_title("Distribution of KL Values")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Bottom right: Cumulative KL contribution
    ax6 = plt.subplot(2, 3, 6)

    # Sort dimensions by final KL value and compute cumulative contribution
    final_kl_sorted_idx = np.argsort(latest_kl_per_dim)[::-1]  # Descending order
    final_kl_sorted = latest_kl_per_dim[final_kl_sorted_idx]
    cumsum_kl = np.cumsum(final_kl_sorted)
    total_kl = np.sum(latest_kl_per_dim)
    cumsum_percentage = 100.0 * cumsum_kl / total_kl

    ax6.plot(
        range(1, latent_dim + 1),
        cumsum_percentage,
        "o-",
        color="purple",
        linewidth=2,
        markersize=4,
    )
    ax6.axhline(y=90, color="gray", linestyle="--", alpha=0.7, label="90% of total KL")
    ax6.axhline(y=95, color="gray", linestyle=":", alpha=0.7, label="95% of total KL")
    ax6.set_xlabel("Number of Top Dimensions")
    ax6.set_ylabel("Cumulative KL (%)")
    ax6.set_title("Cumulative KL Contribution")
    ax6.set_xticks(range(1, latent_dim + 1))
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_ylim([0, 105])

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
