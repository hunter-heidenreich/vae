import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets
from torchvision.transforms import v2 as T
from tqdm import tqdm

from model import VAE, VAEConfig
from plotting import (collect_latents, save_gradient_diagnostics,
                      save_interpolation_combined_figure,
                      save_kl_diagnostics_combined,
                      save_latent_combined_figure, save_latent_marginals,
                      save_recon_figure, save_samples_figure,
                      save_training_curves)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataloaders(batch_size: int):
    transform = T.ToTensor()

    train_data = datasets.MNIST(
        os.path.expanduser("~/.pytorch/MNIST_data/"),
        download=True,
        train=True,
        transform=transform,
    )
    test_data = datasets.MNIST(
        os.path.expanduser("~/.pytorch/MNIST_data/"),
        download=True,
        train=False,
        transform=transform,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(
    model: VAE,
    dataloader,
    optimizer,
    prev_updates: int,
    device,
    writer=None,
    history: dict | None = None,
    max_grad_norm: float | None = None,
    log_interval: int = 100,
    epoch: int = 0,
    *,
    use_analytic_kl: bool = True,
    use_distributions: bool = False,
    n_latent_samples: int = 1,
    analyze_gradients: bool = True,
):
    """
    Trains the model on the given data.

    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        optimizer: The optimizer.
        prev_updates (int): Number of previous updates (for global step tracking).
        device: Device to use for training.
        writer: TensorBoard writer for logging.
        history: Dictionary to store training history.
        max_grad_norm (float | None): Maximum gradient norm for clipping. None disables clipping.
        log_interval (int): Log metrics every N steps.
        epoch (int): Current epoch number for history tracking.
        use_analytic_kl (bool): Whether to use manual analytic KL formula (True) or torch.distributions.kl.kl_divergence (False).
        use_distributions (bool): Whether to use torch.distributions for sampling.
        n_latent_samples (int): Number of latent samples per input for ELBO estimation.
        analyze_gradients (bool): Whether to compute detailed gradient diagnostics (slower but more informative).
    """
    model.train()

    # Accumulators for epoch-level metrics
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_grad_norm = 0.0  # This will be the realized (clipped) norm
    epoch_unclipped_grad_norm = 0.0  # Track unclipped norm separately

    # Only initialize gradient analysis accumulators if needed
    if analyze_gradients:
        epoch_recon_grad_norm = 0.0
        epoch_kl_grad_norm = 0.0
        epoch_recon_contribution = 0.0
        epoch_kl_contribution = 0.0
        epoch_recon_total_cosine = 0.0
        epoch_kl_total_cosine = 0.0
        epoch_recon_kl_cosine = 0.0

    n_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, _) in enumerate(progress_bar):
        n_upd = prev_updates + batch_idx

        data = data.to(device)

        optimizer.zero_grad()
        output = model(
            data,
            compute_loss=True,
            reconstruct=False,
            use_analytic_kl=use_analytic_kl,
            use_distributions=use_distributions,
            n_samples=n_latent_samples,
        )
        loss = output.loss

        # Initialize gradient analysis variables with default values
        recon_grad_norm = 0.0
        kl_grad_norm = 0.0
        recon_contribution = 0.0
        kl_contribution = 0.0
        recon_total_cosine = torch.tensor(0.0)
        kl_total_cosine = torch.tensor(0.0)
        recon_kl_cosine = torch.tensor(0.0)

        if analyze_gradients:
            # Compute gradients for individual loss components using torch.autograd.grad
            # This is more efficient than multiple backward passes but still adds overhead

            # Get model parameters for gradient computation
            params = list(model.parameters())

            # Compute reconstruction gradients
            recon_grads = torch.autograd.grad(
                output.loss_recon,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            # Handle None gradients for unused parameters
            recon_grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(recon_grads, params)
            ]
            recon_grad_norm_sq = torch.stack(
                [g.norm(2).pow(2) for g in recon_grads]
            ).sum()
            recon_grad_norm = torch.sqrt(recon_grad_norm_sq).item()

            # Compute KL gradients
            kl_grads = torch.autograd.grad(
                output.loss_kl,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            # Handle None gradients for unused parameters
            kl_grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(kl_grads, params)
            ]
            kl_grad_norm_sq = torch.stack([g.norm(2).pow(2) for g in kl_grads]).sum()
            kl_grad_norm = torch.sqrt(kl_grad_norm_sq).item()

        # Compute total gradients (this is always needed)
        loss.backward()

        # Get the UNCLIPPED total norm and gradients
        total_grads_unclipped = [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in model.parameters()
        ]
        unclipped_grad_norm_sq = torch.stack(
            [g.norm(2).pow(2) for g in total_grads_unclipped]
        ).sum()
        unclipped_grad_norm = torch.sqrt(unclipped_grad_norm_sq).item()

        # Now apply clipping and get the realized norm
        if max_grad_norm is not None:
            # This clips p.grad in-place
            clip_grad_norm_(model.parameters(), max_grad_norm)
            # The realized norm is the smaller of the two
            realized_grad_norm = min(unclipped_grad_norm, max_grad_norm)
        else:
            realized_grad_norm = unclipped_grad_norm

        if analyze_gradients:
            # Compute gradient contribution metrics using UNCLIPPED gradients for direction analysis
            recon_total_dot = sum(
                (rg * tg).sum().item()
                for rg, tg in zip(recon_grads, total_grads_unclipped)
            )
            kl_total_dot = sum(
                (kg * tg).sum().item()
                for kg, tg in zip(kl_grads, total_grads_unclipped)
            )
            recon_kl_dot = sum(
                (rg * kg).sum().item() for rg, kg in zip(recon_grads, kl_grads)
            )

            # Use the unclipped norm for normalization (direction analysis)
            total_grad_norm_sq_unclipped = unclipped_grad_norm * unclipped_grad_norm
            recon_grad_norm_sq = recon_grad_norm * recon_grad_norm
            kl_grad_norm_sq = kl_grad_norm * kl_grad_norm

            # Compute contribution magnitudes (how much each component contributes to total gradient direction)
            recon_contribution = recon_total_dot / (
                total_grad_norm_sq_unclipped + 1e-8
            )  # Normalized contribution
            kl_contribution = kl_total_dot / (total_grad_norm_sq_unclipped + 1e-8)

            # Compute cosine similarity between gradients
            recon_total_cosine = recon_total_dot / (
                torch.sqrt(
                    torch.tensor(recon_grad_norm_sq * total_grad_norm_sq_unclipped)
                )
                + 1e-8
            )
            kl_total_cosine = kl_total_dot / (
                torch.sqrt(torch.tensor(kl_grad_norm_sq * total_grad_norm_sq_unclipped))
                + 1e-8
            )
            recon_kl_cosine = recon_kl_dot / (
                torch.sqrt(torch.tensor(recon_grad_norm_sq * kl_grad_norm_sq)) + 1e-8
            )

        optimizer.step()

        # Accumulate for epoch averages
        epoch_loss += loss.item()
        epoch_recon_loss += output.loss_recon.item()
        epoch_kl_loss += output.loss_kl.item()
        epoch_grad_norm += realized_grad_norm  # Store the realized (clipped) norm
        epoch_unclipped_grad_norm += (
            unclipped_grad_norm  # Store unclipped norm for comparison
        )

        if analyze_gradients:
            epoch_recon_grad_norm += recon_grad_norm
            epoch_kl_grad_norm += kl_grad_norm
            epoch_recon_contribution += recon_contribution
            epoch_kl_contribution += kl_contribution
            epoch_recon_total_cosine += recon_total_cosine.item()
            epoch_kl_total_cosine += kl_total_cosine.item()
            epoch_recon_kl_cosine += recon_kl_cosine.item()

        n_batches += 1

        # Update progress bar
        progress_bar.set_postfix(
            {
                "Loss": f"{loss.item():.3f}",
                "Recon": f"{output.loss_recon.item():.3f}",
                "KL": f"{output.loss_kl.item():.3f}",
                "GNorm": f"{realized_grad_norm:.2f}",
            }
        )

        if n_upd % log_interval == 0:
            if writer is not None:
                global_step = n_upd
                writer.add_scalar("Loss/Train", loss.item(), global_step)
                writer.add_scalar(
                    "Loss/Train/BCE", output.loss_recon.item(), global_step
                )
                writer.add_scalar("Loss/Train/KLD", output.loss_kl.item(), global_step)
                writer.add_scalar(
                    "GradNorm/Train/Realized", realized_grad_norm, global_step
                )
                writer.add_scalar(
                    "GradNorm/Train/Unclipped", unclipped_grad_norm, global_step
                )

                if analyze_gradients:
                    writer.add_scalar(
                        "GradNorm/Train/Recon", recon_grad_norm, global_step
                    )
                    writer.add_scalar("GradNorm/Train/KL", kl_grad_norm, global_step)
                    writer.add_scalar(
                        "GradContrib/Train/Recon", recon_contribution, global_step
                    )
                    writer.add_scalar(
                        "GradContrib/Train/KL", kl_contribution, global_step
                    )
                    writer.add_scalar(
                        "GradCosine/Train/Recon_Total",
                        recon_total_cosine.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "GradCosine/Train/KL_Total", kl_total_cosine.item(), global_step
                    )
                    writer.add_scalar(
                        "GradCosine/Train/Recon_KL", recon_kl_cosine.item(), global_step
                    )

    # Record epoch-level averages
    if history is not None and n_batches > 0:
        history.setdefault("epoch", []).append(epoch)
        history.setdefault("loss", []).append(epoch_loss / n_batches)
        history.setdefault("recon", []).append(epoch_recon_loss / n_batches)
        history.setdefault("kl", []).append(epoch_kl_loss / n_batches)
        history.setdefault("grad_norm_realized", []).append(
            epoch_grad_norm / n_batches
        )  # Renamed and clarified
        history.setdefault("grad_norm_unclipped", []).append(
            epoch_unclipped_grad_norm / n_batches
        )  # Added

        if analyze_gradients:
            history.setdefault("recon_grad_norm", []).append(
                epoch_recon_grad_norm / n_batches
            )  # Renamed for clarity
            history.setdefault("kl_grad_norm", []).append(
                epoch_kl_grad_norm / n_batches
            )  # Renamed for clarity
            history.setdefault("recon_contrib", []).append(
                epoch_recon_contribution / n_batches
            )
            history.setdefault("kl_contrib", []).append(
                epoch_kl_contribution / n_batches
            )
            history.setdefault("recon_total_cosine", []).append(
                epoch_recon_total_cosine / n_batches
            )
            history.setdefault("kl_total_cosine", []).append(
                epoch_kl_total_cosine / n_batches
            )
            history.setdefault("recon_kl_cosine", []).append(
                epoch_recon_kl_cosine / n_batches
            )

    return prev_updates + len(dataloader)


def test(
    model: VAE,
    dataloader,
    cur_step: int,
    device,
    writer=None,
    history: dict | None = None,
    epoch: int = 0,
    *,
    use_analytic_kl: bool = True,
    use_distributions: bool = False,
    n_latent_samples: int = 1,
):
    """
    Tests the model on the given data.

    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
        history: Dictionary to store test history.
        epoch (int): Current epoch number for history tracking.
        use_analytic_kl (bool): Whether to use manual analytic KL formula (True) or torch.distributions.kl.kl_divergence (False).
        use_distributions (bool): Whether to use torch.distributions for sampling.
        n_latent_samples (int): Number of latent samples per input for ELBO estimation.
    """
    model.eval()
    test_loss = 0.0
    test_recon_loss = 0.0
    test_kl_loss = 0.0

    last_output = None
    last_data = None

    # For KL per dimension computation
    kl_per_dim_accum = None
    n_batches_kl = 0

    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Testing"):
            data = data.to(device)

            output = model(
                data,
                compute_loss=True,
                use_analytic_kl=use_analytic_kl,
                use_distributions=use_distributions,
                n_samples=n_latent_samples,
            )

            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()

            # Compute KL per dimension using already computed values from forward pass
            mu = output.mu
            logvar = 2 * output.std.log()  # logvar = 2 * log(std)
            from plotting import compute_kl_per_dimension

            kl_per_dim_batch = compute_kl_per_dimension(mu, logvar)

            if kl_per_dim_accum is None:
                kl_per_dim_accum = kl_per_dim_batch.clone()
            else:
                kl_per_dim_accum += kl_per_dim_batch
            n_batches_kl += 1

            last_output = output
            last_data = data

    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)

    # Average KL per dimension across all test batches
    if kl_per_dim_accum is not None:
        kl_per_dim_avg = (kl_per_dim_accum / n_batches_kl).cpu().numpy().tolist()
    else:
        kl_per_dim_avg = []

    print(
        f"====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})"
    )

    # Print KL per dimension summary
    active_count = sum(1 for kl in kl_per_dim_avg if kl >= 0.1)
    print(f"====> Active units: {active_count}/{len(kl_per_dim_avg)} (threshold=0.1)")

    # Store test history at epoch level
    if history is not None:
        history.setdefault("epoch", []).append(epoch)
        history.setdefault("loss", []).append(float(test_loss))
        history.setdefault("recon", []).append(float(test_recon_loss))
        history.setdefault("kl", []).append(float(test_kl_loss))
        history.setdefault("kl_per_dim", []).append(kl_per_dim_avg)

    if writer is not None and last_output is not None and last_data is not None:
        writer.add_scalar("Loss/Test", test_loss, global_step=cur_step)
        writer.add_scalar(
            "Loss/Test/BCE", last_output.loss_recon.item(), global_step=cur_step
        )
        writer.add_scalar(
            "Loss/Test/KLD", last_output.loss_kl.item(), global_step=cur_step
        )

        # Log reconstructions
        writer.add_images(
            "Test/Reconstructions",
            torch.sigmoid(last_output.x_logits.view(-1, 1, 28, 28)),
            global_step=cur_step,
        )
        writer.add_images(
            "Test/Originals", last_data.view(-1, 1, 28, 28), global_step=cur_step
        )

        # Log random samples from the latent space
        z = torch.randn((16, last_output.z.shape[1]), device=device)
        samples = torch.sigmoid(model.decode(z))
        writer.add_images(
            "Test/Samples", samples.view(-1, 1, 28, 28), global_step=cur_step
        )

    return test_loss


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    path: str,
    best: bool = False,
):
    payload = {
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(payload, path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Variational Autoencoder on MNIST"
    )

    # Model architecture
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=10,
        help="Dimensionality of the latent space (default: 10)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Dimensionality of the hidden layer (default: 512)",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=784,
        help="Dimensionality of the input (default: 784 for MNIST)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for optimizer (default: 3e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for training and testing (default: 100)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Maximum gradient norm for clipping; set to None to disable (default: None)",
    )

    # VAE-specific options
    parser.add_argument(
        "--use-torch-kl",
        action="store_true",
        default=False,
        help="Use torch.distributions.kl.kl_divergence instead of manual analytic formula (default: use manual formula)",
    )
    parser.add_argument(
        "--use-distributions",
        action="store_true",
        default=False,
        help="Use torch.distributions for sampling (default: False)",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory for saving runs (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="How often to log training metrics (default: 100 steps)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )

    # Performance and analysis options
    parser.add_argument(
        "--analyze-gradients",
        action="store_true",
        default=False,
        help="Enable detailed gradient analysis (slower but provides gradient diagnostics)",
    )

    # Evaluation and visualization
    parser.add_argument(
        "--max-latent-batches",
        type=int,
        default=400,
        help="Max batches to collect for latent visualization (default: 400)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Number of samples to generate for visualization (default: 64)",
    )
    parser.add_argument(
        "--n-recon",
        type=int,
        default=16,
        help="Number of reconstructions to visualize (default: 16)",
    )
    parser.add_argument(
        "--interp-steps",
        type=int,
        default=15,
        help="Number of steps for latent interpolation between two datapoints (default: 15)",
    )
    parser.add_argument(
        "--interp-method",
        type=str,
        choices=["slerp", "lerp"],
        default="slerp",
        help="Interpolation method in latent space (default: slerp)",
    )
    parser.add_argument(
        "--n-latent-samples",
        type=int,
        default=1,
        help="Number of latent samples per input for ELBO estimation (default: 1)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Device setup
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            mps_available = False
            try:
                _backends = getattr(torch, "backends", None)
                _mps = (
                    getattr(_backends, "mps", None) if _backends is not None else None
                )
                mps_available = bool(_mps.is_available()) if _mps is not None else False
            except Exception:
                mps_available = False
            device = torch.device("mps") if mps_available else torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Data loaders
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Model
    config = VAEConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    )
    model = VAE(config).to(device)

    n_params = count_parameters(model)
    print(f"Number of parameters: {n_params:,}")
    print(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Setup directories
    if args.run_dir is None:
        run_dir = f"runs/mnist/vae_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        run_dir = args.run_dir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    fig_dir = os.path.join(run_dir, "figures")
    ensure_dir(ckpt_dir)
    ensure_dir(fig_dir)
    writer = SummaryWriter(run_dir)

    # Training loop
    prev_updates = 0
    train_history = {}
    test_history = {}
    best_val = float("inf")
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        prev_updates = train(
            model,
            train_loader,
            optimizer,
            prev_updates,
            device,
            writer,
            history=train_history,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            epoch=epoch + 1,
            use_analytic_kl=not args.use_torch_kl,
            use_distributions=args.use_distributions,
            n_latent_samples=args.n_latent_samples,
            analyze_gradients=args.analyze_gradients,
        )
        val_loss = test(
            model,
            test_loader,
            prev_updates,
            device,
            writer,
            history=test_history,
            epoch=epoch + 1,
            use_analytic_kl=not args.use_torch_kl,
            use_distributions=args.use_distributions,
            n_latent_samples=args.n_latent_samples,
        )

        # Save checkpoints
        save_checkpoint(
            model,
            optimizer,
            epoch=epoch + 1,
            step=prev_updates,
            path=os.path.join(ckpt_dir, "checkpoint_last.pt"),
        )
        # Also keep a rolling epoch checkpoint every N epochs
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch=epoch + 1,
                step=prev_updates,
                path=os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1:03d}.pt"),
            )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch=epoch + 1,
                step=prev_updates,
                path=os.path.join(ckpt_dir, "checkpoint_best.pt"),
                best=True,
            )

    # Post-training analysis figures
    # Use a small batch from test set for reconstructions
    first_batch = next(iter(test_loader))[0]
    # Reconstructions
    save_recon_figure(
        model,
        first_batch,
        device,
        os.path.join(fig_dir, "reconstructions.webp"),
        n=args.n_recon,
    )
    save_samples_figure(
        model,
        device,
        args.latent_dim,
        os.path.join(fig_dir, "samples.webp"),
        n=args.n_samples,
        grid=(8, 8),
    )
    # Combined interpolation figure (between examples + latent sweep)
    save_interpolation_combined_figure(
        model,
        test_loader,
        device,
        os.path.join(fig_dir, "interpolation.webp"),
        steps=args.interp_steps,
        method=args.interp_method,
        sweep_steps=15,
    )
    # Latent space
    Z, Y = collect_latents(
        model, test_loader, device, max_batches=args.max_latent_batches
    )
    save_latent_combined_figure(Z, Y, os.path.join(fig_dir, "mnist-2d-combined.webp"))
    save_latent_marginals(Z, os.path.join(fig_dir, "mnist-1d-hists.webp"))

    # Loss and grad norm curves (from epoch-level data)
    save_training_curves(
        train_history, test_history, os.path.join(fig_dir, "losses.webp")
    )
    # Unified gradient diagnostics plot (only if gradient analysis was enabled)
    if args.analyze_gradients:
        save_gradient_diagnostics(
            train_history, os.path.join(fig_dir, "grad-diagnostics.webp")
        )

    # KL per dimension diagnostics (the "Next Level" plot)
    save_kl_diagnostics_combined(
        test_history, os.path.join(fig_dir, "kl-diagnostics-combined.webp")
    )

    writer.close()


if __name__ == "__main__":
    main()
