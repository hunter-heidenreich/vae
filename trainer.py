"""
VAE Trainer with comprehensive management capabilities.
"""

import os
from typing import Any

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from model import VAE
from plotting import (collect_latents, compute_kl_per_dimension,
                      save_gradient_diagnostics,
                      save_interpolation_combined_figure,
                      save_kl_diagnostics_combined,
                      save_latent_combined_figure, save_latent_marginals,
                      save_recon_figure, save_samples_figure,
                      save_training_curves)
from trainer_config import TrainerConfig


class VAETrainer:
    """
    A comprehensive trainer for Variational Autoencoders with full management capabilities.

    Handles training, validation, checkpointing, logging, directory setup, optimizer creation,
    and post-training analysis with extensive gradient analysis and configurable options.
    """

    def __init__(
        self,
        model: VAE,
        config: TrainerConfig,
    ):
        """
        Initialize the VAE trainer.

        Args:
            model: The VAE model to train
            config: TrainerConfig with all training parameters
        """
        self.model = model
        self.config = config

        # Setup device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        # Setup random seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup directories
        self.run_dir = config.get_run_dir()
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.fig_dir = os.path.join(self.run_dir, "figures")
        self._ensure_directories()

        # Setup TensorBoard writer
        self.writer = SummaryWriter(self.run_dir)

        # Training state
        self.global_step = 0
        self.train_history = {}
        self.val_history = {}

        print("VAE Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Run directory: {self.run_dir}")
        print(f"  Parameters: {self._count_parameters():,}")

    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                mps_available = False
                try:
                    _backends = getattr(torch, "backends", None)
                    _mps = (
                        getattr(_backends, "mps", None)
                        if _backends is not None
                        else None
                    )
                    mps_available = (
                        bool(_mps.is_available()) if _mps is not None else False
                    )
                except Exception:
                    mps_available = False
                return torch.device("mps") if mps_available else torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def _ensure_directories(self):
        """Create necessary directories."""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def _count_parameters(self) -> int:
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> int:
        """
        Train the model for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number (1-indexed)

        Returns:
            Number of training steps performed this epoch
        """
        self.model.train()

        # Accumulators for epoch-level metrics
        epoch_metrics = {
            "loss": 0.0,
            "recon": 0.0,
            "kl": 0.0,
            "grad_norm_realized": 0.0,
            "grad_norm_unclipped": 0.0,
        }

        # Gradient analysis accumulators (only if enabled)
        if self.config.analyze_gradients:
            grad_metrics = {
                "recon_grad_norm": 0.0,
                "kl_grad_norm": 0.0,
                "recon_kl_cosine": 0.0,
                "recon_contrib": 0.0,
                "kl_contrib": 0.0,
            }
            epoch_metrics.update(grad_metrics)

        n_batches = 0
        steps_this_epoch = 0

        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

        for _, (data, _) in enumerate(progress_bar):
            data = data.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(
                data,
                compute_loss=True,
                reconstruct=False,
            )

            # Gradient analysis (if enabled)
            step_grad_metrics = {}
            if self.config.analyze_gradients:
                step_grad_metrics = self._compute_gradient_metrics(output)

            # Backward pass
            output.loss.backward()

            # Gradient norms and clipping
            grad_norms = self._handle_gradient_clipping()

            # Optimizer step
            self.optimizer.step()

            # Update metrics (cache .item() calls)
            loss_item = output.loss.item()
            recon_item = output.loss_recon.item()
            kl_item = output.loss_kl.item()

            epoch_metrics["loss"] += loss_item
            epoch_metrics["recon"] += recon_item
            epoch_metrics["kl"] += kl_item
            epoch_metrics["grad_norm_realized"] += grad_norms["realized"]
            epoch_metrics["grad_norm_unclipped"] += grad_norms["unclipped"]

            # Add gradient metrics if enabled
            for key, value in step_grad_metrics.items():
                epoch_metrics[key] += value

            n_batches += 1
            steps_this_epoch += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss_item:.3f}",
                    "Recon": f"{recon_item:.3f}",
                    "KL": f"{kl_item:.3f}",
                    "GNorm": f"{grad_norms['realized']:.2f}",
                }
            )

            # Logging
            if self.global_step % self.config.log_interval == 0:
                self._log_training_step(output, grad_norms, step_grad_metrics)

        # Record epoch-level averages
        self._record_epoch_metrics(epoch_metrics, n_batches, epoch, is_train=True)

        return steps_this_epoch

    def validate(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Validate the model on the given dataset.

        Args:
            dataloader: Validation data loader
            epoch: Current epoch number (1-indexed)

        Returns:
            Average validation loss
        """
        self.model.eval()

        val_metrics = {
            "loss": 0.0,
            "recon": 0.0,
            "kl": 0.0,
        }

        # For KL per dimension computation
        kl_per_dim_accum = None

        with torch.no_grad():
            for data, _ in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
                data = data.to(self.device)
                output = self.model(
                    data,
                    compute_loss=True,
                )

                val_metrics["loss"] += output.loss.item()
                val_metrics["recon"] += output.loss_recon.item()
                val_metrics["kl"] += output.loss_kl.item()

                # Compute KL per dimension
                mu = output.mu
                logvar = 2 * output.std.log()  # logvar = 2 * log(std)
                kl_per_dim_batch = compute_kl_per_dimension(mu, logvar)

                if kl_per_dim_accum is None:
                    kl_per_dim_accum = kl_per_dim_batch.clone()
                else:
                    kl_per_dim_accum += kl_per_dim_batch

        # Compute averages
        n_batches = len(dataloader)
        for key in val_metrics:
            val_metrics[key] /= n_batches

        # Average KL per dimension
        kl_per_dim_avg = []
        if kl_per_dim_accum is not None:
            kl_per_dim_avg = (kl_per_dim_accum / n_batches).cpu().numpy().tolist()

        # Print validation results
        print(
            f"====> Validation loss: {val_metrics['loss']:.4f} "
            f"(BCE: {val_metrics['recon']:.4f}, KLD: {val_metrics['kl']:.4f})"
        )

        active_count = sum(1 for kl in kl_per_dim_avg if kl >= 0.1)
        print(
            f"====> Active units: {active_count}/{len(kl_per_dim_avg)} (threshold=0.1)"
        )

        # Record validation history
        self._record_epoch_metrics(val_metrics, 1, epoch, is_train=False)
        # Add kl_per_dim separately since it's a different type
        self.val_history.setdefault("kl_per_dim", []).append(kl_per_dim_avg)

        # TensorBoard logging
        self._log_validation_step(val_metrics)

        return val_metrics["loss"]

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """
        Complete training loop with validation and checkpointing.

        Args:
            train_loader: Training data loader
            test_loader: Validation/test data loader
        """
        print(f"Starting training for {self.config.num_epochs} epochs...")

        best_val = float("inf")

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            # Train for one epoch
            self.train_epoch(train_loader, epoch + 1)

            # Validate
            val_loss = self.validate(test_loader, epoch + 1)

            # Save checkpoints
            self.save_checkpoint(
                epoch=epoch + 1,
                filepath=os.path.join(self.ckpt_dir, "checkpoint_last.pt"),
            )

            # Also keep a rolling epoch checkpoint every N epochs
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(
                    epoch=epoch + 1,
                    filepath=os.path.join(
                        self.ckpt_dir, f"checkpoint_epoch_{epoch + 1:03d}.pt"
                    ),
                )

            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(
                    epoch=epoch + 1,
                    filepath=os.path.join(self.ckpt_dir, "checkpoint_best.pt"),
                    is_best=True,
                )

        print(f"Training completed! Best validation loss: {best_val:.4f}")

    def generate_analysis_plots(self, test_loader: DataLoader):
        """
        Generate all post-training analysis plots and figures.

        Args:
            test_loader: Test data loader for analysis
        """
        print("Generating analysis plots...")

        # Use a small batch from test set for reconstructions
        first_batch = next(iter(test_loader))[0]

        # Reconstructions
        save_recon_figure(
            self.model,
            first_batch,
            self.device,
            os.path.join(self.fig_dir, "reconstructions.webp"),
            n=self.config.n_recon,
        )

        # Generated samples
        save_samples_figure(
            self.model,
            self.device,
            self.model.config.latent_dim,
            os.path.join(self.fig_dir, "samples.webp"),
            n=self.config.n_samples,
            grid=(8, 8),
        )

        # Combined interpolation figure (between examples + latent sweep)
        save_interpolation_combined_figure(
            self.model,
            test_loader,
            self.device,
            os.path.join(self.fig_dir, "interpolation.webp"),
            steps=self.config.interp_steps,
            method=self.config.interp_method,
            sweep_steps=15,
        )

        # Latent space analysis
        Z, Y = collect_latents(
            self.model,
            test_loader,
            self.device,
            max_batches=self.config.max_latent_batches,
        )
        save_latent_combined_figure(
            Z, Y, os.path.join(self.fig_dir, "mnist-2d-combined.webp")
        )
        save_latent_marginals(Z, os.path.join(self.fig_dir, "mnist-1d-hists.webp"))

        # Training curves
        save_training_curves(
            self.train_history,
            self.val_history,
            os.path.join(self.fig_dir, "losses.webp"),
        )

        # Gradient diagnostics (only if gradient analysis was enabled)
        if self.config.analyze_gradients:
            save_gradient_diagnostics(
                self.train_history, os.path.join(self.fig_dir, "grad-diagnostics.webp")
            )

        # KL per dimension diagnostics
        save_kl_diagnostics_combined(
            self.val_history, os.path.join(self.fig_dir, "kl-diagnostics-combined.webp")
        )

        print(f"Analysis plots saved to {self.fig_dir}")

    def _compute_gradient_metrics(self, output) -> dict[str, float]:
        """
        Compute detailed gradient analysis metrics.

        Args:
            output: VAEOutput from forward pass

        Returns:
            Dictionary of gradient metrics
        """
        params = list(self.model.parameters())

        # Compute reconstruction gradients
        recon_grads = torch.autograd.grad(
            output.loss_recon,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        recon_grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(recon_grads, params)
        ]
        recon_grad_norm = torch.sqrt(
            torch.stack([g.norm(2).pow(2) for g in recon_grads]).sum()
        ).item()

        # Compute KL gradients
        kl_grads = torch.autograd.grad(
            output.loss_kl,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        kl_grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(kl_grads, params)
        ]
        kl_grad_norm = torch.sqrt(
            torch.stack([g.norm(2).pow(2) for g in kl_grads]).sum()
        ).item()

        # Compute cosine similarity between recon and KL gradients
        recon_kl_cosine = 0.0
        if recon_grad_norm > 0 and kl_grad_norm > 0:
            # Compute dot product
            dot_product = (
                torch.stack([(r * k).sum() for r, k in zip(recon_grads, kl_grads)])
                .sum()
                .item()
            )
            recon_kl_cosine = dot_product / (recon_grad_norm * kl_grad_norm)

        # Compute relative contributions to total gradient direction
        total_grad_norm_sq = recon_grad_norm**2 + kl_grad_norm**2
        recon_contrib = 0.0
        kl_contrib = 0.0
        if total_grad_norm_sq > 0:
            recon_contrib = recon_grad_norm**2 / total_grad_norm_sq
            kl_contrib = kl_grad_norm**2 / total_grad_norm_sq

        return {
            "recon_grad_norm": recon_grad_norm,
            "kl_grad_norm": kl_grad_norm,
            "recon_kl_cosine": recon_kl_cosine,
            "recon_contrib": recon_contrib,
            "kl_contrib": kl_contrib,
        }

    def _handle_gradient_clipping(self) -> dict[str, float]:
        """
        Handle gradient clipping and compute gradient norms.

        Returns:
            Dictionary with 'realized' and 'unclipped' gradient norms
        """
        # Get unclipped gradient norm
        total_grads_unclipped = [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in self.model.parameters()
        ]
        unclipped_grad_norm = torch.sqrt(
            torch.stack([g.norm(2).pow(2) for g in total_grads_unclipped]).sum()
        ).item()

        # Apply clipping if specified
        if self.config.max_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            realized_grad_norm = min(unclipped_grad_norm, self.config.max_grad_norm)
        else:
            realized_grad_norm = unclipped_grad_norm

        return {
            "realized": realized_grad_norm,
            "unclipped": unclipped_grad_norm,
        }

    def _log_training_step(
        self, output, grad_norms: dict[str, float], grad_metrics: dict[str, float]
    ):
        """Log training step metrics to TensorBoard."""
        step = self.global_step
        self.writer.add_scalar("Loss/Train", output.loss.item(), step)
        self.writer.add_scalar("Loss/Train/BCE", output.loss_recon.item(), step)
        self.writer.add_scalar("Loss/Train/KLD", output.loss_kl.item(), step)
        self.writer.add_scalar("GradNorm/Train/Realized", grad_norms["realized"], step)
        self.writer.add_scalar(
            "GradNorm/Train/Unclipped", grad_norms["unclipped"], step
        )

        for key, value in grad_metrics.items():
            category = key.split("_")[0]  # recon, kl
            metric_name = "_".join(key.split("_")[1:])  # grad_norm, contribution, etc.
            self.writer.add_scalar(
                f"Grad{metric_name.title()}/Train/{category.title()}", value, step
            )

    def _log_validation_step(self, val_metrics: dict[str, Any]):
        """Log validation step metrics to TensorBoard."""
        step = self.global_step
        self.writer.add_scalar("Loss/Test", val_metrics["loss"], step)
        self.writer.add_scalar("Loss/Test/BCE", val_metrics["recon"], step)
        self.writer.add_scalar("Loss/Test/KLD", val_metrics["kl"], step)

    def _record_epoch_metrics(
        self, metrics: dict[str, Any], n_batches: int, epoch: int, is_train: bool
    ):
        """Record epoch-level metrics in history."""
        history = self.train_history if is_train else self.val_history

        history.setdefault("epoch", []).append(epoch)

        for key, value in metrics.items():
            if key == "kl_per_dim":  # Special handling for list values
                history.setdefault(key, []).append(value)
            else:  # Numeric values - compute average
                avg_value = value / n_batches
                history.setdefault(key, []).append(avg_value)

    def save_checkpoint(self, epoch: int, filepath: str, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            filepath: Path to save checkpoint
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": self.train_history,
            "val_history": self.val_history,
            "config": self.config,
            "is_best": is_best,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.train_history = checkpoint.get("train_history", {})
        self.val_history = checkpoint.get("val_history", {})

        return checkpoint

    def close(self):
        """Clean up resources."""
        if self.writer is not None:
            self.writer.close()

    def get_train_history(self) -> dict[str, Any]:
        """Get training history."""
        return self.train_history.copy()

    def get_val_history(self) -> dict[str, Any]:
        """Get validation history."""
        return self.val_history.copy()
