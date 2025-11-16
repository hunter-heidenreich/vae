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

from metrics import MetricsAccumulator, TrainingHistory
from model import VAE
from plotting import (
    collect_all_latent_data,
    compute_kl_per_dimension,
    make_plot_path,
    save_gradient_diagnostics,
    save_interpolation_and_sweep_figures,
    save_kl_diagnostics_separate,
    save_latent_combined_figure,
    save_latent_evolution_plots,
    save_latent_marginals,
    save_logvar_combined_figure,
    save_logvar_marginals,
    save_parameter_diagnostics,
    save_recon_figure,
    save_samples_figure,
    save_training_curves,
)
from trainer_config import TrainerConfig

ACTIVE_UNIT_KL_THRESHOLD = 0.1
DEFAULT_INTERPOLATION_SWEEP_STEPS = 15
DEFAULT_SAMPLES_GRID_SIZE = (8, 8)


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

        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.run_dir = config.get_run_dir()
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.fig_dir = os.path.join(self.run_dir, "figures")
        self._ensure_directories()

        self.writer = SummaryWriter(self.run_dir)

        self.global_step = 0
        self.history = TrainingHistory()

        self.previous_params = None
        self.param_norms_history = []

        print("VAE Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Run directory: {self.run_dir}")
        print(f"  Parameters: {self._count_parameters():,}")

    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            # Check for MPS (Apple Silicon) availability
            try:
                backends = getattr(torch, "backends", None)
                if backends is not None:
                    mps = getattr(backends, "mps", None)
                    if mps is not None and mps.is_available():
                        return torch.device("mps")
            except Exception:
                pass

            return torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def _ensure_directories(self):
        """Create necessary directories."""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def _count_parameters(self) -> int:
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _get_current_lr(self) -> float:
        """
        Get current learning rate with linear warmup.

        Returns:
            Current learning rate based on global step and warmup configuration
        """
        if (
            self.config.warmup_steps <= 0
            or self.global_step >= self.config.warmup_steps
        ):
            return self.config.learning_rate

        # Linear warmup: lr = base_lr * ((current_step + 1) / warmup_steps)
        # We add 1 to current_step so that step 0 has lr > 0
        warmup_factor = (self.global_step + 1) / self.config.warmup_steps
        current_lr = self.config.learning_rate * warmup_factor

        return current_lr

    def _compute_grad_norm(self, gradients: list[torch.Tensor]) -> float:
        """Compute L2 norm of a list of gradients efficiently."""
        if not gradients:
            return 0.0
        # Use a generator expression with Python's sum to avoid
        # creating intermediate tensor via torch.stack
        # Start with first gradient's squared norm to ensure we have a tensor
        grad_iter = iter(gradients)
        total_norm_sq = next(grad_iter).norm(2).pow(2)
        for g in grad_iter:
            total_norm_sq = total_norm_sq + g.norm(2).pow(2)
        return torch.sqrt(total_norm_sq).item()

    def _copy_model_parameters(self) -> dict[str, list[torch.Tensor]]:
        """Create a deep copy of current model parameters for change tracking."""
        return {
            "encoder": [
                p.clone().detach() for p in self.model.get_encoder_parameters()
            ],
            "decoder": [
                p.clone().detach() for p in self.model.get_decoder_parameters()
            ],
        }

    def _compute_parameter_changes(self) -> dict[str, float]:
        """Compute parameter change norms since last epoch."""
        if self.previous_params is None:
            self.previous_params = self._copy_model_parameters()
            return {
                "encoder_param_change_norm": 0.0,
                "decoder_param_change_norm": 0.0,
                "total_param_change_norm": 0.0,
                "encoder_param_change_rel": 0.0,
                "decoder_param_change_rel": 0.0,
                "total_param_change_rel": 0.0,
            }

        current_encoder_params = self.model.get_encoder_parameters()
        current_decoder_params = self.model.get_decoder_parameters()

        encoder_changes = [
            (curr - prev).detach()
            for curr, prev in zip(
                current_encoder_params, self.previous_params["encoder"]
            )
        ]
        decoder_changes = [
            (curr - prev).detach()
            for curr, prev in zip(
                current_decoder_params, self.previous_params["decoder"]
            )
        ]

        encoder_change_norm = self._compute_grad_norm(encoder_changes)
        decoder_change_norm = self._compute_grad_norm(decoder_changes)
        total_change_norm = torch.sqrt(
            torch.tensor(encoder_change_norm**2 + decoder_change_norm**2)
        ).item()

        encoder_current_norm = self._compute_grad_norm(
            [p.detach() for p in current_encoder_params]
        )
        decoder_current_norm = self._compute_grad_norm(
            [p.detach() for p in current_decoder_params]
        )
        total_current_norm = torch.sqrt(
            torch.tensor(encoder_current_norm**2 + decoder_current_norm**2)
        ).item()

        encoder_change_rel = (
            encoder_change_norm / encoder_current_norm
            if encoder_current_norm > 0
            else 0.0
        )
        decoder_change_rel = (
            decoder_change_norm / decoder_current_norm
            if decoder_current_norm > 0
            else 0.0
        )
        total_change_rel = (
            total_change_norm / total_current_norm if total_current_norm > 0 else 0.0
        )

        self.previous_params = self._copy_model_parameters()

        return {
            "encoder_param_change_norm": encoder_change_norm,
            "decoder_param_change_norm": decoder_change_norm,
            "total_param_change_norm": total_change_norm,
            "encoder_param_change_rel": encoder_change_rel,
            "decoder_param_change_rel": decoder_change_rel,
            "total_param_change_rel": total_change_rel,
        }

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

        metrics_accumulator = MetricsAccumulator(self.config.analyze_gradients)
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

        for _, (data, _) in enumerate(progress_bar):
            should_compute_metrics = self.global_step % self.config.log_interval == 0
            step_metrics = self._train_step(data, return_metrics=should_compute_metrics)

            self.global_step += 1

            if step_metrics is not None:
                metrics_accumulator.add_step(step_metrics)
                self._log_training_step_from_metrics(step_metrics)

                self.history.record_step_metrics(
                    step_metrics, self.global_step - 1, is_train=True
                )

                latest_metrics = metrics_accumulator.get_latest()
                progress_bar.set_postfix(
                    {
                        "Loss": f"{latest_metrics.get('loss', 0):.3f}",
                        "Recon": f"{latest_metrics.get('recon', 0):.3f}",
                        "KL": f"{latest_metrics.get('kl', 0):.3f}",
                        "GNorm": f"{latest_metrics.get('grad_norm_realized', 0):.2f}",
                        "LR": f"{latest_metrics.get('learning_rate', 0):.1e}",
                    }
                )

        epoch_averages = metrics_accumulator.get_averages()

        param_norms = self.model.compute_parameter_norms()
        param_changes = self._compute_parameter_changes()
        epoch_averages.update(param_norms)
        epoch_averages.update(param_changes)

        self.history.record_epoch_metrics(epoch_averages, epoch, is_train=True)

        return metrics_accumulator.get_count()

    def _train_step(
        self, data: torch.Tensor, return_metrics: bool = False
    ) -> dict[str, float] | None:
        """
        Execute a single training step and return metrics.

        Args:
            data: Input batch data
            return_metrics: Whether to return metrics dictionary

        Returns:
            Dictionary of step metrics
        """
        data = data.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(
            data,
            compute_loss=True,
            reconstruct=False,
        )

        # Gradient analysis (if enabled) - MUST be done before backward() frees the graph
        step_grad_metrics = {}
        if return_metrics and self.config.analyze_gradients:
            step_grad_metrics = self._compute_gradient_metrics(output)

        # Get current learning rate for warmup (do this BEFORE backward pass to capture correct LR)
        current_lr = self._get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        output.loss.backward()
        grad_norms = self._handle_gradient_clipping()
        self.optimizer.step()

        if not return_metrics:
            return

        step_metrics = {
            "loss": output.loss.item(),
            "recon": output.loss_recon.item(),
            "kl": output.loss_kl.item(),
            "grad_norm_realized": grad_norms["realized"],
            "grad_norm_unclipped": grad_norms["unclipped"],
            "learning_rate": current_lr,
        }

        step_metrics.update(step_grad_metrics)

        return step_metrics

    def validate(self, dataloader: DataLoader, epoch: int) -> dict[str, float]:
        """
        Validate the model on the given dataset.

        Args:
            dataloader: Validation data loader
            epoch: Current epoch number (1-indexed)

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        metrics_accumulator = MetricsAccumulator(
            enable_gradient_analysis=False, enable_latent_tracking=True
        )

        kl_per_dim_accum = None

        with torch.no_grad():
            for data, _ in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
                data = data.to(self.device)
                output = self.model(
                    data,
                    compute_loss=True,
                )

                step_metrics = {
                    "loss": output.loss.item(),
                    "recon": output.loss_recon.item(),
                    "kl": output.loss_kl.item(),
                }
                metrics_accumulator.add_step(step_metrics)

                mu = output.mu
                std = output.std

                mu_means_batch = mu.mean(dim=0).cpu().numpy().tolist()
                mu_stds_batch = mu.std(dim=0).cpu().numpy().tolist()
                sigma_means_batch = std.mean(dim=0).cpu().numpy().tolist()
                sigma_stds_batch = std.std(dim=0).cpu().numpy().tolist()

                metrics_accumulator.add_latent_stats(
                    mu_means_batch, mu_stds_batch, sigma_means_batch, sigma_stds_batch
                )

                # Check if we're using log-variance parameterization
                is_logvar = not (
                    self.model.config.use_softplus_std
                    or self.model.config.bound_std is not None
                )
                if is_logvar:
                    # logvar = 2 * log(std) when std = exp(0.5 * logvar)
                    logvar = 2 * std.log()
                    kl_per_dim_batch = compute_kl_per_dimension(
                        mu, logvar, is_logvar=True
                    )
                else:
                    kl_per_dim_batch = compute_kl_per_dimension(
                        mu, std, is_logvar=False
                    )

                if kl_per_dim_accum is None:
                    kl_per_dim_accum = kl_per_dim_batch.clone()
                else:
                    kl_per_dim_accum += kl_per_dim_batch

        val_metrics = metrics_accumulator.get_averages()
        n_batches = metrics_accumulator.get_count()

        kl_per_dim_avg = []
        if kl_per_dim_accum is not None:
            kl_per_dim_avg = (kl_per_dim_accum / n_batches).cpu().numpy().tolist()

        print(
            f"====> Validation loss: {val_metrics['loss']:.4f} "
            f"(BCE: {val_metrics['recon']:.4f}, KLD: {val_metrics['kl']:.4f})"
        )

        active_count = sum(1 for kl in kl_per_dim_avg if kl >= ACTIVE_UNIT_KL_THRESHOLD)
        print(
            f"====> Active units: {active_count}/{len(kl_per_dim_avg)} (threshold={ACTIVE_UNIT_KL_THRESHOLD})"
        )

        # Add parameter norms to validation metrics (but not parameter changes, as those are training-specific)
        param_norms = self.model.compute_parameter_norms()
        val_metrics.update(param_norms)

        self.history.record_epoch_metrics(val_metrics, epoch, is_train=False)
        self.history.val_history.setdefault("kl_per_dim", []).append(kl_per_dim_avg)

        latent_stats = metrics_accumulator.get_latent_statistics()
        for stat_name, stat_values in latent_stats.items():
            self.history.val_history.setdefault(stat_name, []).append(stat_values)

        self._log_validation_step(val_metrics)

        return val_metrics

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """
        Complete training loop with validation and checkpointing.

        Args:
            train_loader: Training data loader
            test_loader: Validation/test data loader
        """
        print(f"Starting training for {self.config.num_epochs} epochs...")

        best_val_loss = float("inf")
        best_val_metrics = None
        best_epoch = 0
        final_val_metrics = None

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            self.train_epoch(train_loader, epoch + 1)

            val_metrics = self.validate(test_loader, epoch + 1)
            val_loss = val_metrics["loss"]
            final_val_metrics = val_metrics

            self.save_checkpoint(
                epoch=epoch + 1,
                filepath=os.path.join(self.ckpt_dir, "checkpoint_last.pt"),
            )

            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(
                    epoch=epoch + 1,
                    filepath=os.path.join(
                        self.ckpt_dir,
                        f"checkpoint_epoch_{epoch + 1:03d}.pt",
                    ),
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_val_metrics = val_metrics.copy()
                self.save_checkpoint(
                    epoch=epoch + 1,
                    filepath=os.path.join(self.ckpt_dir, "checkpoint_best.pt"),
                    is_best=True,
                )

        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")

        self.best_val_metrics = best_val_metrics
        self.best_epoch = best_epoch
        self.final_val_metrics = final_val_metrics

    def load_best_model_for_generation(self):
        """
        Load the best model weights for generation/evaluation, preserving current training history.

        This method only loads the model weights, not the training history, so we can use
        the best model for generation while keeping the complete training history for plotting.
        """
        best_checkpoint_path = os.path.join(self.ckpt_dir, "checkpoint_best.pt")
        if os.path.exists(best_checkpoint_path):
            print(f"Loading best model checkpoint from: {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)

            # Only load model weights, preserve current training history and other state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            best_epoch = checkpoint.get("epoch", "unknown")
            print(f"Loaded best model weights from epoch {best_epoch}")
            return best_epoch
        else:
            print("Warning: Best model checkpoint not found, using current model state")
            return None

    def generate_analysis_plots(self, test_loader: DataLoader):
        """
        Generate all post-training analysis plots and figures.

        Uses complete training history for plotting, but loads best model for generation tasks.

        Args:
            test_loader: Test data loader for analysis
        """
        print("Generating analysis plots...")

        # === PART 1: PLOTTING (using complete training history) ===

        save_training_curves(
            self.history.get_train_history(),
            self.history.get_val_history(),
            self.history.get_train_step_history(),
            self.fig_dir,
            best_epoch=getattr(self, "best_epoch", None),
        )

        if self.config.analyze_gradients:
            save_gradient_diagnostics(
                self.history.get_train_history(),
                self.history.get_train_step_history(),
                self.fig_dir,
                best_epoch=getattr(self, "best_epoch", None),
            )

        save_parameter_diagnostics(
            self.history.get_train_history(),
            self.history.get_val_history(),
            self.fig_dir,
            best_epoch=getattr(self, "best_epoch", None),
        )

        save_kl_diagnostics_separate(
            self.history.get_val_history(),
            self.fig_dir,
        )

        save_latent_evolution_plots(
            self.history.get_val_history(),
            self.fig_dir,
        )

        # === PART 2: GENERATION (using best model) ===

        print("Loading best model for generation tasks...")
        self.load_best_model_for_generation()

        first_batch = next(iter(test_loader))[0]

        save_recon_figure(
            self.model,
            first_batch,
            self.device,
            make_plot_path(self.fig_dir, "reconstructions", group="generation"),
            n=self.config.n_recon,
        )

        save_samples_figure(
            self.model,
            self.device,
            self.model.config.latent_dim,
            make_plot_path(self.fig_dir, "samples", group="generation"),
            grid=DEFAULT_SAMPLES_GRID_SIZE,
        )

        save_interpolation_and_sweep_figures(
            self.model,
            test_loader,
            self.device,
            make_plot_path(self.fig_dir, "generation", group="generation"),
            steps=self.config.interp_steps,
            method=self.config.interp_method,
            sweep_steps=DEFAULT_INTERPOLATION_SWEEP_STEPS,
        )

        # OPTIMIZED: Single pass over data
        print("Collecting latent space data (single pass)...")
        Z, Mu, Std, Y = collect_all_latent_data(
            self.model,
            test_loader,
            self.device,
            max_batches=self.config.max_latent_batches,
        )

        save_latent_combined_figure(
            Z,
            Y,
            make_plot_path(self.fig_dir, "latent", "combined", "latent_space"),
        )
        save_latent_marginals(
            Z,
            make_plot_path(self.fig_dir, "latent", "marginals", "latent_space"),
        )

        LogVar = np.log(Std**2)
        save_logvar_combined_figure(
            LogVar,
            Y,
            make_plot_path(self.fig_dir, "logvar", "combined", "latent_space"),
        )
        save_logvar_marginals(
            LogVar,
            make_plot_path(self.fig_dir, "logvar", "marginals", "latent_space"),
        )

        print(f"Analysis plots saved to {self.fig_dir}")

    def _compute_gradient_metrics(self, output) -> dict[str, float]:
        """
        Compute detailed gradient analysis metrics with encoder/decoder separation.

        Args:
            output: VAEOutput from forward pass

        Returns:
            Dictionary of gradient metrics including encoder/decoder breakdown
        """
        encoder_params = self.model.get_encoder_parameters()
        decoder_params = self.model.get_decoder_parameters()
        all_params = list(self.model.parameters())

        # === RECONSTRUCTION GRADIENTS ===

        # Total reconstruction gradients
        recon_grads = torch.autograd.grad(
            output.loss_recon,
            all_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        recon_grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(recon_grads, all_params)
        ]
        recon_grad_norm = self._compute_grad_norm(recon_grads)

        # Encoder-only reconstruction gradients
        recon_encoder_grads = torch.autograd.grad(
            output.loss_recon,
            encoder_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        recon_encoder_grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(recon_encoder_grads, encoder_params)
        ]
        recon_encoder_grad_norm = self._compute_grad_norm(recon_encoder_grads)

        # Decoder-only reconstruction gradients
        recon_decoder_grads = torch.autograd.grad(
            output.loss_recon,
            decoder_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        recon_decoder_grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(recon_decoder_grads, decoder_params)
        ]
        recon_decoder_grad_norm = self._compute_grad_norm(recon_decoder_grads)

        # === KL GRADIENTS ===

        # Total KL gradients (should be encoder-only)
        kl_grads = torch.autograd.grad(
            output.loss_kl,
            all_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        kl_grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(kl_grads, all_params)
        ]
        kl_grad_norm = self._compute_grad_norm(kl_grads)

        # Encoder-only KL gradients (should match total KL)
        kl_encoder_grads = torch.autograd.grad(
            output.loss_kl,
            encoder_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        kl_encoder_grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(kl_encoder_grads, encoder_params)
        ]
        kl_encoder_grad_norm = self._compute_grad_norm(kl_encoder_grads)

        # Decoder KL gradients (should be ~0)
        kl_decoder_grads = torch.autograd.grad(
            output.loss_kl,
            decoder_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        kl_decoder_grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(kl_decoder_grads, decoder_params)
        ]
        kl_decoder_grad_norm = self._compute_grad_norm(kl_decoder_grads)

        # === TOTAL GRADIENTS (RECON + KL) ===

        # Total encoder gradients (recon + KL on encoder)
        total_encoder_grad_norm_sq = (
            recon_encoder_grad_norm**2 + kl_encoder_grad_norm**2
        )
        total_encoder_grad_norm = torch.sqrt(
            torch.tensor(total_encoder_grad_norm_sq)
        ).item()

        # Total decoder gradients (only recon, KL should be ~0)
        total_decoder_grad_norm = recon_decoder_grad_norm  # KL decoder should be ~0

        # === COSINE SIMILARITIES ===

        # Original total recon vs KL cosine
        recon_kl_cosine = 0.0
        if recon_grad_norm > 0 and kl_grad_norm > 0:
            grad_pairs = list(zip(recon_grads, kl_grads))
            if grad_pairs:
                dot_product = (grad_pairs[0][0] * grad_pairs[0][1]).sum()
                for r, k in grad_pairs[1:]:
                    dot_product = dot_product + (r * k).sum()
                recon_kl_cosine = dot_product.item() / (recon_grad_norm * kl_grad_norm)

        # Encoder: recon vs KL cosine
        recon_kl_encoder_cosine = 0.0
        if recon_encoder_grad_norm > 0 and kl_encoder_grad_norm > 0:
            grad_pairs = list(zip(recon_encoder_grads, kl_encoder_grads))
            if grad_pairs:
                dot_product = (grad_pairs[0][0] * grad_pairs[0][1]).sum()
                for r, k in grad_pairs[1:]:
                    dot_product = dot_product + (r * k).sum()
                recon_kl_encoder_cosine = dot_product.item() / (
                    recon_encoder_grad_norm * kl_encoder_grad_norm
                )

        # === RELATIVE CONTRIBUTIONS ===

        # Original total contributions
        total_grad_norm_sq = recon_grad_norm**2 + kl_grad_norm**2
        recon_contrib = 0.0
        kl_contrib = 0.0
        if total_grad_norm_sq > 0:
            recon_contrib = recon_grad_norm**2 / total_grad_norm_sq
            kl_contrib = kl_grad_norm**2 / total_grad_norm_sq

        # Encoder contributions (recon vs KL on encoder)
        encoder_grad_norm_sq = recon_encoder_grad_norm**2 + kl_encoder_grad_norm**2
        recon_encoder_contrib = 0.0
        kl_encoder_contrib = 0.0
        if encoder_grad_norm_sq > 0:
            recon_encoder_contrib = recon_encoder_grad_norm**2 / encoder_grad_norm_sq
            kl_encoder_contrib = kl_encoder_grad_norm**2 / encoder_grad_norm_sq

        return {
            # === ORIGINAL METRICS ===
            "recon_grad_norm": recon_grad_norm,
            "kl_grad_norm": kl_grad_norm,
            "recon_kl_cosine": recon_kl_cosine,
            "recon_contrib": recon_contrib,
            "kl_contrib": kl_contrib,
            # === RECONSTRUCTION GRADIENTS BY COMPONENT ===
            "recon_encoder_grad_norm": recon_encoder_grad_norm,
            "recon_decoder_grad_norm": recon_decoder_grad_norm,
            # === KL GRADIENTS BY COMPONENT ===
            "kl_encoder_grad_norm": kl_encoder_grad_norm,
            "kl_decoder_grad_norm": kl_decoder_grad_norm,  # Should be ~0
            # === TOTAL GRADIENTS BY COMPONENT ===
            "total_encoder_grad_norm": total_encoder_grad_norm,
            "total_decoder_grad_norm": total_decoder_grad_norm,
            # === ENCODER-SPECIFIC ANALYSIS ===
            "recon_kl_encoder_cosine": recon_kl_encoder_cosine,
            "recon_encoder_contrib": recon_encoder_contrib,
            "kl_encoder_contrib": kl_encoder_contrib,
        }

    def _handle_gradient_clipping(self) -> dict[str, float]:
        """
        Handle gradient clipping and compute gradient norms efficiently.

        Returns:
            Dictionary with 'realized' and 'unclipped' gradient norms
        """
        params = [p for p in self.model.parameters() if p.grad is not None]
        if not params:
            return {"realized": 0.0, "unclipped": 0.0}

        # Use float('inf') as the max_norm to ONLY compute the norm if no clipping is desired
        max_norm = self.config.max_grad_norm
        if max_norm is None:
            max_norm = float("inf")

        # clip_grad_norm_ computes and returns the total unclipped norm
        unclipped_grad_norm = clip_grad_norm_(params, max_norm=max_norm).item()

        if self.config.max_grad_norm is not None:
            realized_grad_norm = min(unclipped_grad_norm, self.config.max_grad_norm)
        else:
            realized_grad_norm = unclipped_grad_norm

        return {
            "realized": realized_grad_norm,
            "unclipped": unclipped_grad_norm,
        }

    def _log_training_step_from_metrics(self, step_metrics: dict[str, float]):
        """Log training step metrics to TensorBoard from metrics dictionary."""
        all_mappings = {
            "loss": "Loss/Train",
            "recon": "Loss/Train/BCE",
            "kl": "Loss/Train/KLD",
            "grad_norm_realized": "GradNorm/Train/Realized",
            "grad_norm_unclipped": "GradNorm/Train/Unclipped",
            "recon_grad_norm": "GradNorm/Train/Recon",
            "kl_grad_norm": "GradNorm/Train/Kl",
            "recon_kl_cosine": "GradAlignment/Train/ReconKlCosine",
            "recon_contrib": "GradContribution/Train/Recon",
            "kl_contrib": "GradContribution/Train/Kl",
            "learning_rate": "LearningRate/Train",
        }

        for metric_key, tb_name in all_mappings.items():
            if metric_key in step_metrics:
                self.writer.add_scalar(
                    tb_name, step_metrics[metric_key], self.global_step
                )

    def _log_validation_step(self, val_metrics: dict[str, float]):
        """Log validation step metrics to TensorBoard."""
        val_mappings = {
            "loss": "Loss/Test",
            "recon": "Loss/Test/BCE",
            "kl": "Loss/Test/KLD",
        }

        for metric_key, tb_name in val_mappings.items():
            if metric_key in val_metrics:
                self.writer.add_scalar(
                    tb_name, val_metrics[metric_key], self.global_step
                )

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
            "train_history": self.history.get_train_history(),
            "val_history": self.history.get_val_history(),
            "train_step_history": self.history.get_train_step_history(),
            "config": self.config,
            "is_best": is_best,
        }

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

        train_history = checkpoint.get("train_history", {})
        val_history = checkpoint.get("val_history", {})
        train_step_history = checkpoint.get("train_step_history", {})
        self.history.train_history = train_history
        self.history.val_history = val_history
        self.history.train_step_history = train_step_history

        return checkpoint

    def save_performance_metrics(self, filepath: str):
        """
        Save final performance metrics to JSON file.

        Args:
            filepath: Path to save the performance metrics JSON
        """
        import json
        from datetime import datetime

        param_count = self._count_parameters()
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "parameters": param_count,
                "input_shape": self.model.config.input_shape,
                "hidden_dim": self.model.config.hidden_dim,
                "latent_dim": self.model.config.latent_dim,
                "architecture": "VAE",
            },
            "training_info": {
                "epochs_completed": self.config.num_epochs,
                "total_steps": self.global_step,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
            "final_metrics": {
                "epoch": self.config.num_epochs,
                "total_loss": self.final_val_metrics["loss"]
                if hasattr(self, "final_val_metrics") and self.final_val_metrics
                else None,
                "reconstruction_loss": self.final_val_metrics["recon"]
                if hasattr(self, "final_val_metrics") and self.final_val_metrics
                else None,
                "kl_loss": self.final_val_metrics["kl"]
                if hasattr(self, "final_val_metrics") and self.final_val_metrics
                else None,
            },
            "best_metrics": {
                "total_loss": self.best_val_metrics["loss"]
                if hasattr(self, "best_val_metrics") and self.best_val_metrics
                else None,
                "reconstruction_loss": self.best_val_metrics["recon"]
                if hasattr(self, "best_val_metrics") and self.best_val_metrics
                else None,
                "kl_loss": self.best_val_metrics["kl"]
                if hasattr(self, "best_val_metrics") and self.best_val_metrics
                else None,
            },
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(performance_data, f, indent=2, default=str)

    def close(self):
        """Clean up resources."""
        if self.writer is not None:
            self.writer.close()

    def get_train_history(self) -> dict[str, Any]:
        """Get training history."""
        return self.history.get_train_history()

    def get_val_history(self) -> dict[str, Any]:
        """Get validation history."""
        return self.history.get_val_history()

    def get_train_step_history(self) -> dict[str, Any]:
        """Get training step-level history."""
        return self.history.get_train_step_history()
