"""
Metrics collection and tracking utilities.
"""

from typing import Any

import numpy as np


class MetricsAccumulator:
    """
    A utility class for accumulating and averaging metrics during training/validation.

    Handles both scalar metrics (that get averaged) and list metrics (that get appended).
    """

    def __init__(
        self,
        enable_gradient_analysis: bool = False,
        enable_latent_tracking: bool = False,
    ):
        self.enable_gradient_analysis = enable_gradient_analysis
        self.enable_latent_tracking = enable_latent_tracking
        self.metrics: dict[str, list[float]] = {}
        self.latent_stats: dict[str, list[list[float]]] = {}  # For per-dimension stats
        self.count = 0

        self.base_metrics = {
            "loss",
            "recon",
            "kl",
            "grad_norm_realized",
            "grad_norm_unclipped",
            "learning_rate",
        }

        self.gradient_metrics = {
            "recon_grad_norm",
            "kl_grad_norm",
            "recon_kl_cosine",
            "recon_contrib",
            "kl_contrib",
            "recon_encoder_grad_norm",
            "recon_decoder_grad_norm",
            "kl_encoder_grad_norm",
            "kl_decoder_grad_norm",
            "total_encoder_grad_norm",
            "total_decoder_grad_norm",
            "recon_kl_encoder_cosine",
            "recon_encoder_contrib",
            "kl_encoder_contrib",
        }

        self.parameter_metrics = {
            "encoder_param_norm",
            "decoder_param_norm",
            "total_param_norm",
            "encoder_param_change_norm",
            "decoder_param_change_norm",
            "total_param_change_norm",
            "encoder_param_change_rel",
            "decoder_param_change_rel",
            "total_param_change_rel",
        }

        self._initialize_metrics()

    def _initialize_metrics(self):
        for metric in self.base_metrics:
            self.metrics[metric] = []

        for metric in self.parameter_metrics:
            self.metrics[metric] = []

        if self.enable_gradient_analysis:
            for metric in self.gradient_metrics:
                self.metrics[metric] = []

        if self.enable_latent_tracking:
            self.latent_stats = {
                "mu_means": [],
                "mu_stds": [],
                "std_means": [],
                "std_stds": [],
            }

    def add_step(self, step_metrics: dict[str, float]):
        for metric_name, value in step_metrics.items():
            if metric_name in self.metrics:
                self.metrics[metric_name].append(value)

        self.count += 1

    def add_latent_stats(
        self,
        mu_batch: list[float],
        mu_std_batch: list[float],
        sigma_batch: list[float],
        sigma_std_batch: list[float],
    ):
        if not self.enable_latent_tracking:
            return

        self.latent_stats["mu_means"].append(mu_batch)
        self.latent_stats["mu_stds"].append(mu_std_batch)
        self.latent_stats["std_means"].append(sigma_batch)
        self.latent_stats["std_stds"].append(sigma_std_batch)

    def get_averages(self) -> dict[str, float]:
        if self.count == 0:
            return {}

        averages = {}
        for metric_name, values in self.metrics.items():
            if values:
                averages[metric_name] = sum(values) / len(values)

        return averages

    def get_latent_statistics(self) -> dict[str, list[float]]:
        if not self.enable_latent_tracking or not self.latent_stats["mu_means"]:
            return {}

        mu_means_array = np.array(self.latent_stats["mu_means"])
        mu_stds_array = np.array(self.latent_stats["mu_stds"])
        sigma_means_array = np.array(self.latent_stats["std_means"])
        sigma_stds_array = np.array(self.latent_stats["std_stds"])
        return {
            "mu_mean_per_dim": np.mean(mu_means_array, axis=0).tolist(),
            "mu_std_per_dim": np.mean(mu_stds_array, axis=0).tolist(),
            "mu_mean_overall": np.mean(mu_means_array),
            "mu_std_overall": np.mean(mu_stds_array),
            "sigma_mean_per_dim": np.mean(sigma_means_array, axis=0).tolist(),
            "sigma_std_per_dim": np.mean(sigma_stds_array, axis=0).tolist(),
            "sigma_mean_overall": np.mean(sigma_means_array),
            "sigma_std_overall": np.mean(sigma_stds_array),
        }

    def get_latest(self) -> dict[str, float]:
        latest = {}
        for metric_name, values in self.metrics.items():
            if values:
                latest[metric_name] = values[-1]

        return latest

    def reset(self):
        for metric_name in self.metrics:
            self.metrics[metric_name].clear()

        if self.enable_latent_tracking:
            for stat_name in self.latent_stats:
                self.latent_stats[stat_name].clear()

        self.count = 0

    def get_count(self) -> int:
        return self.count


class TrainingHistory:
    """
    Manages training and validation history with proper handling of different metric types.
    """

    def __init__(self):
        self.train_history: dict[str, list[Any]] = {}
        self.val_history: dict[str, list[Any]] = {}
        self.train_step_history: dict[str, list[Any]] = {}

    def record_epoch_metrics(
        self, metrics: dict[str, Any], epoch: int, is_train: bool = True
    ):
        history = self.train_history if is_train else self.val_history

        history.setdefault("epoch", []).append(epoch)

        for key, value in metrics.items():
            history.setdefault(key, []).append(value)

    def record_step_metrics(
        self, metrics: dict[str, Any], step: int, is_train: bool = True
    ):
        if not is_train:
            return

        self.train_step_history.setdefault("step", []).append(step)

        for key, value in metrics.items():
            self.train_step_history.setdefault(key, []).append(value)

    def get_train_history(self) -> dict[str, list[Any]]:
        return {k: v.copy() for k, v in self.train_history.items()}

    def get_val_history(self) -> dict[str, list[Any]]:
        return {k: v.copy() for k, v in self.val_history.items()}

    def get_train_step_history(self) -> dict[str, list[Any]]:
        return {k: v.copy() for k, v in self.train_step_history.items()}

    def clear(self):
        self.train_history.clear()
        self.val_history.clear()
        self.train_step_history.clear()
