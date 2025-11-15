"""
Metrics collection and tracking utilities.
"""

from typing import Any, Dict, List


class MetricsAccumulator:
    """
    A utility class for accumulating and averaging metrics during training/validation.

    Handles both scalar metrics (that get averaged) and list metrics (that get appended).
    """

    def __init__(self, enable_gradient_analysis: bool = False):
        """
        Initialize the metrics accumulator.

        Args:
            enable_gradient_analysis: Whether to track gradient analysis metrics
        """
        self.enable_gradient_analysis = enable_gradient_analysis
        self.metrics: Dict[str, List[float]] = {}
        self.count = 0

        # Define base metrics that are always tracked
        self.base_metrics = {
            "loss",
            "recon",
            "kl",
            "grad_norm_realized",
            "grad_norm_unclipped",
            "learning_rate",
        }

        # Define gradient analysis metrics
        self.gradient_metrics = {
            "recon_grad_norm",
            "kl_grad_norm",
            "recon_kl_cosine",
            "recon_contrib",
            "kl_contrib",
        }

        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize all metric lists based on configuration."""
        # Always initialize base metrics
        for metric in self.base_metrics:
            self.metrics[metric] = []

        # Initialize gradient metrics if enabled
        if self.enable_gradient_analysis:
            for metric in self.gradient_metrics:
                self.metrics[metric] = []

    def add_step(self, step_metrics: Dict[str, float]):
        """
        Add metrics from a single training/validation step.

        Args:
            step_metrics: Dictionary of metric name -> value pairs
        """
        for metric_name, value in step_metrics.items():
            if metric_name in self.metrics:
                self.metrics[metric_name].append(value)

        self.count += 1

    def get_averages(self) -> Dict[str, float]:
        """
        Get averaged metrics across all accumulated steps.

        Returns:
            Dictionary of metric name -> average value pairs
        """
        if self.count == 0:
            return {}

        averages = {}
        for metric_name, values in self.metrics.items():
            if values:  # Only compute average if we have values
                averages[metric_name] = sum(values) / len(values)

        return averages

    def get_latest(self) -> Dict[str, float]:
        """
        Get the latest (most recent) value for each metric.

        Returns:
            Dictionary of metric name -> latest value pairs
        """
        latest = {}
        for metric_name, values in self.metrics.items():
            if values:
                latest[metric_name] = values[-1]

        return latest

    def reset(self):
        """Reset all accumulated metrics."""
        for metric_name in self.metrics:
            self.metrics[metric_name].clear()
        self.count = 0

    def get_count(self) -> int:
        """Get the number of steps accumulated."""
        return self.count


class TrainingHistory:
    """
    Manages training and validation history with proper handling of different metric types.
    """

    def __init__(self):
        """Initialize empty training history."""
        self.train_history: Dict[str, List[Any]] = {}
        self.val_history: Dict[str, List[Any]] = {}
        self.train_step_history: Dict[str, List[Any]] = {}

    def record_epoch_metrics(
        self, metrics: Dict[str, Any], epoch: int, is_train: bool = True
    ):
        """
        Record epoch-level metrics in history.

        Args:
            metrics: Dictionary of metrics to record
            epoch: Current epoch number
            is_train: Whether these are training (True) or validation (False) metrics
        """
        history = self.train_history if is_train else self.val_history

        # Always record the epoch
        history.setdefault("epoch", []).append(epoch)

        # Record all provided metrics
        for key, value in metrics.items():
            history.setdefault(key, []).append(value)

    def record_step_metrics(
        self, metrics: Dict[str, Any], step: int, is_train: bool = True
    ):
        """
        Record step-level metrics in history.

        Args:
            metrics: Dictionary of metrics to record
            step: Current step number
            is_train: Whether these are training (True) or validation (False) metrics
        """
        if not is_train:
            # We don't record step-level metrics for validation
            return

        # Always record the step
        self.train_step_history.setdefault("step", []).append(step)

        # Record all provided metrics
        for key, value in metrics.items():
            self.train_step_history.setdefault(key, []).append(value)

    def get_train_history(self) -> Dict[str, List[Any]]:
        """Get a copy of training history."""
        return {k: v.copy() for k, v in self.train_history.items()}

    def get_val_history(self) -> Dict[str, List[Any]]:
        """Get a copy of validation history."""
        return {k: v.copy() for k, v in self.val_history.items()}

    def get_train_step_history(self) -> Dict[str, List[Any]]:
        """Get a copy of training step-level history."""
        return {k: v.copy() for k, v in self.train_step_history.items()}

    def clear(self):
        """Clear all history."""
        self.train_history.clear()
        self.val_history.clear()
        self.train_step_history.clear()
