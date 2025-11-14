"""
Configuration classes for VAE training.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrainerConfig:
    """Configuration for VAE trainer behavior and setup."""

    # Training hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 100
    max_grad_norm: float | None = None

    # Logging and checkpointing
    run_dir: str | None = None
    log_interval: int = 5
    checkpoint_interval: int = 5

    # Performance and analysis options
    analyze_gradients: bool = False

    # Evaluation and visualization
    max_latent_batches: int = 400
    n_samples: int = 64
    n_recon: int = 16
    interp_steps: int = 15
    interp_method: str = "slerp"  # "slerp" or "lerp"
    n_latent_samples: int = 1

    # Device and reproducibility
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    seed: int | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.interp_method not in ["slerp", "lerp"]:
            raise ValueError(
                f"interp_method must be 'slerp' or 'lerp', got {self.interp_method}"
            )

        if self.device not in ["auto", "cpu", "cuda", "mps"]:
            raise ValueError(
                f"device must be one of ['auto', 'cpu', 'cuda', 'mps'], got {self.device}"
            )

    def get_run_dir(self) -> str:
        """Get the run directory, generating one if not specified."""
        if self.run_dir is None:
            return f"runs/mnist/vae_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return self.run_dir
