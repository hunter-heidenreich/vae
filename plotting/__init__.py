# Main plotting package for VAE visualizations
# Expose commonly used functions at package level for easy importing

from .core import (decode_samples, figure_context, grid_from_images,
                   model_inference)
from .data import (apply_pca_if_needed, collect_latents, collect_latents_with_logvar,
                   compute_kl_per_dimension)
from .diagnostics import (save_gradient_diagnostics,
                          save_kl_diagnostics_combined, save_training_curves)
from .generation import (save_interpolation_combined_figure, save_recon_figure,
                         save_samples_figure)
from .latent_space import (save_latent_combined_figure, save_latent_marginals,
                          save_logvar_combined_figure, save_logvar_marginals)

# For backward compatibility, expose everything that was in plot.py
__all__ = [
    # Core utilities
    "model_inference",
    "figure_context",
    "decode_samples",
    "grid_from_images",
    # Data processing
    "collect_latents",
    "collect_latents_with_logvar",
    "apply_pca_if_needed",
    "compute_kl_per_dimension",
    # Generation plots
    "save_samples_figure",
    "save_recon_figure",
    "save_interpolation_combined_figure",
    # Latent space plots
    "save_latent_combined_figure",
    "save_latent_marginals",
    "save_logvar_combined_figure",
    "save_logvar_marginals",
    # Diagnostic plots
    "save_training_curves",
    "save_gradient_diagnostics",
    "save_kl_diagnostics_combined",
]
