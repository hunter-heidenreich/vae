# Main plotting package for VAE visualizations
# Expose commonly used functions at package level for easy importing

from .core import (decode_samples, figure_context, grid_from_images,
                   model_inference)
from .data import (apply_pca_if_needed, collect_all_latent_data,
                   collect_latents, collect_latents_with_std,
                   compute_kl_per_dimension, get_colormap_colors)
from .generation import (save_interpolation_and_sweep_figures,
                         save_interpolation_figure, save_latent_sweep_figure,
                         save_recon_figure, save_samples_figure)
from .gradient_analysis import save_gradient_diagnostics
from .kl_analysis import save_kl_diagnostics_separate
from .latent_space import (save_latent_combined_figure, save_latent_marginals,
                           save_logvar_combined_figure, save_logvar_marginals)
from .training_curves import save_training_curves

# For backward compatibility, expose everything that was in plot.py
__all__ = [
    # Core utilities
    "model_inference",
    "figure_context",
    "decode_samples",
    "grid_from_images",
    # Data processing
    "collect_latents",
    "collect_latents_with_std",
    "collect_all_latent_data",
    "apply_pca_if_needed",
    "compute_kl_per_dimension",
    "get_colormap_colors",
    # Generation plots
    "save_samples_figure",
    "save_recon_figure",
    "save_interpolation_figure",
    "save_latent_sweep_figure",
    "save_interpolation_and_sweep_figures",
    # Latent space plots
    "save_latent_combined_figure",
    "save_latent_marginals",
    "save_logvar_combined_figure",
    "save_logvar_marginals",
    # Diagnostic plots
    "save_training_curves",
    "save_gradient_diagnostics",
    "save_kl_diagnostics_separate",
]
