# Main plotting package for VAE visualizations
# Expose commonly used functions at package level for easy importing

from .core import model_inference, figure_context, decode_samples, grid_from_images
from .data import collect_latents, apply_pca_if_needed, compute_kl_per_dimension
from .generation import save_samples_figure, save_recon_figure, save_interpolation_combined_figure
from .latent_space import save_latent_combined_figure, save_latent_marginals
from .diagnostics import (
    save_training_curves,
    save_gradient_diagnostics,
    save_kl_diagnostics_combined,
)

# For backward compatibility, expose everything that was in plot.py
__all__ = [
    # Core utilities
    "model_inference",
    "figure_context", 
    "decode_samples",
    "grid_from_images",
    
    # Data processing
    "collect_latents",
    "apply_pca_if_needed", 
    "compute_kl_per_dimension",
    
    # Generation plots
    "save_samples_figure",
    "save_recon_figure", 
    "save_interpolation_combined_figure",
    
    # Latent space plots
    "save_latent_combined_figure",
    "save_latent_marginals",
    
    # Diagnostic plots
    "save_training_curves",
    "save_gradient_diagnostics", 
    "save_kl_diagnostics_combined",
]