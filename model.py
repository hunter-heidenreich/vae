"""Variational Autoencoder (VAE) model implementation."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation: str) -> nn.Module:
    """Get activation function by name."""
    activation_lower = activation.lower()
    ACTIVATION_MAP = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
    }
    if activation_lower not in ACTIVATION_MAP:
        supported = ", ".join(ACTIVATION_MAP.keys())
        raise ValueError(f"Unsupported activation '{activation}'. Supported: {supported}")
    return ACTIVATION_MAP[activation_lower]


@dataclass
class VAEConfig:
    """VAE model configuration specifying architecture and behavior."""
    hidden_dim: int
    latent_dim: int
    
    input_shape: tuple[int, int, int] = (1, 28, 28)  # Default: MNIST
    activation: str = "tanh"  # Default: tanh, what was used in the original VAE paper 
    use_softplus_std: bool = False  # Whether to use softplus for std parameterization
    n_samples: int = 1  # Number of latent samples per input during training


@dataclass
class VAEOutput:
    """VAE forward pass output containing all relevant tensors and optional losses."""
    x_logits: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    std: torch.Tensor
    
    x_recon: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    loss_recon: torch.Tensor | None = None
    loss_kl: torch.Tensor | None = None


class VAE(nn.Module):
    """Variational Autoencoder with support for deterministic and probabilistic reconstruction."""
    
    DEFAULT_EPS = 1e-8
    
    def __init__(self, config: VAEConfig) -> None:
        """Initialize VAE with given configuration.
        
        Args:
            config: VAE configuration specifying architecture and behavior
        """
        super().__init__()
        self.config = config
        
        # Build encoder: input -> hidden -> latent parameters (mu, sigma)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(config.input_shape))), config.hidden_dim),
            get_activation(config.activation),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),
        )
        
        # Build decoder: latent -> hidden -> reconstructed input        
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            get_activation(config.activation),
            nn.Linear(config.hidden_dim, int(torch.prod(torch.tensor(config.input_shape)))),
            nn.Unflatten(1, config.input_shape),
        )
        
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        encoder_output = self.encoder(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=-1)
        return mu, sigma
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction logits"""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick for differentiable sampling."""
        epsilon = torch.randn_like(std)
        return mu + std * epsilon
    
    def forward(
        self,
        x: torch.Tensor,
        compute_loss: bool = True,
        reconstruct: bool = False,
        eps: float = DEFAULT_EPS,
    ) -> VAEOutput:
        """Forward pass through the VAE.
        
        Args:
            x: Input tensor of shape (batch_size, *input_shape)
            compute_loss: Whether to compute VAE loss components
            reconstruct: Whether to return reconstructions or distributions
            eps: Small epsilon value for numerical stability

        Returns:
            VAEOutput containing all relevant tensors and optionally computed losses
        """
        # Prepare input for multiple sampling if needed
        x_expanded = self._expand_for_sampling(x) if self.config.n_samples > 1 else x
        
        # Encode and sample from latent space
        mu, sigma = self.encode(x)
        std = self._sigma_to_std(sigma, eps=eps)
        mu_expanded, std_expanded = self._expand_latent_params(mu, std)
        z = self.reparameterize(mu_expanded, std_expanded)
        
        # Decode latent samples
        x_logits = self.decode(z)
        
        # Create output object
        output = VAEOutput(
            x_logits=x_logits,
            z=z,
            mu=mu,
            std=std,
            x_recon=torch.sigmoid(x_logits) if reconstruct else None,
        )
        
        # Compute losses if requested
        if compute_loss:
            loss, loss_recon, loss_kl = self._compute_loss(
                x_expanded, x_logits, mu, sigma, std
            )
            output.loss = loss
            output.loss_recon = loss_recon
            output.loss_kl = loss_kl
        
        return output

    
    # ==================== Helper Methods ====================
    
    def _sigma_to_std(self, sigma: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
        """Convert sigma parameter to standard deviation."""
        if self.config.use_softplus_std:
            return F.softplus(sigma) + eps
        else:
            return torch.exp(0.5 * sigma)  # sigma represents log-variance
    
    def _expand_for_sampling(self, x: torch.Tensor) -> torch.Tensor:
        """Expand input tensor for multiple sampling."""
        shape_dims = [1] * len(self.config.input_shape)
        x_expanded = x.unsqueeze(1).repeat(1, self.config.n_samples, *shape_dims)
        return x_expanded.view(-1, *self.config.input_shape)
    
    def _expand_latent_params(
        self, mu: torch.Tensor, std: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand latent parameters for multiple sampling."""
        if self.config.n_samples == 1:
            return mu, std
        
        mu_expanded = (
            mu.unsqueeze(1)
            .repeat(1, self.config.n_samples, 1)
            .view(-1, self.config.latent_dim)
        )
        std_expanded = (
            std.unsqueeze(1)
            .repeat(1, self.config.n_samples, 1)
            .view(-1, self.config.latent_dim)
        )
        
        return mu_expanded, std_expanded

    # ==================== Loss Computation ====================
    
    def _compute_loss(
        self,
        x: torch.Tensor,
        x_logits: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss components for deterministic reconstruction."""
        loss_recon = self._compute_reconstruction_loss(x, x_logits)
        loss_kl = self._compute_kl_loss(mu, sigma, std)
        return loss_recon + loss_kl, loss_recon, loss_kl
    
    def _compute_reconstruction_loss(
        self, x: torch.Tensor, x_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss using binary cross-entropy."""
        return F.binary_cross_entropy_with_logits(
            x_logits, x, reduction="sum"
        ) / x.size(0)
    
    def _compute_kl_loss(
        self, mu: torch.Tensor, sigma: torch.Tensor, std: torch.Tensor, eps: float = DEFAULT_EPS
    ) -> torch.Tensor:
        """Compute KL divergence between latent distribution and standard normal prior."""
        # Analytical KL: KL(N(μ,σ²) || N(0,1)) = 0.5 * Σ(μ² + σ² - 1 - log(σ²))
        if self.config.use_softplus_std:
            # sigma is just the raw output, need to use std directly: σ
            kl_per_sample = 0.5 * torch.sum(
                mu.pow(2) + std.pow(2) - 1 - torch.log(std.pow(2) + eps), dim=1
            )
        else:
            # sigma represents log-variance parameterization: log(σ²)
            kl_per_sample = 0.5 * torch.sum(
                mu.pow(2) + sigma.exp() - 1 - sigma, dim=1
            )

        return kl_per_sample.mean()
