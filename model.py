from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    """Output dataclass for VAE forward pass."""

    x_logits: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    std: torch.Tensor

    x_recon: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    loss_recon: torch.Tensor | None = None
    loss_kl: torch.Tensor | None = None


@dataclass
class VAEConfig:
    """Configuration for VAE model architecture."""

    input_dim: int
    hidden_dim: int
    latent_dim: int

    input_shape: tuple[int, int, int] = (1, 28, 28)  # Default input shape for MNIST
    activation: str = "tanh"
    use_torch_distributions: bool = False
    n_samples: int = 1


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()

        self.config = config

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.input_dim, config.hidden_dim),
            getattr(nn, config.activation.capitalize())(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            getattr(nn, config.activation.capitalize())(),
            nn.Linear(config.hidden_dim, config.input_dim),
            nn.Unflatten(1, config.input_shape),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var

    def decode(self, z: torch.Tensor):
        """Decode latent representation to logits (apply sigmoid externally if needed)."""
        return self.decoder(z)

    def _reparameterize(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick

        Args:
            mu: Mean tensor
            std: Standard deviation tensor

        Returns:
            z: Sampled latent vector
        """
        if self.config.use_torch_distributions:
            z = torch.distributions.Normal(
                mu, std
            ).rsample()  # rsample() enables backpropagation
        else:
            z = mu + std * torch.randn_like(std)

        return z

    def forward(
        self,
        x: torch.Tensor,
        compute_loss: bool = True,
        reconstruct: bool = False,
    ) -> VAEOutput:
        """
        Forward pass through VAE.

        Args:
            x: input tensor
            compute_loss: whether to compute BCE+KL
            reconstruct: whether to return reconstructed x
        """
        x_ = x.unsqueeze(1).repeat(
            1, self.config.n_samples, *([1] * len(self.config.input_shape))
        )
        x_ = x_.view(-1, *self.config.input_shape)

        # Only encode once, no matter n_samples
        mu, log_var = self.encode(x_)
        std = torch.exp(0.5 * log_var)

        # Expand mu and std for n_samples
        mu_ = (
            mu.unsqueeze(1)
            .repeat(1, self.config.n_samples, 1)
            .view(-1, self.config.latent_dim)
        )
        std_ = (
            std.unsqueeze(1)
            .repeat(1, self.config.n_samples, 1)
            .view(-1, self.config.latent_dim)
        )

        z = self._reparameterize(mu_, std_)

        x_logits = self.decode(z)

        output = VAEOutput(
            x_logits=x_logits,
            z=z,
            mu=mu,
            std=std,
            x_recon=torch.sigmoid(x_logits) if reconstruct else None,
        )

        if compute_loss:
            loss, loss_recon, loss_kl = self._compute_loss(x_, x_logits, mu, log_var)
            output.loss = loss
            output.loss_recon = loss_recon
            output.loss_kl = loss_kl

        return output

    def _compute_loss(
        self,
        x: torch.Tensor,
        x_logits: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss components: total loss, reconstruction loss, KL divergence loss."""
        loss_recon = F.binary_cross_entropy_with_logits(
            x_logits, x, reduction="sum"
        ) / x.size(0)

        if not self.config.use_torch_distributions:
            loss_kl = (
                0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var, dim=1).mean()
            )
        else:
            std = torch.exp(0.5 * log_var)
            prior = torch.distributions.Normal(
                torch.zeros_like(mu), torch.ones_like(std)
            )
            posterior = torch.distributions.Normal(mu, std)
            loss_kl = (
                torch.distributions.kl.kl_divergence(posterior, prior).sum(dim=1).mean()
            )

        loss = loss_recon + loss_kl

        return loss, loss_recon, loss_kl
