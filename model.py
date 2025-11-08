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


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Flatten(),  # (B, 1, 28, 28) -> (B, 784)
            nn.Linear(input_dim, hidden_dim),  # (B, 784) -> (B, hidden_dim)
            nn.Tanh(),  # (B, hidden_dim) -> (B, hidden_dim)
            nn.Linear(
                hidden_dim, latent_dim * 2
            ),  # (B, hidden_dim) -> (B, latent_dim*2)
        )

        # Decoder network (outputs logits; apply sigmoid only for visualization)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # (B, latent_dim) -> (B, hidden_dim)
            nn.Tanh(),  # (B, hidden_dim) -> (B, hidden_dim)
            nn.Linear(hidden_dim, input_dim),  # (B, hidden_dim) -> (B, 784)
            nn.Unflatten(1, (1, 28, 28)),  # (B, 784) -> (B, 1, 28, 28)
        )

        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var

    def decode(self, z: torch.Tensor):
        """Decode latent representation to logits (apply sigmoid externally if needed)."""
        return self.decoder(z)

    def _reparameterize(
        self, mu: torch.Tensor, std: torch.Tensor, use_distributions: bool = False
    ) -> torch.Tensor:
        """
        Reparameterization trick

        Args:
            mu: Mean tensor
            std: Standard deviation tensor
            use_distributions: Whether to use torch.distributions for sampling

        Returns:
            z: Sampled latent vector
        """
        if use_distributions:
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
        use_analytic_kl: bool = True,
        use_distributions: bool = False,
        n_samples: int = 1,
    ) -> VAEOutput:
        """
        Forward pass through VAE.

        Args:
            x: input tensor shaped (B, 1, 28, 28) or flattened (B, 784)
            compute_loss: whether to compute BCE+KL
            reconstruct: whether to return reconstructed x
            use_analytic_kl: whether to use manual analytic formula (True) or torch.distributions.kl.kl_divergence (False)
            use_distributions: whether to use torch.distributions for sampling
            n_samples: number of samples to draw from latent distribution
        """
        x_ = x.unsqueeze(1).repeat(
            1, n_samples, 1, 1, 1
        )  # (B, C, H, W) -> (B, n_samples, C, H, W)
        x_ = x_.view(-1, *x.shape[1:])  # (B*n_samples, C, H, W)

        # Only encode once, no matter n_samples
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)

        # Expand mu and std for n_samples
        mu_ = (
            mu.unsqueeze(1).repeat(1, n_samples, 1).view(-1, self.latent_dim)
        )  # (B*n_samples, latent_dim)
        std_ = (
            std.unsqueeze(1).repeat(1, n_samples, 1).view(-1, self.latent_dim)
        )  # (B*n_samples, latent_dim)

        z = self._reparameterize(mu_, std_, use_distributions=use_distributions)

        x_logits = self.decode(z)

        output = VAEOutput(
            x_logits=x_logits,
            z=z,
            mu=mu,
            std=std,
            x_recon=torch.sigmoid(x_logits) if reconstruct else None,
        )

        if compute_loss:
            loss, loss_recon, loss_kl = self._compute_loss(
                x_, x_logits, mu, log_var, use_analytic_kl=use_analytic_kl
            )
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
        use_analytic_kl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss components: total loss, reconstruction loss, KL divergence loss."""
        loss_recon = F.binary_cross_entropy_with_logits(
            x_logits, x, reduction="sum"
        ) / x.size(0)

        if use_analytic_kl:
            # KL divergence loss using manual analytic formula
            loss_kl = (
                0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var, dim=1).mean()
            )
        else:
            # KL divergence loss using torch.distributions.kl.kl_divergence (also analytic)
            std = torch.exp(0.5 * log_var)
            prior = torch.distributions.Normal(
                torch.zeros_like(mu), torch.ones_like(std)
            )
            posterior = torch.distributions.Normal(mu, std)
            loss_kl = (
                torch.distributions.kl.kl_divergence(posterior, prior).sum(dim=1).mean()
            )

        # Total loss
        loss = loss_recon + loss_kl

        return loss, loss_recon, loss_kl
