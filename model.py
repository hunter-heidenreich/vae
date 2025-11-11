"""Variational Autoencoder (VAE) model implementation."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    """VAE forward pass output containing all relevant tensors and optional losses."""
    x_logits: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    std: torch.Tensor
    
    x_recon: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    loss_recon: Optional[torch.Tensor] = None
    loss_kl: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None
    x_recon_distribution: Optional[torch.distributions.Beta] = None
    # Logit Normal parameters
    logit_mu: Optional[torch.Tensor] = None
    logit_sigma: Optional[torch.Tensor] = None
    x_recon_logit_normal: Optional[torch.distributions.TransformedDistribution] = None


@dataclass
class VAEConfig:
    """VAE model configuration specifying architecture and behavior."""
    input_dim: int
    hidden_dim: int
    latent_dim: int
    
    input_shape: Tuple[int, int, int] = (1, 28, 28)  # Default: MNIST
    activation: str = "tanh"
    use_torch_distributions: bool = False
    n_samples: int = 1
    use_softplus_std: bool = False
    probabilistic_mode: str | None = None  # "beta" or "logit_normal"


@dataclass
class MLPConfig:
    """Configuration for a simple Multi-Layer Perceptron (MLP)."""
    input_shape: Tuple[int, ...]
    hidden_dims: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    activation: str = "tanh"
    flatten_input: bool = False
    unflatten_output: bool = False


class MLP(nn.Module):
    
    def __init__(self, config: MLPConfig) -> None:
        """Initialize MLP with given configuration.
        
        Args:
            config: MLP configuration specifying architecture
        """
        super().__init__()
        self.config = config
        
        layers = []
        input_dim = 1
        for dim in config.input_shape:
            input_dim *= dim
        
        if config.flatten_input:
            layers.append(nn.Flatten())
        
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(config.activation))
            prev_dim = hidden_dim
        
        output_dim = 1
        for dim in config.output_shape:
            output_dim *= dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if config.unflatten_output:
            layers.append(nn.Unflatten(1, config.output_shape))
        
        self.model = nn.Sequential(*layers)
    
    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Get activation function by name.
        
        Args:
            activation: Name of activation function
            
        Returns:
            PyTorch activation module
            
        Raises:
            ValueError: If activation function is not supported
        """
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


class GaussianHead(nn.Module):
    """Gaussian head producing mean and log-variance parameters."""

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, 2, dim=-1)
    
    def rsample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_per_sample = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1)
        return kl_per_sample.mean()
    
    def rel_log_prob(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return (x - mu) ** 2 / logvar.exp() + logvar
    
    
class SoftplusGaussianHead(nn.Module):
    """Gaussian head producing mean and softplus standard deviation parameters."""

    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> list[torch.Tensor]:
        mu, softplus_std = torch.chunk(x, 2, dim=-1)
        return [mu, F.softplus(softplus_std) + eps]
    
    def rsample(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(std) * std
    
    def kl_divergence(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        kl_per_sample = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 1 - 2 * torch.log(std), dim=1)
        return kl_per_sample.mean()
    
    def rel_log_prob(self, x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - mu) ** 2 / (std ** 2) + 2 * torch.log(std)
    

class BetaHead(nn.Module):
    """Beta head producing alpha and beta parameters for Beta distribution."""

    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> list[torch.Tensor]:
        alpha_raw, beta_raw = torch.chunk(x, 2, dim=-1)
        alpha = F.softplus(alpha_raw) + eps
        beta = F.softplus(beta_raw) + eps
        return [alpha, beta]
    
    def rsample(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        beta_dist = torch.distributions.Beta(alpha, beta)
        return beta_dist.rsample()
    
    def kl_divergence(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("KL divergence for BetaHead is not implemented.")
    
    def rel_log_prob(self, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        beta_dist = torch.distributions.Beta(alpha, beta)
        return -beta_dist.log_prob(x)
    
class LogitGaussianHead(nn.Module):
    """Logit Normal head producing mu and sigma parameters for Logit Normal distribution."""

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, 2, dim=-1)
    
    def rsample(self, logit_mu: torch.Tensor, logit_logvar: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit_mu + torch.randn_like(logit_logvar) * torch.exp(0.5 * logit_logvar))
    
    def kl_divergence(self, logit_mu: torch.Tensor, logit_logvar: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("KL divergence for LogitNormalHead is not implemented.")
    
    def rel_log_prob(self, x: torch.Tensor, logit_mu: torch.Tensor, logit_logvar: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x_clamped = torch.clamp(x, min=eps, max=1 - eps)
        logit_x = torch.log(x_clamped) - torch.log(1 - x_clamped)
        return (logit_x - logit_mu) ** 2 / logit_logvar.exp() + logit_logvar


class LogitSoftplusGaussianHead(nn.Module):
    """Logit Normal head producing mu and softplus sigma parameters for Logit Normal distribution."""

    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> list[torch.Tensor]:
        logit_mu, softplus_logit_std = torch.chunk(x, 2, dim=-1)
        logit_std = F.softplus(softplus_logit_std) + eps
        return [logit_mu, logit_std]
    
    def rsample(self, logit_mu: torch.Tensor, logit_std: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit_mu + torch.randn_like(logit_std) * logit_std)
    
    def kl_divergence(self, logit_mu: torch.Tensor, logit_std: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("KL divergence for LogitSoftplusGaussianHead is not implemented.")
    
    def rel_log_prob(self, x: torch.Tensor, logit_mu: torch.Tensor, logit_std: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x_clamped = torch.clamp(x, min=eps, max=1 - eps)
        logit_x = torch.log(x_clamped) - torch.log(1 - x_clamped)
        return (logit_x - logit_mu) ** 2 / (logit_std ** 2) + 2 * torch.log(logit_std)


class VAE(nn.Module):
    """Variational Autoencoder with support for deterministic and probabilistic reconstruction."""
    
    DEFAULT_EPS = 1e-12
    
    def __init__(self, config: VAEConfig) -> None:
        """Initialize VAE with given configuration.
        
        Args:
            config: VAE configuration specifying architecture and behavior
        """
        super().__init__()
        self.config = config
        
        # Build encoder: input -> hidden -> latent parameters (mu, sigma)
        self.encoder = MLP(
            MLPConfig(
                input_shape=config.input_shape,
                hidden_dims=(config.hidden_dim,),
                output_shape=(2 * config.latent_dim,),
                activation=config.activation,
                flatten_input=True,
            )
        )
        self.encoder_head = GaussianHead() if not config.use_softplus_std else SoftplusGaussianHead()
        
        # Build decoder: latent -> hidden -> reconstructed input        
        self.decoder = MLP(
            MLPConfig(
                input_shape=(config.latent_dim,),
                hidden_dims=(config.hidden_dim,),
                output_shape=config.input_shape if config.probabilistic_mode is None else (2, *config.input_shape),
                activation=config.activation,
                unflatten_output=config.probabilistic_mode is None,
            )
        )
        self.decoder_head = nn.Identity() if config.probabilistic_mode is None else (
            BetaHead() if config.probabilistic_mode == "beta" else
            LogitSoftplusGaussianHead() if config.probabilistic_mode == "logit_normal" and config.use_softplus_std else
            LogitGaussianHead()
        )
    
    # ==================== Encoding and Decoding ====================
    
    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode input to latent distribution parameters."""
        encoded = self.encoder(x)
        return self.encoder_head(encoded)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction logits or probabilistic distribution parameters.
        
        Args:
            z: Latent samples of shape (batch_size, latent_dim)
            
        Returns:
            For deterministic mode: Reconstruction logits of original input shape.
            For probabilistic mode: Raw parameters for the chosen distribution (Beta or Logit Normal).
        """
        decoder_output = self.decoder(z)
        
        if self.config.probabilistic_reconstruction:
            # Return raw parameters (will be split into alpha/beta in forward pass)
            return decoder_output
        else:
            # Apply unflatten for deterministic mode
            return self.unflatten(decoder_output)

    # ==================== Sampling ====================
    
    def _reparameterize(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick for differentiable sampling.
        
        The reparameterization trick allows gradients to flow through the
        stochastic sampling operation by expressing the sample as a deterministic
        function of the parameters and external noise.
        
        Args:
            mu: Mean parameters of shape (batch_size, latent_dim)
            std: Standard deviation parameters of shape (batch_size, latent_dim)
            
        Returns:
            Sampled latent vectors of shape (batch_size, latent_dim)
        """
        if self.config.use_torch_distributions:
            # Use PyTorch distributions for sampling (more robust)
            distribution = torch.distributions.Normal(mu, std)
            return distribution.rsample()  # rsample() enables backpropagation
        else:
            # Manual reparameterization: z = μ + σ * ε, where ε ~ N(0,1)
            epsilon = torch.randn_like(std)
            return mu + std * epsilon

    # ==================== Forward Pass ====================
    
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
        z = self._reparameterize(mu_expanded, std_expanded)
        
        # Decode latent samples
        decoder_output = self.decode(z)
        
        # Handle reconstruction based on mode
        if self.config.probabilistic_reconstruction:
            return self._create_probabilistic_output(
                x_expanded, decoder_output, z, mu, sigma, std, 
                compute_loss, reconstruct, eps
            )
        else:
            return self._create_deterministic_output(
                x_expanded, decoder_output, z, mu, sigma, std,
                compute_loss, reconstruct
            )
    
    # ==================== Helper Methods ====================
    
    def _sigma_to_std(self, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def _process_beta_parameters(self, decoder_output: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process decoder output into Beta distribution parameters.
        
        Args:
            decoder_output: Raw decoder output of shape (batch_size, input_dim * 2)
            eps: Small epsilon value for numerical stability
            
        Returns:
            Tuple of (alpha, beta) parameters with original input shape
        """
        # Split the output into alpha and beta parameters
        alpha_raw, beta_raw = torch.chunk(decoder_output, self.BETA_PARAM_COUNT, dim=-1)
        
        # Apply softplus to ensure positivity and add epsilon for stability
        alpha = F.softplus(alpha_raw) + eps 
        beta = F.softplus(beta_raw) + eps
        
        # Reshape to original input shape
        alpha = self.unflatten_alpha(alpha)
        beta = self.unflatten_beta(beta)
        
        return alpha, beta

    def _process_logit_normal_parameters(self, decoder_output: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process decoder output into Logit Normal distribution parameters.
        
        Args:
            decoder_output: Raw decoder output of shape (batch_size, input_dim * 2)
            eps: Small epsilon value for numerical stability
            
        Returns:
            Tuple of (mu, sigma) parameters with original input shape for the base Normal distribution
        """
        # Split the output into mu and log_sigma parameters for the base Normal distribution
        logit_mu_raw, logit_log_sigma_raw = torch.chunk(decoder_output, self.LOGIT_NORMAL_PARAM_COUNT, dim=-1)
        
        # mu can be any real value (no constraints)
        logit_mu = logit_mu_raw
        
        # Apply softplus to log_sigma to ensure positivity and add epsilon for stability
        logit_sigma = F.softplus(logit_log_sigma_raw) + eps
        
        # Reshape to original input shape
        logit_mu = self.unflatten_logit_mu(logit_mu)
        logit_sigma = self.unflatten_logit_sigma(logit_sigma)
        
        return logit_mu, logit_sigma

    def _create_probabilistic_output(
        self,
        x_expanded: torch.Tensor,
        decoder_output: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        std: torch.Tensor,
        compute_loss: bool,
        reconstruct: bool,
        eps: float,
    ) -> VAEOutput:
        """Create VAEOutput for probabilistic reconstruction mode."""
        if self.config.probabilistic_mode == "beta":
            return self._create_beta_output(
                x_expanded, decoder_output, z, mu, sigma, std, compute_loss, reconstruct, eps
            )
        elif self.config.probabilistic_mode == "logit_normal":
            return self._create_logit_normal_output(
                x_expanded, decoder_output, z, mu, sigma, std, compute_loss, reconstruct, eps
            )
        else:
            raise ValueError(f"Unsupported probabilistic mode: {self.config.probabilistic_mode}")

    def _create_beta_output(
        self,
        x_expanded: torch.Tensor,
        decoder_output: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        std: torch.Tensor,
        compute_loss: bool,
        reconstruct: bool,
        eps: float,
    ) -> VAEOutput:
        """Create VAEOutput for Beta distribution reconstruction mode."""
        # Process Beta distribution parameters
        alpha, beta = self._process_beta_parameters(decoder_output, eps=eps)
        
        # Create Beta distribution and compute reconstruction if requested
        x_recon_distribution = None
        x_recon = None
        if reconstruct:
            x_recon_distribution = torch.distributions.Beta(alpha, beta)
            x_recon = x_recon_distribution.mean
        
        # Create output object
        output = VAEOutput(
            x_logits=decoder_output,
            z=z,
            mu=mu,
            std=std,
            alpha=alpha,
            beta=beta,
            x_recon_distribution=x_recon_distribution,
            x_recon=x_recon,
        )
        
        # Compute losses if requested
        if compute_loss:
            loss, loss_recon, loss_kl = self._compute_loss_beta(
                x_expanded, alpha, beta, mu, sigma, std, eps=eps
            )
            output.loss = loss
            output.loss_recon = loss_recon
            output.loss_kl = loss_kl
        
        return output

    def _create_logit_normal_output(
        self,
        x_expanded: torch.Tensor,
        decoder_output: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        std: torch.Tensor,
        compute_loss: bool,
        reconstruct: bool,
        eps: float,
    ) -> VAEOutput:
        """Create VAEOutput for Logit Normal distribution reconstruction mode."""
        # Process Logit Normal distribution parameters
        logit_mu, logit_sigma = self._process_logit_normal_parameters(decoder_output, eps=eps)
        
        # Create Logit Normal distribution and compute reconstruction if requested
        x_recon_logit_normal = None
        x_recon = None
        if reconstruct:
            # Create base Normal distribution
            base_dist = torch.distributions.Normal(logit_mu, logit_sigma)
            # Apply sigmoid transform to map from R to (0, 1)
            transform = torch.distributions.transforms.SigmoidTransform()
            x_recon_logit_normal = torch.distributions.TransformedDistribution(base_dist, transform)
            # Since TransformedDistribution.mean is not implemented, we'll use the transformation of the base mean
            # For logit-normal, the mean is sigmoid(base_mean) which is a reasonable approximation
            x_recon = torch.sigmoid(logit_mu)
        
        # Create output object
        output = VAEOutput(
            x_logits=decoder_output,
            z=z,
            mu=mu,
            std=std,
            logit_mu=logit_mu,
            logit_sigma=logit_sigma,
            x_recon_logit_normal=x_recon_logit_normal,
            x_recon=x_recon,
        )
        
        # Compute losses if requested
        if compute_loss:
            loss, loss_recon, loss_kl = self._compute_loss_logit_normal(
                x_expanded, logit_mu, logit_sigma, mu, sigma, std, eps=eps
            )
            output.loss = loss
            output.loss_recon = loss_recon
            output.loss_kl = loss_kl
        
        return output

    def _create_deterministic_output(
        self,
        x_expanded: torch.Tensor,
        x_logits: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        std: torch.Tensor,
        compute_loss: bool,
        reconstruct: bool,
    ) -> VAEOutput:
        """Create VAEOutput for deterministic reconstruction mode."""
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

    # ==================== Loss Computation ====================
    
    def _compute_loss(
        self,
        x: torch.Tensor,
        x_logits: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self, mu: torch.Tensor, sigma: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between latent distribution and standard normal prior."""
        if self.config.use_torch_distributions:
            return self._compute_kl_with_distributions(mu, std)
        else:
            return self._compute_kl_analytical(mu, sigma)
    
    def _compute_kl_with_distributions(
        self, mu: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence using torch.distributions."""
        prior = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std)
        )
        posterior = torch.distributions.Normal(mu, std)
        
        kl_per_sample = torch.distributions.kl.kl_divergence(posterior, prior)
        return kl_per_sample.sum(dim=1).mean()
    
    def _compute_kl_analytical(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence analytically for log-variance parameterization."""
        if self.config.use_softplus_std:
            raise ValueError(
                "Analytical KL computation requires log-variance parameterization. "
                "Set use_softplus_std=False or use_torch_distributions=True."
            )
        
        # Analytical KL: KL(N(μ,σ²) || N(0,1)) = 0.5 * Σ(μ² + σ² - 1 - log(σ²))
        kl_per_sample = 0.5 * torch.sum(
            mu.pow(2) + sigma.exp() - 1 - sigma, dim=1
        )
        return kl_per_sample.mean()
    
    def _compute_loss_beta(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        std: torch.Tensor,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss components for Beta distribution reconstruction."""
        loss_recon = self._compute_beta_reconstruction_loss(x, alpha, beta, eps=eps)
        loss_kl = self._compute_kl_loss(mu, sigma, std)
        return loss_recon + loss_kl, loss_recon, loss_kl

    def _compute_loss_logit_normal(
        self,
        x: torch.Tensor,
        logit_mu: torch.Tensor,
        logit_sigma: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        std: torch.Tensor,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss components for Logit Normal distribution reconstruction."""
        loss_recon = self._compute_logit_normal_reconstruction_loss(x, logit_mu, logit_sigma, eps=eps)
        loss_kl = self._compute_kl_loss(mu, sigma, std)
        return loss_recon + loss_kl, loss_recon, loss_kl
    
    def _compute_beta_reconstruction_loss(
        self, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Compute Beta distribution negative log-likelihood reconstruction loss."""
        beta_dist = torch.distributions.Beta(alpha, beta)
        x_clamped = torch.clamp(x, min=eps, max=1 - eps)
        log_prob = beta_dist.log_prob(x_clamped)
        return -log_prob.sum(dim=(1, 2, 3)).mean()

    def _compute_logit_normal_reconstruction_loss(
        self, x: torch.Tensor, logit_mu: torch.Tensor, logit_sigma: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Compute Logit Normal distribution negative log-likelihood reconstruction loss."""
        # Create base Normal distribution
        base_dist = torch.distributions.Normal(logit_mu, logit_sigma)
        # Apply sigmoid transform to map from R to (0, 1)
        transform = torch.distributions.transforms.SigmoidTransform()
        logit_normal_dist = torch.distributions.TransformedDistribution(base_dist, transform)
        
        # Clamp input to avoid numerical issues at boundaries
        x_clamped = torch.clamp(x, min=eps, max=1 - eps)
        
        # Compute log probability (includes Jacobian automatically)
        log_prob = logit_normal_dist.log_prob(x_clamped)
        return -log_prob.sum(dim=(1, 2, 3)).mean()
