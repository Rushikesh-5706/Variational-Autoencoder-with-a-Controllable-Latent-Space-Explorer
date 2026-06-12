import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class VAE(nn.Module):
    """
    Variational Autoencoder combining encoder and decoder with the
    reparameterization trick.
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * std, eps ~ N(0, I).
        Differentiable w.r.t. mu and logvar because eps is sampled independently.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input images, shape (batch, 1, 28, 28), values in [0, 1]

        Returns:
            x_recon: reconstructed images, shape (batch, 1, 28, 28)
            mu:      latent mean, shape (batch, latent_dim)
            logvar:  latent log-variance, shape (batch, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Draw num_samples latent vectors from the prior N(0, I) and decode them.

        Args:
            num_samples: number of images to generate
            device:      target device

        Returns:
            Generated images, shape (num_samples, 1, 28, 28)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)
