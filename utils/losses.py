import torch
import torch.nn.functional as F


def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    """
    ELBO loss for the VAE.

    Reconstruction loss: binary cross-entropy summed over pixels and
    averaged over the batch. This pairs correctly with the Sigmoid
    output activation.

    KL divergence: closed-form KL between the approximate posterior
    q(z|x) = N(mu, exp(logvar)) and the prior p(z) = N(0, I),
    averaged over the batch.

    Total loss: recon_loss + beta * kld
    beta=1 is the standard VAE; beta < 1 during annealing lets the
    reconstruction term dominate early in training.

    Args:
        x_recon: decoder output, shape (batch, 1, 28, 28), in [0, 1]
        x:       original images, shape (batch, 1, 28, 28), in [0, 1]
        mu:      encoder mean, shape (batch, latent_dim)
        logvar:  encoder log-variance, shape (batch, latent_dim)
        beta:    KL weight (default 1.0)

    Returns:
        (total_loss, recon_loss, kld) — all scalars, averaged over batch
    """
    batch_size = x.size(0)

    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / batch_size

    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar)), averaged over batch
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    total_loss = recon_loss + beta * kld

    return total_loss, recon_loss, kld


def kl_per_dimension(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Mean KL divergence contribution per latent dimension, averaged over batch.

    Useful for diagnosing posterior collapse: dimensions with near-zero KL
    are not being used — the posterior is not deviating from the prior.

    Args:
        mu:     encoder mean, shape (batch, latent_dim)
        logvar: encoder log-variance, shape (batch, latent_dim)

    Returns:
        Tensor of shape (latent_dim,) — mean KL per dimension over the batch
    """
    # per-sample, per-dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (batch, latent_dim)
    return per_dim.mean(dim=0)  # (latent_dim,)
