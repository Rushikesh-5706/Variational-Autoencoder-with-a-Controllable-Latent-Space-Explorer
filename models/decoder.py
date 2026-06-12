import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Convolutional decoder — roughly symmetric to the encoder.

    Input: z of shape (batch, latent_dim)

    Architecture:
        Linear(latent_dim, 512)          -> 512
        Linear(512, 128*7*7)             -> 6272
        Reshape                          -> (batch, 128, 7, 7)
        ConvTranspose2d(128, 64, 4, stride=2, padding=1)  -> (64, 14, 14)
        ConvTranspose2d(64, 32, 4, stride=2, padding=1)   -> (32, 28, 28)
        ConvTranspose2d(32, 1, 3, stride=1, padding=1)    -> (1, 28, 28)
        Sigmoid                                            -> output in [0, 1]
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # 128 * 7 * 7 = 6272  — must match encoder's bottleneck shape exactly
        self.flat_dim = 128 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.flat_dim),
            nn.ReLU(inplace=True),
        )

        self.deconv_stack = nn.Sequential(
            # (128, 7, 7) -> (64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (64, 14, 14) -> (32, 28, 28)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (32, 28, 28) -> (1, 28, 28)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        # z: (batch, latent_dim)
        h = self.fc(z)                        # (batch, 6272)
        h = h.view(h.size(0), 128, 7, 7)     # (batch, 128, 7, 7)
        x_recon = self.deconv_stack(h)        # (batch, 1, 28, 28)
        assert x_recon.shape[1:] == (1, 28, 28), (
            f"Decoder output shape mismatch: expected (N, 1, 28, 28), got {x_recon.shape}"
        )
        return x_recon
