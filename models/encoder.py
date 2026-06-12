import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Convolutional encoder for FashionMNIST (1x28x28 input).

    Architecture:
        Conv2d(1, 32, 4, stride=2, padding=1)  -> (32, 14, 14)
        Conv2d(32, 64, 4, stride=2, padding=1) -> (64, 7, 7)
        Conv2d(64, 128, 3, stride=1, padding=1) -> (128, 7, 7)
        Flatten                                 -> 128*7*7 = 6272
        Linear(6272, 512)                       -> 512
        Linear(512 -> mu), Linear(512 -> logvar)  both shape (latent_dim,)
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_stack = nn.Sequential(
            # 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64x7x7 -> 128x7x7
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # 128 * 7 * 7 = 6272
        self.flat_dim = 128 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor):
        # x: (batch, 1, 28, 28)
        h = self.conv_stack(x)        # (batch, 128, 7, 7)
        h = h.view(h.size(0), -1)     # (batch, 6272)
        h = self.fc(h)                # (batch, 512)
        mu = self.fc_mu(h)            # (batch, latent_dim)
        logvar = self.fc_logvar(h)    # (batch, latent_dim)
        return mu, logvar
