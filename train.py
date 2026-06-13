"""
train.py — Training script for the VAE on FashionMNIST with KL annealing.

KL annealing schedule: beta = min(1.0, epoch / annealing_epochs), where epoch
is 1-indexed. This means beta starts at 1/annealing_epochs (a small but nonzero
value) at epoch 1, reaches 1.0 once epoch == annealing_epochs, and stays at 1.0
for all later epochs. This lets reconstruction learning stabilize before the KL
term ramps up to full weight, which helps avoid posterior collapse.

Usage:
    python train.py [--epochs 30] [--batch-size 64] [--latent-dim 16]
                    [--lr 1e-3] [--annealing-epochs 20]
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from models.vae import VAE
from utils.data import get_dataloaders
from utils.losses import vae_loss

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on FashionMNIST")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--latent-dim", type=int, default=16,
                        help="Latent space dimensionality (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--annealing-epochs", type=int, default=20,
                        help="Number of epochs to ramp beta from 0 to 1 (default: 20)")
    return parser.parse_args()


def train_epoch(model, loader, optimizer, beta, device):
    """Run one training epoch. Returns (mean_recon_loss, mean_kld) over all batches."""
    model.train()
    total_recon = 0.0
    total_kld = 0.0
    num_batches = 0

    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()

        x_recon, mu, logvar = model(x)
        loss, recon_loss, kld = vae_loss(x_recon, x, mu, logvar, beta=beta)

        loss.backward()
        optimizer.step()

        total_recon += recon_loss.item()
        total_kld += kld.item()
        num_batches += 1

    return total_recon / num_batches, total_kld / num_batches


def save_training_curves(log_path: str, out_path: str, latent_dim: int, epochs: int):
    """Read training_log.csv and save a two-subplot figure of loss curves."""
    df = pd.read_csv(log_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df["epoch"], df["recon_loss"], color="#2196F3", linewidth=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reconstruction Loss (BCE / batch)")
    ax1.set_title("Reconstruction Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["epoch"], df["kld"], color="#FF5722", linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("KL Divergence / batch")
    ax2.set_title("KL Divergence")
    ax2.grid(True, alpha=0.3)

    # Shade the annealing region on both plots
    annealing_end = df.loc[df["beta"] < 1.0, "epoch"].max()
    if not np.isnan(annealing_end):
        for ax in (ax1, ax2):
            ax.axvspan(df["epoch"].min(), annealing_end,
                       alpha=0.08, color="gray", label="annealing")

    fig.suptitle(
        f"VAE Training Curves — FashionMNIST (latent_dim={latent_dim}, {epochs} epochs)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved training curves to {out_path}")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    print(f"latent_dim={args.latent_dim}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}, lr={args.lr}, "
          f"annealing_epochs={args.annealing_epochs}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_loader, _ = get_dataloaders(batch_size=args.batch_size)

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_path = "results/training_log.csv"
    # Write CSV header
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "beta", "recon_loss", "kld"])

    for epoch in range(1, args.epochs + 1):
        # beta starts at 1/annealing_epochs at epoch 1, reaches 1.0 at annealing_epochs
        beta = min(1.0, epoch / args.annealing_epochs)

        recon_loss, kld = train_epoch(model, train_loader, optimizer, beta, device)

        print(
            f"epoch {epoch:3d}/{args.epochs}  beta={beta:.3f}  "
            f"recon={recon_loss:.2f}  kld={kld:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(beta, 6), round(recon_loss, 6), round(kld, 6)])

    # Save checkpoint
    checkpoint_path = "models/vae.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}")

    # Generate training curves from the log
    save_training_curves(log_path, "results/training_curves.png", args.latent_dim, args.epochs)


if __name__ == "__main__":
    main()
