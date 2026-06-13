"""
generate_kl_plot.py — Compute and plot mean KL divergence per latent dimension
across the full FashionMNIST test set.

Usage:
    python scripts/generate_kl_plot.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.vae import VAE
from utils.data import get_dataloaders
from utils.losses import kl_per_dimension

LATENT_DIM = int(os.environ.get("LATENT_DIM", "16"))


def main():
    checkpoint = "models/vae.pt"

    device = torch.device("cpu")

    if not os.path.exists(checkpoint):
        print(f"error: checkpoint not found at {checkpoint}")
        print("run train.py first")
        sys.exit(1)

    model = VAE(latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    _, test_loader = get_dataloaders()

    # Accumulate KL-per-dimension across all test batches
    kl_accum = torch.zeros(LATENT_DIM)
    num_batches = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            _, mu, logvar = model(x)
            kl_accum += kl_per_dimension(mu, logvar)
            num_batches += 1

    kl_mean = kl_accum / num_batches  # shape: (latent_dim,)
    kl_np = kl_mean.numpy()

    print("mean KL per dimension:")
    for i, v in enumerate(kl_np):
        print(f"  z[{i:2d}]: {v:.4f}")

    os.makedirs("results", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(LATENT_DIM), kl_np, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Latent dimension index", fontsize=12)
    ax.set_ylabel("Mean KL divergence", fontsize=12)
    ax.set_title("KL Divergence per Latent Dimension — FashionMNIST Test Set", fontsize=13)
    ax.set_xticks(range(LATENT_DIM))
    ax.set_xticklabels([f"z[{i}]" for i in range(LATENT_DIM)], rotation=45, ha="right")
    ax.axhline(y=0.1, color="#FF5722", linestyle="--", linewidth=1, alpha=0.7,
               label="0.1 threshold (dead dimension heuristic)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = "results/kl_per_dimension.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
