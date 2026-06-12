"""
generate_reconstruction.py — Save original, reconstructed, and error-heatmap images
for a given FashionMNIST test-set index.

Usage:
    python scripts/generate_reconstruction.py --index 10
"""

import argparse
import os
import sys

# Allow imports from the project root regardless of where this script is called from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.vae import VAE
from utils.data import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate original, reconstruction, and error heatmap for a test image"
    )
    parser.add_argument("--index", type=int, required=True,
                        help="Test-set index of the image to reconstruct")
    parser.add_argument("--latent-dim", type=int, default=16,
                        help="Latent dimension (must match saved checkpoint, default: 16)")
    parser.add_argument("--checkpoint", type=str, default="models/vae.pt",
                        help="Path to the trained VAE checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu")

    if not os.path.exists(args.checkpoint):
        print(f"error: checkpoint not found at {args.checkpoint}")
        print("run train.py first to generate the checkpoint")
        sys.exit(1)

    model = VAE(latent_dim=args.latent_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(batch_size=1)
    # Grab the specific test image by iterating to the index
    test_dataset = test_loader.dataset
    x, label = test_dataset[args.index]
    x = x.unsqueeze(0)  # (1, 1, 28, 28)

    with torch.no_grad():
        x_recon, mu, logvar = model(x)

    os.makedirs("results", exist_ok=True)

    # Original image
    orig_np = x.squeeze().numpy()  # (28, 28)
    plt.imsave(f"results/original_{args.index}.png", orig_np, cmap="gray", vmin=0, vmax=1)
    print(f"saved results/original_{args.index}.png")

    # Reconstructed image
    recon_np = x_recon.squeeze().numpy()  # (28, 28)
    plt.imsave(f"results/reconstructed_{args.index}.png", recon_np, cmap="gray", vmin=0, vmax=1)
    print(f"saved results/reconstructed_{args.index}.png")

    # Per-pixel absolute error heatmap
    error_np = np.abs(orig_np - recon_np)  # (28, 28)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(error_np, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Absolute pixel error")
    ax.set_title(f"Reconstruction error — test index {args.index}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"results/heatmap_{args.index}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved results/heatmap_{args.index}.png")

    print(
        f"\ntest index {args.index} | class label: {label} | "
        f"mean abs error: {error_np.mean():.4f} | max abs error: {error_np.max():.4f}"
    )


if __name__ == "__main__":
    main()
