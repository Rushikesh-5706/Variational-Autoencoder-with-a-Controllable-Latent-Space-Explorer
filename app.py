"""
app.py — Streamlit interactive latent space explorer for a trained VAE on FashionMNIST.

Sections:
  1. Model loading (with error handling if checkpoint missing)
  2. Latent space map (PCA or t-SNE of test-set encodings, Plotly scatter)
  3. Latent dimension sliders (sidebar) — live decoded image
  4. Reconstruction viewer — original vs reconstructed vs error heatmap
  5. KL-per-dimension diagnostic bar chart

Environment variables:
  LATENT_DIM            (default 16)
  STREAMLIT_SERVER_PORT (read by Streamlit directly)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.decomposition import PCA

from models.vae import VAE
from utils.data import get_dataloaders
from utils.losses import kl_per_dimension

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FASHIONMNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

LATENT_DIM = int(os.environ.get("LATENT_DIM", "16"))
CHECKPOINT_PATH = "models/vae.pt"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VAE Latent Space Explorer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("VAE Latent Space Explorer")
st.caption(
    f"A trained Variational Autoencoder ({LATENT_DIM}-dimensional latent space) on FashionMNIST. "
    "Use the sidebar sliders to navigate the latent space and decode arbitrary points."
)

# ---------------------------------------------------------------------------
# Section 1: Model loading
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model():
    """Load and cache the trained VAE. Returns the model on CPU in eval mode."""
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    model = VAE(latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()

if model is None:
    st.error(
        f"Checkpoint not found at '{CHECKPOINT_PATH}'. "
        "Run `python train.py` first to generate the trained model weights."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Section 2: Encode the full test set (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def encode_test_set():
    """
    Run the full FashionMNIST test set through the encoder.
    Returns mu vectors (10000, latent_dim) and integer class labels (10000,).
    Cached so slider interactions don't re-encode.
    """
    _model = load_model()
    _, test_loader = get_dataloaders(batch_size=256)
    all_mu = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            mu, _ = _model.encoder(x)
            all_mu.append(mu.numpy())
            all_labels.append(y.numpy())
    mu_arr = np.concatenate(all_mu, axis=0)          # (10000, latent_dim)
    label_arr = np.concatenate(all_labels, axis=0)   # (10000,)
    return mu_arr, label_arr


@st.cache_data
def compute_pca_projection(mu_arr: np.ndarray):
    """Project (N, latent_dim) to (N, 2) using PCA."""
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(mu_arr)


@st.cache_data
def compute_tsne_projection(mu_arr: np.ndarray):
    """Project (N, latent_dim) to (N, 2) using t-SNE. Slower — use sparingly."""
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    return tsne.fit_transform(mu_arr)


@st.cache_data
def compute_kl_per_dim():
    """
    Compute mean KL divergence per latent dimension over the full test set.
    Returns a numpy array of shape (latent_dim,).
    """
    _model = load_model()
    _, test_loader = get_dataloaders(batch_size=256)
    kl_accum = torch.zeros(LATENT_DIM)
    n = 0
    with torch.no_grad():
        for x, _ in test_loader:
            _, mu, logvar = _model(x)
            kl_accum += kl_per_dimension(mu, logvar)
            n += 1
    return (kl_accum / n).numpy()

# ---------------------------------------------------------------------------
# Section 2: Latent space map
# ---------------------------------------------------------------------------

st.header("Latent Space Map")
st.caption(
    "Scatter plot of the test-set encoder outputs projected to 2D. "
    "Each point is one image; color indicates the FashionMNIST class."
)

proj_method = st.selectbox(
    "Projection method",
    options=["PCA (fast)", "t-SNE (slow, ~30s)"],
    index=0,
)

mu_arr, label_arr = encode_test_set()

if proj_method.startswith("PCA"):
    coords_2d = compute_pca_projection(mu_arr)
else:
    with st.spinner("Running t-SNE on 10,000 points — this takes ~30 seconds..."):
        coords_2d = compute_tsne_projection(mu_arr)

class_names = [FASHIONMNIST_CLASSES[l] for l in label_arr]

scatter_fig = px.scatter(
    x=coords_2d[:, 0],
    y=coords_2d[:, 1],
    color=class_names,
    labels={"x": "Component 1", "y": "Component 2", "color": "Class"},
    title=f"Test-set latent codes — {proj_method.split()[0]} projection",
    opacity=0.5,
    height=550,
)
scatter_fig.update_traces(marker_size=3)
st.plotly_chart(scatter_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Sidebar — latent dimension sliders
# ---------------------------------------------------------------------------

st.sidebar.header("Latent Dimension Sliders")
st.sidebar.caption(
    "Set each dimension of z, then the decoder renders the corresponding image."
)

z_values = []
for i in range(LATENT_DIM):
    val = st.sidebar.slider(
        label=f"z[{i}]",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
        key=f"z_{i}",
    )
    z_values.append(val)

z_tensor = torch.tensor(z_values, dtype=torch.float32).unsqueeze(0)  # (1, latent_dim)

with torch.no_grad():
    decoded = model.decoder(z_tensor)  # (1, 1, 28, 28)

decoded_img = decoded.squeeze().numpy()  # (28, 28)

st.header("Decoded Latent Point")
st.caption("Output of the decoder for the current slider values.")

# Upscale to 140x140 (nearest-neighbor so pixels stay crisp)
decoded_large = np.kron(decoded_img, np.ones((5, 5)))  # 28*5=140

col_dec, col_space = st.columns([1, 3])
with col_dec:
    st.image(decoded_large, caption="Decoded image (140x140)", clamp=True)

# ---------------------------------------------------------------------------
# Section 4: Reconstruction viewer
# ---------------------------------------------------------------------------

st.header("Reconstruction Viewer")
st.caption("Pick a test-set index to compare the original, reconstruction, and per-pixel error.")

_, test_loader_single = get_dataloaders(batch_size=1)
test_dataset = test_loader_single.dataset

test_idx = st.number_input(
    "Test-set index (0 to 9999)",
    min_value=0,
    max_value=len(test_dataset) - 1,
    value=10,
    step=1,
)
test_idx = int(test_idx)

x_sample, y_sample = test_dataset[test_idx]
x_input = x_sample.unsqueeze(0)  # (1, 1, 28, 28)

with torch.no_grad():
    x_recon, mu_sample, logvar_sample = model(x_input)

orig_np = x_sample.squeeze().numpy()
recon_np = x_recon.squeeze().numpy()
error_np = np.abs(orig_np - recon_np)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original")
    st.image(np.kron(orig_np, np.ones((5, 5))), caption=FASHIONMNIST_CLASSES[y_sample], clamp=True)

with col2:
    st.subheader("Reconstruction")
    st.image(np.kron(recon_np, np.ones((5, 5))), caption="VAE output", clamp=True)

with col3:
    st.subheader("Absolute Error")
    fig_err, ax_err = plt.subplots(figsize=(3, 3))
    im = ax_err.imshow(error_np, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_err)
    ax_err.axis("off")
    plt.tight_layout()
    st.pyplot(fig_err)
    plt.close(fig_err)

st.caption(
    f"Mean absolute error: {error_np.mean():.4f} | "
    f"Max absolute error: {error_np.max():.4f}"
)

# ---------------------------------------------------------------------------
# Section 5: KL-per-dimension diagnostic
# ---------------------------------------------------------------------------

st.header("KL Divergence per Latent Dimension")

kl_vals = compute_kl_per_dim()

kl_df = pd.DataFrame({"dimension": [f"z[{i}]" for i in range(LATENT_DIM)], "mean_kl": kl_vals})

st.bar_chart(kl_df.set_index("dimension")["mean_kl"])

dead_count = int((kl_vals < 0.1).sum())
active_count = LATENT_DIM - dead_count

st.caption(
    f"Dimensions with KL < 0.1 are treated as 'dead' — the encoder posterior "
    f"hasn't moved from the prior, so those channels carry no information. "
    f"Across the test set, {dead_count} of {LATENT_DIM} dimensions appear dead "
    f"and {active_count} carry meaningful signal."
)
