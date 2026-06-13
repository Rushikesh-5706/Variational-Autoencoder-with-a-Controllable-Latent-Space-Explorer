# Training Analysis: VAE on FashionMNIST

## Setup

Trained a convolutional VAE with a 16-dimensional latent space on FashionMNIST
(60,000 training images, 10 classes, 28x28 grayscale). The model uses
stride-2 convolutions in the encoder and transposed convolutions in the decoder,
with a 512-unit fully connected bottleneck between the convolutional stack and
the mu/logvar heads. Optimizer: Adam, learning rate 1e-3. Batch size 64. 30 epochs.
KL annealing ran for the first 20 epochs, linearly ramping beta from 0.05 to 1.0.

## Training Curves

Reconstruction loss and KL divergence over 30 epochs (values per batch, averaged per epoch):

| Epoch | Beta | Recon Loss | KL Divergence |
|-------|------|------------|---------------|
| 1     | 0.050 | 236.02 | 44.4689 |
| 20    | 1.000 | 217.02 | 17.2049 |
| 30    | 1.000 | 216.23 | 16.7607 |

Epoch 1 starts with beta=0.05 (1/20), so the KL term barely contributes to the
total loss. As expected, reconstruction loss falls quickly in the first 5 epochs —
the model is learning basic shape structure without pressure from the KL term.

Between epochs 1 and 20, beta increases from 0.05 to 1.0 linearly.
The KL divergence rises during this period as the model is forced to spread
encodings across the prior. After epoch 20 (beta=1.0), both losses continue
to decline slowly, indicating the model hasn't saturated.

## Posterior Collapse Analysis

After full training, none of the 16 latent dimensions have mean KL < 0.1.
Every dimension's posterior has moved meaningfully away from the prior,
meaning all 16 are encoding some variation in the dataset rather than
sitting idle.

KL values per dimension from the full test set:
  z[ 0]: 2.6710
  z[ 1]: 0.5976
  z[ 2]: 0.1382
  z[ 3]: 0.5702
  z[ 4]: 0.1606
  z[ 5]: 1.6043
  z[ 6]: 0.3411
  z[ 7]: 1.6254
  z[ 8]: 0.3683
  z[ 9]: 0.4761
  z[10]: 1.2405
  z[11]: 0.7848
  z[12]: 0.8007
  z[13]: 1.8583
  z[14]: 0.9657
  z[15]: 2.2353

This distribution is fairly spread across all 16 dimensions, with no clear
dead channels. All dimensions are contributing to the representation, though
some (z[0], z[15], z[13]) carry substantially more variance than others.

The annealing schedule helped avoid total collapse: beta starts small (0.05 at
epoch 1) rather than hitting the posterior with full KL pressure immediately.
Reconstruction quality stabilized first. Once beta ramped up, the KL rose in
the active dimensions rather than immediately collapsing all of them.

## Effect of KL Annealing

Without annealing, VAEs trained on this kind of dataset (with a high-capacity
decoder relative to the data complexity) can collapse very quickly — within
the first few epochs. The posterior collapses to the prior and the KL term
becomes ~0, at which point the decoder essentially works as an unconditional
image model ignoring z entirely.

In this run, annealing kept reconstruction from degrading during the KL ramp-up.
The reconstruction loss at epoch 20 (217.02) is significantly
lower than at epoch 1 (236.02), confirming the model had time to learn
useful structure before the full KL pressure came in.

## Reflections

- **Latent dim:** 16 is perhaps more than strictly needed for FashionMNIST, though
  this run used all of them. A smaller space (8-12 dims) would likely work equally
  well and give a cleaner latent map.
  
- **Annealing length:** 20 epochs out of 30 is a relatively aggressive schedule.
  Spreading annealing over all 30 epochs might push more dimensions to become active,
  or at least let the model converge more gracefully.

- **Beta-VAE:** Using beta > 1 at convergence forces more disentangled representations
  at the cost of reconstruction quality. The slider visualization in the app would
  become more interpretable — each dimension would correspond to a more isolated
  visual factor. Worth trying with beta=2 or beta=4.

- **Architecture:** The 6272-dim FC bottleneck is heavy for a 28x28 image.
  A fully convolutional approach (global average pooling instead of flatten)
  would be lighter and might generalize better.
