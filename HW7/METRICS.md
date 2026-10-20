# DATA 266 HW7 Metrics

All numbers are from the final top-to-bottom run of HW7_GAN_AE_VAE.ipynb (RUN_LOG.txt, Step 4) on CPU. A second, earlier run gave identical values. The per-epoch values are also in results/hw7_metrics.json.

Every loss is **per image**: binary cross entropy summed over the 784 pixels and averaged over images, plus (for the VAE) the KL divergence summed over the 32 latent dimensions and averaged over images. Lower is better for every metric.

## Personal Parameters

| Field | Value | How it is calculated | HW7 use |
| --- | ---: | --- | --- |
| SID4 | 9486 | given | Reported |
| SEED | 9486 | SID4 | random, NumPy, and torch seeds, the train/validation split, DataLoader shuffling, VAE noise, prior samples |
| SLICE | 486 | SID4 % 1000 | Display images are the first test image of each class at index 486 or later |
| HP_ID | 0 | SID4 % 6 | Reported only; this assignment has no HP_ID mapping, so each model uses one configuration and there is no sweep |
| CLS_A | 6 | SID4 % 10 | Reported only |
| CLS_B | 0 | (CLS_A + 1 + (SID4 // 10) % 9) % 10 | Reported only |

## Data

| Item | Value |
| --- | --- |
| Dataset | FashionMNIST via torchvision (download=True into ./data) |
| Split | 54,000 train / 6,000 validation (random_split 0.9/0.1, seeded with SEED) / 10,000 test |
| Preprocessing | ToImage, ToDtype(float32, scale=True) → [0, 1], flatten → 784 |
| Tensor shapes | image [784], batch [256, 784], latent [256, 32], output [256, 784] |
| Display test indices | [488, 487, 497, 490, 491, 502, 512, 494, 486, 493] (one per class) |

## Models and Settings

| Setting | FC-AE | VAE |
| --- | --- | --- |
| Encoder | Linear(784, 256) + ReLU → Linear(256, 32) | Linear(784, 256) + ReLU → Linear(256, 32) for mu and Linear(256, 32) for logvar |
| Latent | z, 32-d | z = mu + exp(0.5·logvar)·ε, ε ~ N(0, I), 32-d |
| Decoder | Linear(32, 256) + ReLU → Linear(256, 784) + Sigmoid | same |
| Parameters | 419,120 | 427,344 |
| Loss | BCE (reconstruction) | BCE (reconstruction) + KL(q(z\|x) ‖ N(0, I)) |
| Epochs | 20 | 20 |
| Batch size | 256 (211 train batches per epoch) | 256 |
| Optimizer / learning rate | Adam, 1e-3 | Adam, 1e-3 |
| Latent dimension | 32 | 32 |
| Training time (CPU, final run) | 114.3 s | 120.7 s |

## Final Training and Validation Losses (epoch 20)

| Metric | FC-AE | VAE |
| --- | ---: | ---: |
| Final training loss | 212.3813 | 243.0421 (recon 226.9387 + KL 16.1034) |
| Final validation loss | 212.2384 | 243.0682 (recon 226.4851 + KL 16.5831) |
| Best validation loss (epoch) | 212.2384 (20) | 242.8593 (19) |
| First-epoch training loss | 288.3456 | 317.3855 (recon 303.6442 + KL 13.7414) |

## Test Set (10,000 images)

| Metric | FC-AE | VAE |
| --- | ---: | ---: |
| **Reconstruction BCE, deterministic** (VAE decodes z = mu) | **214.1978** | 223.7607 |
| **MSE per pixel, deterministic** | **0.009698** | 0.013927 |
| VAE reconstruction BCE (sampled z, training objective) | – | 228.3524 |
| VAE KL divergence | – | 16.5931 |
| Total loss (FC-AE: BCE; VAE: recon + KL, sampled z) | 214.1978 | 244.9455 |

## Short Comparison

- **Reconstruction:** the FC-AE is better on both metrics measured the same way. Its BCE is 9.56 lower per image (214.20 vs 223.76), and its MSE per pixel is about 30% lower (the VAE's is about 44% higher). Its reconstructions are also visibly a bit sharper (figures/ae_vs_vae_reconstructions.png). Both models lose fine texture such as prints, straps, and buttons.
- **Cost of the KL term:** the VAE pays about 16.6 per image for the KL term, and sampling z instead of using mu raises its reconstruction BCE from 223.76 to 228.35.
- **Generation:** decoding 64 random z ~ N(0, I) with the VAE gives mostly recognizable but blurry clothing items, with some ambiguous or striped samples (figures/vae_prior_samples.png). The FC-AE was not trained to generate, so there are no generation metrics for it.
- **Training:** both models trained smoothly with no sign of overfitting. The FC-AE's validation loss was below its training loss at every epoch. The VAE's was below its training loss through epoch 19 and only 0.03 above it at epoch 20 (243.07 vs 243.04). The VAE's training KL rose slowly (15.54 at epoch 5 to 16.10 at epoch 20) while its reconstruction term fell (238.36 to 226.94).

## Figures (figures/)

fashionmnist_display_images.png, fc_ae_loss_curves.png, fc_ae_reconstructions.png, vae_loss_curves.png, vae_reconstructions.png, vae_prior_samples.png, ae_vs_vae_reconstructions.png

## Hardware

CPU only (Apple M1, 8 GB RAM), torch 2.14.1, torchvision 0.29.1, numpy 2.5.4, deterministic algorithms enabled. One full run takes about 4 minutes once the data is downloaded.
