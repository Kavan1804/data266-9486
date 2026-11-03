# DATA 266 HW9 Metrics

All numbers come from the full run of `HW9_Latent_Diffusion_CFG.ipynb` on a Kaggle T4 GPU (see `results/hw9_metrics.json` and `results/run_log_raw.txt`). Losses are the MSE between the true and predicted noise in the scaled latent space. Lower is better.

## Personal Parameters

| Field | Value | How it is calculated | HW9 use |
| --- | ---: | --- | --- |
| SID4 | 9486 | given | Reported |
| SEED | 9486 | SID4 | all random seeds: latent samples, shuffle, timesteps, noise, label dropout, test-loss noise, x_T, sampling noise |
| SLICE | 486 | SID4 % 1000 | reconstruction set = test images 486 to 1485; forward-diffusion image = first Shirt at index ≥ 486 (index 512) |
| HP_ID | 0 | SID4 % 6 | Reported only (second configuration skipped, as allowed) |
| CLS_A | 6 | SID4 % 10 | class of the forward-diffusion image (Shirt) |
| CLS_B | 0 | (CLS_A + 1 + (SID4 // 10) % 9) % 10 | Reported only |

## Setup

| Item | Value |
| --- | --- |
| Data | FashionMNIST, 60,000 train / 10,000 test |
| Preprocessing | resize 256×256 (bilinear), grayscale → 3 channels, [-1, 1] |
| VAE | stabilityai/sd-vae-ft-mse (frozen), 83,653,863 parameters, latent 4×32×32, scale 0.18215 |
| Schedule | T = 1000, linear β 1e-4 → 0.02 |
| U-Net | base 64, mults (1, 2, 4), attention at 16×16 (e2, d2) and 4×4 bottleneck, 6,406,596 parameters |
| Training | 50 epochs, Adam lr 2e-5, batch 1000 (micro-batch 250), 3,000 optimizer steps, label dropout 0.1 (per batch, as in the demo) |
| Sampling | DDPM 1000 steps, CFG w = 3, 64 samples, labels 0-9 repeated |
| Hardware | Kaggle, Tesla T4 (15.6 GB), mixed precision, peak GPU memory 13.31 GB |
| Software | Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, diffusers 0.42.0, numpy 2.1.3 |

## Runtime

| Stage | Time |
| --- | ---: |
| Reconstruction check (1,000 images) | 229.4 s |
| Latent caching (70,000 images) | 2,368.7 s (39.5 min) |
| Training (50 epochs, about 94 s each) | 4,696.7 s (78.3 min) |
| Whole notebook, from the first to the last log line | 2 h 4 min 44 s |

## VAE Reconstruction (1,000 test images)

| Metric | Value |
| --- | ---: |
| MSE, sampled latent ([-1, 1] scale) | 0.00013155 |
| MSE, mean latent ([-1, 1] scale) | 0.00013155 |
| MSE, sampled latent ([0, 1] scale) | 0.00003289 |
| Highest per-class MSE (Bag) | 0.00014591 |
| Lowest per-class MSE (Coat) | 0.00012155 |
| Scaled latent mean / std (train) | -0.0126 / 1.3700 |

## Forward Diffusion (Shirt, test image 512)

| t | ᾱ_t | MSE of decode(z_t) vs. decode(z_0) |
| ---: | ---: | ---: |
| 0 | 1.0000 | 0.0000 |
| 200 | 0.6590 | 0.0662 |
| 500 | 0.0786 | 0.5357 |
| 800 | 0.0015 | 0.8400 |
| 1000 | 0.00004 | 0.8840 |

## Training

Test loss before training: 1.0727

| Epoch | Train loss | Test loss |
| ---: | ---: | ---: |
| 1 | 0.8995 | 0.7781 |
| 5 | 0.2392 | 0.2188 |
| 10 | 0.1678 | 0.1642 |
| 20 | 0.1270 | 0.1272 |
| 30 | 0.1083 | 0.1083 |
| 40 | 0.0979 | 0.0974 |
| 50 | 0.0935 | 0.0918 |

| Final test loss | Value |
| --- | ---: |
| As during training (per-batch dropout) | 0.0918 |
| Conditional | 0.0915 |
| Unconditional (null label) | 0.0934 |

## One-Step Denoising (real test latents)

| t | Latent MSE, noisy | Latent MSE, denoised |
| ---: | ---: | ---: |
| 200 | 0.3943 | 0.0682 |
| 500 | 1.7935 | 0.2718 |

## Sampling

| Metric | w = 0 | w = 3 |
| --- | ---: | ---: |
| Sampling time (64 images) | 27.9 s | 51.0 s |
| Latent classifier agreement with requested label | 18.8% (12/64) | 20.3% (13/64) |
| Generated z_0 std (real train latents: 1.37) | 1.69 | 1.84 |

Latent classifier accuracy on real test latents: 71.0%. This is a rough check only, not a quality metric like FID.

Agreement by class at w = 3 (6 or 7 samples per class): Sneaker 100%, Dress 57%, Trouser 43%, and 0% for T-shirt/top, Pullover, Coat, Sandal, Shirt, Bag, and Ankle boot.

The samples are weak. In the w = 3 grid, Trouser and Dress are the most recognizable, while all the Sandal and Bag panels are basically black. See Section 19 of the notebook.

## Figures (figures/)

fashionmnist_examples.png, vae_reconstructions.png, latent_channels.png, beta_schedule.png, forward_diffusion_latent.png, loss_curves.png, cfg_samples_w3.png, cfg_denoising_steps.png, cfg_w0_vs_w3.png, cfg_samples_w0.png, one_step_denoising.png
