# DATA 266 HW8 Metrics

All numbers come from the final top-to-bottom run of HW8_Diffusion_DDPM_DDIM.ipynb (RUN_LOG.txt, Step 5) on Apple MPS. An earlier full run (Step 4) gave identical losses and sample statistics; only the timings differed. The per-epoch values are also in results/hw8_metrics.json.

Losses are the mean squared error between the true noise ε and the predicted noise ε_θ(x_t, t), averaged over pixels. Lower is better. Sample statistics use pixels rescaled to [0, 1].

## Personal Parameters

| Field | Value | How it is calculated | HW8 use |
| --- | ---: | --- | --- |
| SID4 | 9486 | given | Reported |
| SEED | 9486 | SID4 | random, NumPy, and torch seeds, DataLoader shuffling, training timesteps and noise, the forward-figure noise, x_T, and DDPM sampling noise |
| SLICE | 486 | SID4 % 1000 | Forward-diffusion image: first test image of digit CLS_A at index 486 or later (test image 499) |
| HP_ID | 0 | SID4 % 6 | Reported only; T, the schedule, batch size, and epochs are fixed by the assignment |
| CLS_A | 6 | SID4 % 10 | Digit shown in the forward-diffusion figure |
| CLS_B | 0 | (CLS_A + 1 + (SID4 // 10) % 9) % 10 | Reported only |

## Data and Schedule

| Item | Value |
| --- | --- |
| Dataset | MNIST via torchvision (download=True into ./data), 60,000 train / 10,000 test |
| Preprocessing | ToTensor + Normalize(0.5, 0.5) → pixels in [-1, 1] |
| Tensor shapes | batch [128, 1, 28, 28], t [128], predicted noise [128, 1, 28, 28] |
| Schedule | T = 100, β linear from 0.0001 to 0.02 |
| ᾱ_t at t = 10 / 30 / 60 / 100 | 0.9900 / 0.9134 / 0.6954 / 0.3636 |
| √ᾱ_100 (image left at t = 100) | 0.603 |

## Model and Training Settings

| Setting | Value |
| --- | --- |
| Model | MiniDenoiser, U-Net-style encoder-decoder, 32-64-128 channels, GroupNorm + ReLU |
| Conv layers / parameters | 15 / 710,593 |
| Timestep conditioning | sinusoidal embedding → MLP (128-d) → per-block Linear(128, C) added per channel |
| Loss | MSE(ε, ε_θ(x_t, t)) |
| Optimizer | Adam, lr 1e-3, cosine decay to 0 (per step) |
| Batch size / epochs / steps | 128 / 50 / 23,450 |
| Training time (MPS) | 5817.9 s (97.0 min) |

## Training Results

| Metric | Value |
| --- | ---: |
| First-batch loss | 1.2902 |
| Epoch 1 / 5 / 10 / 25 / 50 mean training MSE | 0.12903 / 0.06317 / 0.05881 / 0.05445 / 0.05244 |
| Lowest epoch mean (epoch) | 0.05219 (46) |
| Test MSE before training | 1.28678 |
| **Test MSE after training** | **0.05245** |

## Sampling (64 images, all from the same x_T)

| Method | Steps | Time | Speed-up vs DDPM | Mean pixel | Ink fraction | Gray fraction | Diff vs DDPM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real MNIST (1,000 test images) | – | – | – | 0.1223 | 0.1239 | 0.0850 | – |
| DDPM | 100 | 3.56 s | 1.0x | 0.2167 | 0.2181 | **0.1596** | 0 |
| DDIM, η = 0 | 50 | 1.78 s | 2.0x | 0.2604 | 0.2463 | 0.2493 | 0.1407 |
| DDIM, η = 0 | 20 | **0.70 s** | **5.1x** | 0.2595 | 0.2445 | 0.2531 | 0.1402 |

Ink fraction = pixels above 0.5. Gray fraction = pixels between 0.1 and 0.9 (higher means more speckle or blur; real digits are mostly pure black and white). Diff vs DDPM = mean absolute pixel difference from the DDPM sample with the same x_T.

| Check | Value |
| --- | ---: |
| DDIM 20 run twice from the same x_T, max difference | 0.0 (deterministic) |
| DDPM run twice from the same x_T (noise seeds 9487 vs 9488), mean difference | 0.1450 |
| Mean difference DDIM 50 vs DDIM 20 | 0.0046 |

## Diagnostic (start noise)

| Experiment | Result |
| --- | ---: |
| DDPM 100 started from a real x_100, mean \|output − x_0\| | 0.0399 |
| DDIM 20 (η = 0) started from a real x_100, mean \|output − x_0\| | 0.0313 |
| DDIM 20 (η = 1) from pure noise: mean pixel / ink / gray fraction | 0.2144 / 0.2157 / 0.1636 |

## Short Comparison

- **Speed:** time follows the number of network calls. DDIM 50 was 2.0x faster than DDPM and DDIM 20 was 5.1x faster.
- **Fewer steps:** DDIM 50 and DDIM 20 are almost the same images (mean difference 0.0046).
- **Stochasticity:** DDIM with η = 0 is fully deterministic. DDPM gives different images from the same x_T.
- **Quality:** DDPM samples are smoother and cleaner (gray fraction 0.160 vs about 0.25 for DDIM). Most DDPM samples are still digit-like scribbles rather than clear digits.
- **Why:** from a real x_100 both samplers recover the original digits, so the model and the samplers work. The assigned schedule ends at ᾱ_100 = 0.364, so pure N(0, I) noise is not a realistic x_100. DDPM's fresh noise corrects some of this, while deterministic DDIM carries it through. DDIM with η = 1 is smooth again (gray fraction 0.164).

## Figures (figures/)

mnist_examples.png, beta_schedule.png, forward_diffusion.png, training_loss.png, ddpm_samples.png, ddpm_reverse_process.png, ddim_samples.png, ddpm_two_runs.png, ddpm_vs_ddim_grids.png, ddpm_vs_ddim_same_noise.png, reverse_paths_sample0.png, diagnostic_start_noise.png

## Hardware

Apple M1 (8 GB RAM) with the PyTorch MPS backend, torch 2.14.0, torchvision 0.29.0, numpy 2.3.5, deterministic algorithms enabled (warn_only=True; no warnings were printed). One full run takes about 1 hour 38 minutes on MPS, almost all of it training.
