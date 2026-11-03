# DATA 266 Homework 9 — Latent Diffusion with Classifier-Free Guidance

## Overview

A class-conditional Latent Diffusion Model on FashionMNIST. Images are resized to 256×256 (3 channels, [-1, 1]) and encoded with the pretrained Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`) into 4×32×32 latents scaled by 0.18215. A U-Net (base 64 channels, multipliers (1, 2, 4), self-attention at 16×16 and in the bottleneck, time + label embeddings with a null label) is trained to predict the noise in the latent space (T = 1000, linear β from 1e-4 to 0.02, 50 epochs, lr 2e-5, label dropout 0.1). Samples are generated with classifier-free guidance (w = 3) and decoded with the VAE. Built on top of `Demo_9_Stable_Diffusion.ipynb` and `Demo_9_CFG.ipynb`.

Results from the full run on a Kaggle T4 (about 2 hours 5 minutes in total):

- VAE reconstruction MSE: 0.00013 on the [-1, 1] scale (1,000 test images).
- Final test noise-prediction MSE: 0.0918, down from 1.0727 before training. The loss was still falling at epoch 50.
- Generated samples: weak. At w = 3, Trouser and Dress are the most recognizable, but Sandal and Bag come out almost black. A small latent classifier agreed with the requested label for 20.3% of samples at w = 3 and 18.8% at w = 0.

The full numbers are in `METRICS.md`, and the discussion is in Section 19 of the notebook.

## Personal Parameters

SID4 9486 → SEED 9486, SLICE 486, HP_ID 0, CLS_A 6, CLS_B 0.

## Folder Structure

```
HW9/
├── Demo_9_Stable_Diffusion.ipynb    professor's demo (unchanged)
├── Demo_9_CFG.ipynb                 professor's demo (unchanged)
├── HW9_Latent_Diffusion_CFG.ipynb   main notebook with the outputs of the Kaggle run
├── METRICS.md                       results in tables
├── RUN_LOG.txt                      how the notebook was built and run
├── AI_USE.md                        AI use disclosure
├── figures/                         figures saved by the notebook
├── results/                         hw9_metrics.json, run_log_raw.txt
├── checkpoints/                     ldm_unet_seed_9486.pt (not committed, *.pt is gitignored)
├── cache/                           VAE latent cache (not committed)
└── data/                            FashionMNIST, downloaded by the notebook (not committed)
```

## Run Instructions

Only `HW9_Latent_Diffusion_CFG.ipynb` needs to be run. The two demo notebooks are reference material.

**Full run on Kaggle (what I used):** upload the notebook, set the accelerator to GPU T4, and click Run All. The notebook downloads FashionMNIST and the VAE by itself, and no Hugging Face login is needed. Kaggle has no Google Drive backup, so the run has to finish in one session. The last cell writes `HW9_outputs.zip` with `figures/`, `results/`, and the final weights. Download the executed notebook separately with File → Download Notebook.

**Full run on Google Colab:** choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. On Colab the notebook asks once for Google Drive access and backs up the latent cache and the latest checkpoint to `MyDrive/HW9_backup`, so after a disconnect, Run all on a new runtime continues from the last finished epoch. The last cell also downloads the zip.

**Smoke test (any machine, checks that every cell runs):** 64 training and 32 test images, 2 epochs, 10 samples. Run it on a copy so it does not overwrite the reported outputs:

```bash
mkdir -p /tmp/hw9_smoke && cp HW9_Latent_Diffusion_CFG.ipynb /tmp/hw9_smoke/ && cd /tmp/hw9_smoke
HW9_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 HW9_Latent_Diffusion_CFG.ipynb --output smoke_run.ipynb
```

The device is chosen automatically (CUDA, then MPS, then CPU). Set `HW9_DEVICE` to force one, and `HW9_MICRO_BATCH` (default 250) to change how many latents go through the U-Net at once; the effective batch stays 1000.
