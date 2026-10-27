# DATA 266 Homework 8 — Mini Diffusion Model (DDPM to DDIM)

## Overview

A small DDPM trained on MNIST (T = 100, linear β from 0.0001 to 0.02, batch 128, 50 epochs), then sampled with the full 100-step DDPM reverse process and with deterministic DDIM (η = 0) using 50 and 20 steps from the same model and the same starting noise. Built on top of `Demo_8_Diffusion.ipynb`.

| Sampler | Steps | Time (64 images, MPS) | Deterministic | Gray fraction (real MNIST 0.085) |
| --- | ---: | ---: | --- | ---: |
| DDPM | 100 | 3.56 s | No | 0.160 |
| DDIM, η = 0 | 50 | 1.78 s | Yes | 0.249 |
| DDIM, η = 0 | 20 | 0.70 s | Yes | 0.253 |

Final test noise-prediction MSE: 0.05245 (1.28678 before training). See `METRICS.md` for all numbers and `findings.pdf` for the full report.

## Personal Parameters

SID4 9486 → SEED 9486, SLICE 486, HP_ID 0, CLS_A 6, CLS_B 0.

## Folder Structure

```
HW8/
├── Demo_8_Diffusion.ipynb          professor's demo (unchanged)
├── HW8_Diffusion_DDPM_DDIM.ipynb   main notebook, executed with outputs
├── findings.pdf                    report with figures
├── METRICS.md                      all results in tables
├── RUN_LOG.txt                     how the notebook was built and run, plus the raw log
├── AI_USE.md                       AI use disclosure
├── figures/                        12 figures saved by the notebook
├── results/                        hw8_metrics.json, run_log_raw.txt
├── checkpoints/                    mini_ddpm_seed_9486.pt (not committed, *.pt is gitignored)
└── data/                           MNIST, downloaded by the notebook (not committed)
```

## Run Instructions

From inside `HW8/`, with torch, torchvision, numpy, matplotlib, and jupyter installed.

**Quick smoke test** (1,024 images, 1 epoch, checks that every cell runs). The smoke test writes to the same `figures/`, `results/`, and `checkpoints/` folders as the full run, so run it on a copy to keep the reported outputs:

```bash
mkdir -p /tmp/hw8_smoke && cp HW8_Diffusion_DDPM_DDIM.ipynb /tmp/hw8_smoke/ && cd /tmp/hw8_smoke
HW8_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute HW8_Diffusion_DDPM_DDIM.ipynb --output smoke_test_run.ipynb
```

**Full run** (overwrites the notebook outputs, figures, results, and checkpoint):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 HW8_Diffusion_DDPM_DDIM.ipynb
```

Or open the notebook in Jupyter and choose Run All. The device is chosen automatically (CUDA, then MPS, then CPU). Set `HW8_DEVICE=cpu` (or `cuda`/`mps`) to force one. On an Apple M1 the full run took about 97 minutes of training. On a CUDA GPU such as an RTX 4090 it should take only a few minutes. Results on a different device can differ slightly in the last digits because GPU kernels differ, but they are reproducible on the same device.
