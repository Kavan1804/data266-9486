# DATA 266 HW6

## Overview

Homework 6 compares three ways of training the same ResNet-18 encoder (torchvision, no pretrained weights) on STL-10 when only 500 labels are available. The notebook builds on the professor's Demo_6_Self_Supervised_Learning.ipynb, moving it from CIFAR-10 and a tiny CNN to STL-10 and ResNet-18. It covers:

1. **Part A, supervised:** ResNet-18 trained end to end on a balanced 500-image subset (10% of the labeled train split) for 15 epochs.
2. **Part B, rotation SSL:** pretraining on all 100,000 unlabeled images by predicting 0°/90°/180°/270° rotations for 15 epochs, then a frozen encoder with a linear classifier trained on the same 500 images for 20 epochs.
3. **Part C, SimCLR:** NT-Xent contrastive pretraining (cosine similarity, tau = 0.2, MLP projection head, the demo's four augmentations) on 20,000 unlabeled images for 20 epochs, then a frozen encoder with a linear classifier on the same 500 images for 20 epochs.
4. **Part D, nearest neighbors:** top-5 cosine nearest neighbors in the 8,000-image test set for the same three query images under all three encoders.

The submitted results come from one run on Google Colab with a Tesla T4 GPU.

| Model | Test accuracy |
| --- | ---: |
| A: Supervised (500 labels) | 34.80% |
| B: Rotation SSL + linear | 42.58% |
| C: SimCLR + linear | 50.45% |

## Personal Parameters

| Field | Value | HW6 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | Python, NumPy, and PyTorch seeds, the 500-image labeled subset, the 20,000-image SimCLR subset, DataLoader shuffling and worker seeds |
| SLICE | 486 | Starting test index for picking the nearest-neighbor query images |
| HP_ID | 0 | Reported only |
| CLS_A | 6 | Class of the first query image (horse) |
| CLS_B | 0 | Class of the second query image (airplane) |

## Folder Structure

- `HW6_STL10_SSL.ipynb`: the main submission notebook, executed on Colab with outputs intact
- `Demo_6_Self_Supervised_Learning.ipynb`: the professor's reference demo, not modified
- `figures/`: training curves for Parts A to C and the three nearest-neighbor figures for Part D
- `results/hw6_metrics.json`: accuracies, losses, per-epoch histories, settings, labeled indices, nearest-neighbor results, and run times
- `results/run_log_raw.txt`: the timestamped console lines written by the notebook during the Colab run
- `METRICS.md`: summary of results, settings, and run times
- `RUN_LOG.txt`: step-by-step run history, including the console output of the reported run
- `findings.pdf`: short findings write-up
- `AI_USE.md`: AI use disclosure
- `README.md`: this file

The notebook also saves the three encoders to `checkpoints/` (about 45 MB each). They are not in the repository, since `.gitignore` excludes `*.pt` files and they were written to Colab's local disk.

## Run Instructions

Open `HW6_STL10_SSL.ipynb` in Colab with a T4 GPU runtime and run all cells. Nothing needs to be installed, since Colab already has torch, torchvision, numpy, and matplotlib. STL-10 (about 2.6 GB) is downloaded by the notebook into `./data`. The last cell zips `figures/` and `results/` into `hw6_outputs.zip` and downloads it, because Colab's disk is not synced to GitHub. The full run took about 62 minutes on a T4.
