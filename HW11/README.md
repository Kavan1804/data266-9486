# DATA 266 Homework 11 — CLIP Caption-Image Similarity and Zero-Shot CIFAR-10

## Overview

Both parts use OpenAI's CLIP `openai/clip-vit-base-patch32` through `transformers` (`CLIPModel` + `CLIPProcessor`), built on top of `Demo_11_CLIP.ipynb`. The CIFAR-10 loading and evaluation loop come from `Demo_11_Knowledge_Distillation.ipynb`. Nothing is trained.

- **Part 1:** one caption, `"a photo of a shirt."` (from my CLS_A = 6 → Shirt), compared with 100 FashionMNIST test images from each of the 10 classes (1,000 images, starting at index SLICE = 486). The image and caption embeddings are L2-normalized, and their dot product is the cosine similarity. The notebook shows the top 16 and bottom 16 images, 5 images per class with their scores, and the score distribution per class.
- **Part 2:** zero-shot classification of all 10,000 CIFAR-10 test images with one prompt per class, with per-class accuracy, a confusion matrix, the most common mistakes, and misclassified examples. Three prompt sets are compared.

Results from the full run on my Apple M1 (MPS), 3.0 minutes:

- **Part 1:** the scores are between 0.170 and 0.286. Upper-body clothes rank highest and shoes lowest (ROC AUC Shirt vs. rest 0.840), but **T-shirt/top scores slightly higher than Shirt** (mean 0.2552 vs. 0.2525, AUC Shirt vs. T-shirt 0.429). Only 37 of the top 100 images are shirts, and 49 are T-shirts.
- **Part 2:** **88.33%** zero-shot accuracy with `"a photo of a {}."` (89.06% with the demo's prompt and 89.17% with an 8-template ensemble). Per class it goes from 97.1% (horse) to 71.5% (frog). The biggest confusion is deer → horse (137 images).

The full numbers are in `METRICS.md`, and the discussion is in Sections 10, 15 and 16 of the notebook.

## Personal Parameters

SID4 9486 → SEED 9486, SLICE 486, HP_ID 0, CLS_A 6, CLS_B 0.

## Folder Structure

```
HW11/
├── Demo_11_CLIP.ipynb                     professor's demo (unchanged)
├── Demo_11_Knowledge_Distillation.ipynb   professor's demo (unchanged)
├── HW11_CLIP_Similarity_ZeroShot.ipynb    main notebook with the outputs of the M1 run
├── METRICS.md                             results in tables
├── RUN_LOG.txt                            how the notebook was built and run
├── AI_USE.md                              AI use disclosure
├── figures/                               the 8 figures saved by the notebook
├── results/                               hw11_metrics.json, part1_similarity_scores.csv, part2_predictions.csv, run_log_raw.txt
├── data/                                  FashionMNIST and CIFAR-10, downloaded by the notebook (not committed, 437 MB)
└── cache/                                 CIFAR-10 image embeddings (not committed)
```

## Run Instructions

Only `HW11_CLIP_Similarity_ZeroShot.ipynb` needs to be run. The two demo notebooks are reference material.

1. The packages are `torch`, `torchvision`, `transformers`, `scikit-learn`, `pandas` and `matplotlib`. The notebook installs any that are missing.
2. Open the notebook in the `HW11` folder and Run All. No API keys or Hugging Face login are needed.

The first run downloads CLIP (about 600 MB), FashionMNIST and CIFAR-10 (170 MB; the Toronto server was slow for me, about 25 minutes). After that, the whole notebook takes about 3 minutes on my M1. Encoding the 10,000 CIFAR-10 images runs at about 66 images/s, and a re-run loads them from `cache/`.

**Smoke test (checks that every cell runs):** 10 images per class in Part 1 and 500 CIFAR-10 images in Part 2. Run it on a copy so it does not overwrite the reported outputs:

```bash
mkdir -p /tmp/hw11_smoke && cp HW11_CLIP_Similarity_ZeroShot.ipynb /tmp/hw11_smoke/ && cd /tmp/hw11_smoke
HW11_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 HW11_CLIP_Similarity_ZeroShot.ipynb --output smoke_run.ipynb
```

The device is chosen automatically (CUDA, then MPS, then CPU). Set `HW11_DEVICE` to force one.

**Note for Mac users:** keep `torch` imported before `sklearn` (the notebook already does this). With the opposite order, the kernel crashed with a segmentation fault on my M1 (see `RUN_LOG.txt`, Step 3).
