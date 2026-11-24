# DATA 266 Homework 12 — GPTQ Quantization and Knowledge Distillation

## Overview

- **Part 1:** quantizes `facebook/opt-125m` (revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`) to 4 bits with GPTQ, written in PyTorch with packed weights. The base and quantized models are compared on Wikitext-2 test perplexity, p50/p95 latency, peak GPU memory and size on disk (Table HW11.1).
- **Part 2:** trains `distilbert-base-uncased` from the BERT teacher we were given (AG News). First I compare T = 1, 2, 5, 10 at α = 0.5, then α = 0.25, 0.5, 0.75 at the best T. After that the teacher and the best student are compared on the AG News test set (Table HW11.2), with per-class accuracy and five examples where the student fails. T and α are picked on 6,000 held-out training samples, not on the test set.

The instructions say "HW11" in some places, so I kept those table names. The other demo files in the folder are about agents and aren't used.

## Results (full run on a Kaggle T4, 48 minutes)

- **GPTQ:** perplexity 27.66 → 29.72, size 250.5 → 125.1 MB, peak memory 0.276 → 0.159 GB. It's 2.4× slower (p50 514 → 1258 ms) because the weights are unpacked in Python.
- **Distillation:** the best student (T = 10, α = 0.25, picked by validation) gets 94.30%, and the teacher gets 94.43%. It's 2× faster (p50 8.12 → 4.01 ms) and 39% smaller. The T and α settings are almost tied. Business drops the most (−0.53 points).

The findings are in Sections 7, 15 and 16 of the notebook, and all the numbers are in `METRICS.md`.

## Files

- `HW12_GPTQ_Distillation.ipynb`: main notebook with the outputs
- `figures/`, `results/`: loss curves, CSVs, metrics JSON and the raw log
- `METRICS.md`, `RUN_LOG.txt`, `AI_USE.md`

## Personal Parameters

SID4 9486 → SEED 9486, SLICE 486, HP_ID 0, CLS_A 6, CLS_B 0.

## Run Instructions (Kaggle, GPU T4)

1. Upload `teacher_model.7z` (402 MB, too big to commit) as a private Kaggle dataset. No need to unzip it, the notebook does that.
2. Upload `HW12_GPTQ_Distillation.ipynb`, attach that dataset, set Accelerator to GPU T4 x2 and Internet to On.
3. Optional, but worth doing first to check that everything runs: put `import os; os.environ["HW12_SMOKE_TEST"] = "1"` in a new first cell and Run All. Remove that cell and restart the session before the real run.
4. Save Version → Save & Run All. Then download the executed notebook and `HW12_outputs.zip`.
