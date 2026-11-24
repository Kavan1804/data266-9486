# DATA 266 HW12 Metrics

All numbers are from the full Kaggle run on 2026-11-24 (see `results/hw12_metrics.json`). Table names follow the assignment.

## Personal Parameters

| Field | Value | How it is calculated | HW12 use |
| --- | ---: | --- | --- |
| SID4 | 9486 | given | Reported |
| SEED | 9486 | SID4 | calibration windows, validation split, student head, batch order |
| SLICE | 486 | SID4 % 1000 | start of the latency prompt and latency articles |
| HP_ID | 0 | SID4 % 6 | Reported only (adds no temperature, as the assignment says) |
| CLS_A | 6 | SID4 % 10 | Reported only |
| CLS_B | 0 | (CLS_A + 1 + (SID4 // 10) % 9) % 10 | Reported only |

## Setup

| Item | Value |
| --- | --- |
| Hardware | one Tesla T4 (Kaggle), Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2 |
| Base LLM | `facebook/opt-125m` @ `27dcfa74d334bc871f3234de431e71c6eeba5dd6` |
| GPTQ | 4-bit asymmetric, group 128, 1% damping, 128 calibration windows of 512 tokens (Wikitext-2 train), 30.1 s |
| Student | `distilbert/distilbert-base-uncased` @ `12040accade4e8a0f71eabdb258fecc2e7e948be` |
| Distillation data | AG News: 120,000 train (6,000 held out for validation), 7,600 test |
| Total run time | 48.0 min (about 5.7 min per student) |

## Table HW11.1 — GPTQ comparison on the same LLM/task

Quality = Wikitext-2 test perplexity / evaluation loss (287,645 tokens).

| Configuration | Quality metric | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
| --- | ---: | ---: | ---: | ---: |
| Base LLM (fp16) | 27.66 / 3.3200 | 513.7 / 540.9 | 0.276 | 250.5 |
| GPTQ-quantized LLM (4-bit) | 29.72 / 3.3918 | 1257.6 / 1296.6 | 0.159 | 125.1 |

Round-to-nearest check (same 4-bit grid): perplexity 30.50. Peak memory in the perplexity run: 1.297 GB (base), 1.166 GB (GPTQ).

## Table HW11.2 — Distillation comparison on AG News

| Configuration | Accuracy | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
| --- | ---: | ---: | ---: | ---: |
| Teacher (BERT-base) | 94.43% | 8.12 / 9.42 | 0.757 | 438.0 |
| Best distilled student (T = 10, α = 0.25) | 94.30% | 4.01 / 4.38 | 0.565 | 267.8 |

Whole test set: 21.4 s (teacher), 10.7 s (student). Parameters: 109,485,316 and 66,956,548.

## Temperature sweep (α = 0.5)

| T | Validation accuracy | Test accuracy |
| ---: | ---: | ---: |
| 1 | 94.28% | 94.46% |
| 2 | 94.28% | 94.46% |
| 5 | 94.18% | 94.36% |
| **10 (best by validation)** | 94.32% | 94.30% |

## Loss-weighting sweep (T = 10, three fresh students)

| α | Validation accuracy | Test accuracy |
| ---: | ---: | ---: |
| **0.25 (best by validation)** | 94.35% | 94.30% |
| 0.5 | 94.32% | 94.30% |
| 0.75 | 94.30% | 94.41% |

## Per-class test accuracy (1,900 articles each)

| Class | Teacher | Student | Change (points) |
| --- | ---: | ---: | ---: |
| World | 94.89% | 94.79% | −0.11 |
| Sports | 98.84% | 98.68% | −0.16 |
| **Business (most degraded)** | 91.37% | 90.84% | −0.53 |
| Sci/Tech | 92.63% | 92.89% | +0.26 |

Teacher right and student wrong: 123. Teacher wrong and student right: 113.
