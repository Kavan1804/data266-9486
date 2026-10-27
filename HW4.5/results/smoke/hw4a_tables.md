# HW4.a tables [SMOKE TEST]

## Per-epoch history

| epoch | train_loss | val_loss | val_acc | epoch_train_s | val_eval_s | train_samples_per_s |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.9920 | 1.1338 | 0.3125 | 11.8667 | 0.7209 | 4.0449 |
| 2.0000 | 0.8533 | 1.1530 | 0.4375 | 11.7235 | 0.7292 | 4.0943 |

## Per-class test metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| negative | 0.0000 | 0.0000 | 0.0000 | 4 |
| neutral | 0.4667 | 0.6364 | 0.5385 | 11 |
| positive | 0.5556 | 0.5556 | 0.5556 | 9 |

## Table HW4.1 — Serving

| config | tokens_per_s | p50_ms | p95_ms | requests_per_s | total_requests | total_generated_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| HF pipeline, batch 1 | 9.8 | 815.5 | 846.1 | 1.2 | 2 | 16 |
