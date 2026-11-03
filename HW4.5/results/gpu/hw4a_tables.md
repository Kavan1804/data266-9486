# HW4.a tables [GPU RUN — Tesla T4]

## Per-epoch history

| epoch | train_loss | val_loss | val_acc | epoch_train_s | val_eval_s | train_samples_per_s |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.9412 | 0.8019 | 0.6520 | 66.0434 | 4.7701 | 52.9954 |
| 2.0000 | 0.6903 | 0.6946 | 0.6973 | 71.0963 | 5.4724 | 49.2290 |
| 3.0000 | 0.5283 | 0.6993 | 0.7013 | 76.6041 | 5.5645 | 45.6895 |
| 4.0000 | 0.4180 | 0.7165 | 0.7027 | 76.3790 | 5.6632 | 45.8241 |

## Per-class test metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| negative | 0.5455 | 0.5690 | 0.5570 | 116 |
| neutral | 0.7069 | 0.6822 | 0.6944 | 343 |
| positive | 0.7315 | 0.7491 | 0.7402 | 291 |

## Table HW4.1 — Serving

| config | tokens_per_s | p50_ms | p95_ms | requests_per_s | total_requests | total_generated_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| HF pipeline, batch 1 | 29.5 | 1967.6 | 2724.9 | 0.5 | 256 | 16384 |
| vLLM, batch 1 | 164.4 | 389.8 | 399.8 | 2.6 | 256 | 16384 |
| vLLM, batch 8 | 1174.5 | 435.0 | 445.8 | 18.4 | 256 | 16384 |
| vLLM, batch 32 | 3214.0 | 634.6 | 651.4 | 50.2 | 256 | 16384 |
