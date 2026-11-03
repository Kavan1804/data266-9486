# HW4.5 tables [GPU RUN — Tesla T4]

GPU: Tesla T4

## Part A

| batch_size | run_label | epoch_s | optimizer_steps | samples_per_s | peak_alloc_gb | peak_reserved_gb | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [GPU RUN — Tesla T4] | 280.008 | 3500.000 | 12.500 | 2.107 | 2.322 | ok | nan |
| 2 | [GPU RUN — Tesla T4] | 160.324 | 1750.000 | 21.831 | 2.113 | 2.398 | ok | nan |
| 4 | [GPU RUN — Tesla T4] | 112.975 | 875.000 | 30.980 | 2.110 | 2.314 | ok | nan |
| 8 | [GPU RUN — Tesla T4] | 92.484 | 438.000 | 37.845 | 2.130 | 2.336 | ok | nan |
| 16 | [GPU RUN — Tesla T4] | 80.513 | 219.000 | 43.471 | 2.524 | 2.846 | ok | nan |
| 32 | [GPU RUN — Tesla T4] | 73.034 | 110.000 | 47.923 | 3.779 | 4.047 | ok | nan |
| 64 | [GPU RUN — Tesla T4] | 71.248 | 55.000 | 49.124 | 6.152 | 6.328 | ok | nan |
| 128 | [GPU RUN — Tesla T4] | 70.432 | 28.000 | 49.693 | 10.998 | 11.131 | ok | nan |
| 256 | [GPU RUN — Tesla T4] | nan | nan | nan | nan | nan | OOM | OutOfMemoryError: CUDA out of memory. Tried to allocate 384.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 169.81 MiB is free. Including non-PyTorch memory, this process has 14.39 GiB memory in use. Of the al |

## Table HW4.5.1 (Tesla T4)

| technique | max_physical_batch | max_effective_batch | samples_per_s | peak_alloc_gb | test_acc | train_time_s | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (HW4 settings) | 128 | 128 | 47.836 | 3.779 | 0.691 | 292.665 | ok |
| Mixed precision (fp16 autocast + GradScaler) | 256 | 256 | 148.385 | 3.061 | 0.691 | 94.349 | ok |
| Gradient checkpointing | 512 | 512 | 36.050 | 2.132 | 0.691 | 388.349 | ok |
| Gradient accumulation | 128 | 512 | 43.933 | 2.346 | 0.687 | 318.668 | ok |
| 8-bit optimizer (AdamW8bit) | 128 | 128 | 48.052 | 3.135 | 0.687 | 291.351 | ok |
| Best combination: Mixed precision (fp16 autocast + GradScaler) + Gradient accumulation + 8-bit optimizer (AdamW8bit) | 256 | 1024 | 116.290 | 1.704 | 0.677 | 120.389 | ok |

## Ranking

| rank | technique | memory_saved_gb | throughput_lost_sps | score_gb_per_sps | note |
| --- | --- | --- | --- | --- | --- |
| 1 | Mixed precision (fp16 autocast + GradScaler) | 0.718 | -100.548 | inf | saves memory with no throughput loss |
| 2 | 8-bit optimizer (AdamW8bit) | 0.643 | -0.216 | inf | saves memory with no throughput loss |
| 3 | Gradient accumulation | 1.433 | 3.903 | 0.367 | nan |
| 4 | Gradient checkpointing | 1.647 | 11.786 | 0.140 | nan |

## Part D serving

| config | tokens_per_s | p50_ms | p95_ms | requests_per_s | total_requests | total_generated_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| HF pipeline, batch 1 | 32.0 | 1903.4 | 2388.8 | 0.5 | 256 | 16384 |
| vLLM, batch 1 | 161.9 | 394.0 | 406.5 | 2.5 | 256 | 16384 |
| vLLM, batch 8 | 1191.2 | 429.1 | 438.8 | 18.6 | 256 | 16384 |
| vLLM, batch 32 | 3284.7 | 622.6 | 638.0 | 51.3 | 256 | 16384 |

## Part D vs HW4.a

| config | tokens_per_s | p50_ms | p95_ms | tokens_per_s_hw4a | p50_ms_hw4a | p95_ms_hw4a | tokens_per_s_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HF pipeline, batch 1 | 32.0 | 1903.4 | 2388.8 | 29.5 | 1967.6 | 2724.9 | 8.7 |
| vLLM, batch 1 | 161.9 | 394.0 | 406.5 | 164.4 | 389.8 | 399.8 | -1.5 |
| vLLM, batch 8 | 1191.2 | 429.1 | 438.8 | 1174.5 | 435.0 | 445.8 | 1.4 |
| vLLM, batch 32 | 3284.7 | 622.6 | 638.0 | 3214.0 | 634.6 | 651.4 | 2.2 |

## KV cache (HF, batch 8)

| context_len | batch_size | new_tokens | kv_theory_gb | weights_gb | run_label | status | latency_ms | gen_tokens_per_s | peak_alloc_gb | peak_reserved_gb | peak_above_weights_gb | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 8 | 16 | 0.048 | 0.990 | [GPU RUN — Tesla T4] | ok | 701.683 | 182.419 | 1.177 | 1.254 | 0.187 | nan |
| 1024 | 8 | 16 | 0.095 | 0.990 | [GPU RUN — Tesla T4] | ok | 914.202 | 140.013 | 1.364 | 1.439 | 0.373 | nan |
| 2048 | 8 | 16 | 0.189 | 0.990 | [GPU RUN — Tesla T4] | ok | 1476.217 | 86.708 | 1.737 | 1.865 | 0.747 | nan |
| 4096 | 8 | 16 | 0.376 | 0.990 | [GPU RUN — Tesla T4] | ok | 2893.225 | 44.241 | 2.484 | 2.699 | 1.493 | nan |
| 8192 | 8 | 16 | 0.751 | 0.990 | [GPU RUN — Tesla T4] | ok | 6450.239 | 19.844 | 3.977 | 4.369 | 2.986 | nan |
| 16384 | 8 | 16 | 1.501 | 0.990 | [GPU RUN — Tesla T4] | ok | 16543.188 | 7.737 | 6.963 | 7.680 | 5.973 | nan |
| 32752 | 8 | 16 | 3.000 | 0.990 | [GPU RUN — Tesla T4] | OOM | nan | nan | nan | nan | nan | OutOfMemoryError: CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a total capacity of 14.56 GiB of which 989.81 MiB is free. Including non-PyTorch memory, this process has 13.59 GiB memory in use. Of the allo |
