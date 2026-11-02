# HW4.5 tables [SMOKE TEST]

GPU: none (CPU only)

## Part A

| batch_size | run_label | epoch_s | optimizer_steps | samples_per_s | peak_alloc_gb | peak_reserved_gb | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [SMOKE TEST] | 38.501 | 48 | 1.247 | nan | nan | ok | nan |
| 2 | [SMOKE TEST] | 25.851 | 24 | 1.857 | nan | nan | ok | nan |
| 4 | [SMOKE TEST] | 23.041 | 12 | 2.083 | nan | nan | ok | nan |

## Table HW4.5.1 (none (CPU only))

| technique | max_physical_batch | max_effective_batch | samples_per_s | peak_alloc_gb | test_acc | train_time_s | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (HW4 settings) | 8.000 | 8.000 | 3.635 | nan | 0.500 | 26.410 | ok |
| Mixed precision (bf16 autocast) | 8.000 | 8.000 | 0.068 | nan | 0.500 | 1412.047 | ok |
| Gradient checkpointing | 8.000 | 8.000 | 2.981 | nan | 0.500 | 32.207 | ok |
| Gradient accumulation | 8.000 | 32.000 | 2.877 | nan | 0.583 | 33.365 | ok |
| 8-bit optimizer (AdamW8bit) | nan | nan | nan | nan | nan | nan | not run |
| Best combination | nan | nan | nan | nan | nan | nan | not run |

## Ranking

| rank | technique | memory_saved_gb | throughput_lost_sps | score_gb_per_sps | note |
| --- | --- | --- | --- | --- | --- |
| - | Mixed precision (bf16 autocast) | nan | 3.567 | nan | peak memory not measurable on CPU |
| - | Gradient checkpointing | nan | 0.654 | nan | peak memory not measurable on CPU |
| - | Gradient accumulation | nan | 0.758 | nan | peak memory not measurable on CPU |
| - | 8-bit optimizer (AdamW8bit) | nan | nan | nan | excluded: not run [DEPENDENCY ERROR] ModuleNotFoundError: No module named 'bitsandbytes' |

## Part D serving

| config | tokens_per_s | p50_ms | p95_ms | requests_per_s | total_requests | total_generated_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| HF pipeline, batch 1 | 9.9 | 810.9 | 827.4 | 1.2 | 2 | 16 |

## Part D vs HW4.a

| config | tokens_per_s | p50_ms | p95_ms | tokens_per_s_hw4a | p50_ms_hw4a | p95_ms_hw4a | tokens_per_s_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HF pipeline, batch 1 | 9.9 | 810.9 | 827.4 | 9.5 | 843.1 | 890.3 | 4.0 |

## KV cache (HF, batch 8)

| context_len | batch_size | new_tokens | kv_theory_gb | weights_gb | run_label | status | latency_ms | gen_tokens_per_s | peak_alloc_gb | peak_reserved_gb | peak_above_weights_gb | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 8 | 4 | 0.012 | nan | [SMOKE TEST] | ok | 2238.260 | 14.297 | nan | nan | nan | nan |
| 128 | 8 | 4 | 0.024 | nan | [SMOKE TEST] | ok | 3320.255 | 9.638 | nan | nan | nan | nan |
