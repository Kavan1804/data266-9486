# DATA 266 HW2.5 Metrics

## Personal Parameters

| Field | Value | HW2.5 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | Reported only |
| SLICE | 486 | Reported only |
| HP_ID | 0 | Reported only; no HW2.5 mapping |
| CLS_A | 6 | Reported only |
| CLS_B | 0 | Reported only |

## Table HW2.5.1 — Required Summary

Every value in this table must be traceable to a UUID-labelled result in `reports/RUN_LOG.txt`.

| Measurement | Your GPU | Notes |
|---|---|---|
| Peak achieved TFLOPS (BF16) | To be measured | UUID-labelled run |
| % of theoretical peak (BF16) | To be measured | UUID-labelled run |
| Effective bandwidth (GB/s) | To be measured | UUID-labelled run |
| Naive attention OOM length | To be measured | Smallest tested failure/largest tested success |
| Fused attention OOM length | To be measured | Smallest tested failure/largest tested success |
| Steady-state / peak throughput | To be measured | Based on thermal log |
| Throttle onset (s, or none) | To be measured | Based on thermal log |

## Part A — Onboarding and Provenance

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Complete: Part A cells of `HW2.5.ipynb` |
| Execution | Pending; requires the GPU workstation (target card: RTX 4090) |
| `nvidia-smi -q` capture | To be measured — `results/system_info/nvidia_smi_full_query.txt` |

### Hardware Information

Vendor-documented values below assume the target RTX 4090, pre-filled in the `VENDOR_SPECS` dict in `HW2.5.ipynb`'s Part A section; edit that dict by hand and update this table if this is ever run on a different card (e.g. an RTX 5090).

| Field | Value | Source |
| --- | --- | --- |
| GPU name | To be measured | `nvidia-smi` (measured) |
| GPU UUID | To be measured | `nvidia-smi` (measured) |
| Driver version | To be measured | `nvidia-smi` (measured) |
| CUDA version | To be measured | `torch.version.cuda` (measured) |
| VRAM capacity | To be measured (spec: 24 GB GDDR6X) | Measured value from `nvidia-smi`; spec from vendor documentation |
| Reported power limit | To be measured (spec: 450 W reference board) | Measured value from `nvidia-smi`; spec from vendor documentation |
| Architecture | Ada Lovelace (AD102) | Vendor documentation |
| Memory type | GDDR6X | Vendor documentation |
| Memory bandwidth | 1008 GB/s (spec) | Vendor documentation |
| Tensor-core generation | 4th generation (Ada) | Vendor documentation |
| Reduced precisions supported by tensor cores | FP16, BF16, TF32, INT8, INT4, FP8 | Vendor documentation |
| Vendor documentation citation | https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/ | — |

Vendor-documented figures above come from the `VENDOR_SPECS` dict in `HW2.5.ipynb`'s Part A section, reproducing NVIDIA's published RTX 4090 reference-card specifications. None of these were re-verified against a live NVIDIA page at authoring time because outbound network access was unavailable in that environment — spot-check them on the workstation (which has internet access) before treating this table as final, and update `VENDOR_SPECS` in the notebook if any figure needs correction.

## Part B — Precision and Achieved Throughput

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Complete: Part B cells of `HW2.5.ipynb` |
| Execution | Pending; requires the GPU workstation (target card: RTX 4090) |
| Matrix sizes | 1024, 4096, 8192, 16384 |
| Precisions | FP32, TF32, FP16, BF16 |
| Repetitions/warm-up | Recorded per row in `results/precision/precision_benchmark_results.csv` (see `REPS_BY_SIZE`/`WARMUP_BY_SIZE` in the source) |
| FLOP formula | `2 * N^3` (multiply-add pair per output-element partial product) |
| Output file | `results/precision/precision_benchmark_results.csv` |
| Figure | `figures/precision_tflops_vs_matrix_size.png` |

### Results

To be filled in from `results/precision/precision_benchmark_results.csv` after execution.

| Matrix N | Precision | Mean latency (ms) | Achieved TFLOPS | % of theoretical peak | GPU UUID |
| ---: | --- | ---: | ---: | ---: | --- |
| To be measured | | | | | |

### Plateau Discussion

<!-- For each precision, state the matrix size at which achieved TFLOPS
     plateaus (stops increasing meaningfully with N) and explain, from the
     actual data, why smaller matrices fall short of peak throughput (e.g.
     kernel-launch/overhead dominating at small N, insufficient work to
     saturate all SMs, memory-bound regime at small N per the Part C
     roofline). Do not state a plateau size before the benchmark has run. -->

### Additional Lower-Precision Probe (FP8)

To be filled in from `results/precision/fp8_availability_probe.json`.

| Field | Value |
| --- | --- |
| Precision attempted | FP8 (E4M3) via `torch._scaled_mm` |
| Status | To be measured |
| Details | To be measured — copy the exact `error_message`/measurement from the JSON probe |

## Part C — Bandwidth-Bound vs. Compute-Bound

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Complete: Part C cells of `HW2.5.ipynb` |
| Execution | Pending; requires the GPU workstation (target card: RTX 4090) |
| Memory-bound operation | Elementwise addition, target `N = 400,000,000` float32 elements (adaptively halved and retried down to `N = 12,500,000` on a CUDA out-of-memory error; actual size used is recorded per row) |
| Compute-bound operation | Square FP32 matmul, `N = 8192` |
| Output file | `results/bandwidth/bandwidth_benchmark_results.csv` |

### Results

To be filled in from `results/bandwidth/bandwidth_benchmark_results.csv`.

| Operation | Category | Latency (ms) | Achieved GB/s | % of spec bandwidth | Arithmetic intensity (FLOPs/byte) | Roofline classification | GPU UUID |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| To be measured | | | | | | | |

Roofline ridge point (theoretical peak FP32 FLOPS / theoretical peak bandwidth) is computed in the `roofline_ridge_point()` function in `HW2.5.ipynb`'s Part C section, from the same `VENDOR_SPECS` constants used in Part A/B.

## Part D — Cost of Attention

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Complete: Part D cells of `HW2.5.ipynb` |
| Execution | Pending; requires the GPU workstation (target card: RTX 4090) |
| Configuration | Batch size 1, 1 attention head, head dimension 64, dtype bfloat16, `torch.inference_mode()` |
| Sequence lengths (grid) | 512, 1024, 2048, 4096, 8192, 16384; if the required grid contains no failure (expected on a 24 GB RTX 4090), the notebook automatically extends the search geometrically (doubling, up to 12 extension steps or a 120s-per-probe time budget) until it finds one, then bisects between the bracket (up to 6 refinement probes) |
| Naive implementation | From-scratch scaled dot-product attention that materializes the full `[seq_len, seq_len]` score/probability matrix |
| Fused implementation | `torch.nn.functional.scaled_dot_product_attention` restricted to the Flash/memory-efficient backends |
| Output file | `results/attention/attention_benchmark_results.csv` |
| Summary file | `results/attention/attention_summary.json` |
| Figure | `figures/attention_peak_memory_vs_seqlen.png` |

### Results

To be filled in from `results/attention/attention_benchmark_results.csv`.

| Implementation | Seq len | Status | Latency (ms) | Peak memory (MB) | GPU UUID |
| --- | ---: | --- | ---: | ---: | --- |
| To be measured | | | | | |

### OOM Boundaries

| Implementation | Largest tested success | Smallest tested failure | Notes |
| --- | ---: | ---: | --- |
| Naive | To be measured | To be measured | Bracket only; no exact single-token boundary is claimed |
| Fused | To be measured | To be measured | Bracket only; no exact single-token boundary is claimed |

### Measured Memory Curve Fit (naive implementation)

`peak_memory_mb ≈ a·seq_len² + b·seq_len + c`, fit with `numpy.polyfit` in the analysis cell right after the Part D benchmark loop in `HW2.5.ipynb`.

| Coefficient | Value |
| --- | ---: |
| `a` (quadratic, MB per seq_len²) | To be measured |
| `b` (linear, MB per seq_len) | To be measured |
| `c` (intercept, MB) | To be measured |

### Speedup (fused vs. naive, where both succeeded)

To be filled in from `attention_summary.json`'s `speedups_fused_over_naive`.

| Seq len | Naive latency (ms) | Fused latency (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| To be measured | | | |

### What the Fused Kernel Avoids Materializing

The fused scaled-dot-product-attention backend (Flash/memory-efficient attention) never materializes the full `[seq_len, seq_len]` attention probability matrix in GPU memory. It processes queries, keys, and values in tiles, computing partial attention scores and an online (running) softmax so only small per-tile score buffers and running output/normalization statistics stay resident. Because the `O(seq_len²)` score and probability matrices — and the extra GPU-memory read/write traffic they require — are avoided entirely, the fused kernel's peak memory scales close to `O(seq_len)` instead of `O(seq_len²)`, which is why it keeps succeeding at sequence lengths where the naive implementation runs out of memory.

## Part E — Sustained Load and Thermal Behaviour

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Complete: Part E cells of `HW2.5.ipynb` |
| Execution | Pending; requires the GPU workstation (target card: RTX 4090), ~20 minutes |
| Sampling interval | 5 seconds |
| Fields logged per sample | elapsed time, GPU clock, memory clock, temperature, power draw, GPU utilization, GPU UUID, cumulative matmuls, interval throughput |
| Output file | `results/thermal/thermal_log.csv` (flushed continuously) |
| Analysis file | `results/thermal/thermal_analysis.json` |
| Figure | `figures/thermal_clock_temperature_vs_time.png` |

### Thermal Findings

To be filled in from `results/thermal/thermal_analysis.json` after execution. Do not state any of the following before the test has actually run.

| Field | Value |
| --- | --- |
| Throttling occurred? | To be measured |
| Temperature ceiling at throttle onset (`temperature_c_at_throttle_onset`) | To be measured |
| Power ceiling at throttle onset (`power_draw_w_at_throttle_onset`) | To be measured |
| Throttle onset time (s, or "none") | To be measured |
| Peak throughput, first 30 s (matmuls/s) | To be measured |
| Steady-state throughput, final 5 min (matmuls/s) | To be measured |
| Steady-state as % of peak throughput | To be measured |

Note: `temperature_c_at_throttle_onset`/`power_draw_w_at_throttle_onset` are the sensor readings at the detected onset sample -- the ceiling the assignment asks about -- not the run's overall maximum temperature/power (also available in `thermal_analysis.json` as `max_temperature_c_observed`/`max_power_draw_w_observed` for context).

## Part F — Traceability

Every "To be measured" value in this file must be filled in with a number that also appears, tagged with the same GPU UUID, timestamp, and command, in a `reports/RUN_LOG.txt` entry. `results/run_log_raw.txt` is auto-appended by the notebook (one UUID-labelled, timestamped line per measurement) and is the raw evidence to copy from when writing those `RUN_LOG.txt` entries.
