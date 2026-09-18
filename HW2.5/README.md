# DATA 266 HW2.5 — GPU Assignment I: Precision, Bandwidth, and the Cost of Attention

## Personal Parameters

| Field | Value | HW2.5 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | Reported only; this assignment measures hardware, not model training |
| SLICE | 486 | Reported only |
| HP_ID | 0 | Reported only; HW2.5 defines no HP_ID mapping |
| CLS_A | 6 | Reported only |
| CLS_B | 0 | Reported only |

## Scope

HW2.5 benchmarks a single NVIDIA GPU on a workstation. The assignment was originally scoped around a lab RTX 4090, but this repository targets an **RTX 4060** (8 GB) instead — every module auto-detects the connected GPU via `nvidia-smi` and looks up the matching vendor-documented specs (see `KNOWN_GPU_VENDOR_SPECS` in `src/system_info.py`, which also still supports the RTX 4090 if you ever run this on that card):

- **Part A** — onboarding and provenance: reservation/GPU-hour records, full `nvidia-smi -q` capture, and separating measured hardware facts from vendor-documented specifications.
- **Part B** — dense matmul throughput (FP32/TF32/FP16/BF16) at N = 1024, 4096, 8192, 16384, achieved TFLOPS and percent of theoretical peak, plus an FP8 availability probe.
- **Part C** — a memory-bound elementwise-add benchmark and a compute-bound matmul benchmark, with arithmetic intensity and a roofline classification.
- **Part D** — naive vs. fused (`torch.nn.functional.scaled_dot_product_attention`) attention at sequence lengths 512–16384, peak memory, OOM handling, and a measured quadratic memory-growth fit.
- **Part E** — a 20-minute sustained load with clock/temperature/power logging every 5 seconds, for throttling analysis.
- **Part F** — the required summary table (`reports/METRICS.md`) and experiment log (`reports/RUN_LOG.txt`).

This repository was assembled on a Mac with no NVIDIA GPU. Every GPU-dependent result is left empty or explicitly marked "to be measured on the GPU workstation" until the code below is actually executed there — no benchmark numbers, GPU UUIDs, or thermal conclusions in this repository are invented.

## One-Time Setup (development machine, e.g., your Mac)

Everything here is safe to run without a GPU: it clones the repo and reviews the code/notebook, but does not execute any CUDA benchmark.

```bash
git clone <this-repository-url>
cd data266-9486/HW2.5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # torch install here will be CPU-only; that's fine for review
```

## Running the Assignment — GPU Workstation ONLY

The commands in this section require an NVIDIA GPU (this run targets an RTX 4060; the RTX 4090 is also supported), the NVIDIA driver, and CUDA-enabled PyTorch. Do not attempt them on the development machine.

### 1. Clone and set up the environment on the workstation

```bash
git clone <this-repository-url>
cd data266-9486/HW2.5
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match the workstation's CUDA version
pip install -r requirements.txt
```

### 2. Verify the GPU is visible

```bash
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Both commands must report your GPU (e.g. "NVIDIA GeForce RTX 4060") before continuing. `src/system_info.get_active_vendor_specs()` reads this same name to pick the correct vendor spec entry automatically — if it reports a card that is not yet in `KNOWN_GPU_VENDOR_SPECS` (currently RTX 4090 and RTX 4060), add an entry there before running Parts B/C.

### 3. Part A — capture `nvidia-smi -q` and hardware info

```bash
./scripts/capture_gpu_info.sh
```

This writes `results/system_info/nvidia_smi_full_query.txt`, `results/system_info/measured_hardware_info.json`, and a copy of the vendor-documented spec constants for this run. Record the reported GPU UUID and session details in `provenance/reservation_record.md` and `reports/RUN_LOG.txt`.

### 4. Parts B–D — precision, bandwidth, and attention benchmarks

```bash
./scripts/run_benchmarks.sh
```

This runs `src/precision_benchmarks.py`, `src/bandwidth_benchmarks.py`, and `src/attention_benchmarks.py` in sequence and generates the Part B and Part D figures.

### 5. Part E — 20-minute sustained load and thermal log

```bash
./scripts/run_thermal_test.sh          # defaults to 1200 seconds
./scripts/run_thermal_test.sh 1200     # explicit duration in seconds
```

The CSV log at `results/thermal/thermal_log.csv` is flushed continuously, so it remains usable even if the run is interrupted. This also runs the throttling/throughput analysis and generates the Part E figure.

### 6. Generate any remaining figures

```bash
python3 -m src.plotting --all
```

(`run_benchmarks.sh` and `run_thermal_test.sh` already call `plotting` for the figures they produce; this is only needed if you re-run an individual benchmark module by hand.)

### 7. Update the reports

Open `notebooks/HW2.5_GPU_Benchmark.ipynb` and run it top to bottom to walk through Parts A–E against the actual saved results, then update from the observed numbers only:

- `reports/METRICS.md` — fill in Table HW2.5.1 and the per-part sections.
- `reports/RUN_LOG.txt` — append one entry per run using the template already in the file.
- `provenance/reservation_record.md` and `provenance/gpu_hours.md` — record the actual reservation and GPU-hours used.

### 8. Commit and tag the completed assignment

Run from wherever you finished updating the reports (workstation or after syncing back to your development machine):

```bash
git add HW2.5/
git commit -m "Add executed RTX 4060 results for HW2.5"
git push
git tag hw2-5
git push origin hw2-5
```

Do not create the `hw2-5` tag until the benchmarks have actually been executed on the GPU workstation and the reports reflect real measured results.

## Repository Structure

```
HW2.5/
├── README.md
├── requirements.txt
├── notebooks/HW2.5_GPU_Benchmark.ipynb
├── src/                     # system_info, precision/bandwidth/attention/thermal benchmarks, plotting
├── scripts/                 # capture_gpu_info.sh, run_benchmarks.sh, run_thermal_test.sh
├── results/                 # system_info/, precision/, bandwidth/, attention/, thermal/ — empty until executed
├── figures/                 # generated PNGs — empty until executed
├── reports/METRICS.md, RUN_LOG.txt
├── provenance/reservation_record.md, gpu_hours.md
└── AI_USE.md
```

## Reproducibility and Honesty Notes

- Every CSV/JSON row produced by `src/` carries the real GPU UUID returned by `nvidia-smi` at run time; it is never hardcoded.
- Vendor-documented GPU specifications (architecture, memory type/bandwidth, tensor-core generation, supported reduced precisions, theoretical peak TFLOPS per precision) live in one place per supported card, `KNOWN_GPU_VENDOR_SPECS` in `src/system_info.py` (RTX 4090 and RTX 4060), clearly separated from anything measured. `get_active_vendor_specs()` auto-detects which card is attached and every benchmark module reads from it, so "% of theoretical peak" always compares against the right card's numbers. Each entry's citation and verification note explain how it was sourced and what should be double-checked on the workstation (which has internet access, unlike the environment this repository was assembled/updated in) — the RTX 4060 entry's Tensor Core figures in particular were derived rather than read off a single vendor table, and need the closest look.
- No benchmark result, GPU UUID, temperature, power draw, or OOM boundary in this repository is invented. Anything GPU-dependent is either absent or explicitly marked "to be measured on the GPU workstation" until the scripts above are actually run there.

## Files Intentionally Excluded from Git

- Python virtual environments (`.venv/`)
- Large raw thermal logs beyond the committed run, if multiple attempts are made
- Any `__pycache__/` directories
