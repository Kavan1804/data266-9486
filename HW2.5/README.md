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

Everything for this assignment lives in one notebook, **`HW2.5.ipynb`** — no separate `.py` modules or shell scripts. This run targets the assignment's specified GPU Lab **RTX 4090** workstation. The vendor-spec cell near the top of the notebook is a single dictionary you edit by hand to match whichever card you actually use (e.g. an RTX 5090).

The notebook covers:

- **Part A** — onboarding and provenance: full `nvidia-smi -q` capture, measured hardware facts, and a vendor-documented spec dictionary kept clearly separate from anything measured.
- **Part B** — dense matmul throughput (FP32/TF32/FP16/BF16) at N = 1024, 4096, 8192, 16384, achieved TFLOPS and percent of theoretical peak, plus an FP8 availability probe.
- **Part C** — a memory-bound elementwise-add benchmark and a compute-bound matmul benchmark, with arithmetic intensity and a roofline classification.
- **Part D** — naive vs. fused (`torch.nn.functional.scaled_dot_product_attention`) attention at sequence lengths 512–16384, peak memory, OOM handling, and a measured quadratic memory-growth fit.
- **Part E** — a 20-minute sustained load with clock/temperature/power logging every 5 seconds, for throttling analysis.
- **Part F** — pointers to fill in the required summary table (`reports/METRICS.md`) and experiment log (`reports/RUN_LOG.txt`).

This repository was assembled on a Mac with no NVIDIA GPU. Every GPU-dependent result is left empty or explicitly marked "to be measured on the GPU workstation" until the notebook is actually executed there — no benchmark numbers, GPU UUIDs, or thermal conclusions in this repository are invented.

## One-Time Setup (development machine, e.g., your Mac)

Safe to run without a GPU: it clones the repo and lets you read the notebook, but does not execute any CUDA cell.

```bash
git clone <this-repository-url>
cd data266-9486/HW2.5
jupyter notebook HW2.5.ipynb
```

The notebook's first cell (`%pip install -q torch numpy pandas matplotlib`) installs everything it needs — there is no separate `requirements.txt`/`pip install -r` step. Running that cell here installs a CPU-only `torch`, which is fine for review.

## Running the Assignment — GPU Workstation ONLY

Everything below requires an NVIDIA GPU and driver. Do not attempt it on the development machine.

### 1. Clone the repository on the workstation

```bash
git clone <this-repository-url>
cd data266-9486/HW2.5
```

### 2. Verify the GPU is visible

```bash
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
```

This must report your GPU (e.g. "NVIDIA GeForce RTX 4090") before continuing.

### 3. Run the notebook top to bottom

```bash
jupyter notebook HW2.5.ipynb
```

Run every cell in order:

- The **first cell** (`%pip install -q torch numpy pandas matplotlib`) installs everything the notebook needs — no separate environment setup step.
- **Part A** cells capture `nvidia-smi -q` and measured hardware facts into `results/system_info/`. Edit the `VENDOR_SPECS` dict in the cell right after to match whatever card `measured_hardware_info['name']` actually reports (it's pre-filled for an RTX 4090), citing NVIDIA's official spec page.
- **Part B–D** cells run automatically and save CSVs under `results/` plus figures under `figures/`.
- **Part E**'s sustained-load cell blocks for ~20 minutes — let it run to completion for the real submission, then the following cells analyze and plot the log.

### 4. Update the reports

The notebook auto-appends every measurement, UUID-labelled and timestamped, to `results/run_log_raw.txt` as it runs — use that as your source of truth alongside the values printed in each cell and saved under `results/`:

- `reports/METRICS.md` — fill in Table HW2.5.1 and the per-part sections.
- `reports/RUN_LOG.txt` — append one entry per run using the template already in the file.
- `provenance/reservation_record.md` and `provenance/gpu_hours.md` — record the actual reservation and GPU-hours used.

### 5. Commit and tag the completed assignment

Run from wherever you finished updating the reports (workstation or after syncing back to your development machine):

```bash
git add HW2.5/
git commit -m "Add executed RTX 4090 results for HW2.5"
git push
git tag hw2-5
git push origin hw2-5
```

Do not create the `hw2-5` tag until the benchmarks have actually been executed on the GPU workstation and the reports reflect real measured results.

## Repository Structure

```
HW2.5/
├── README.md
├── HW2.5.ipynb              # every Part A-F cell, self-contained; first cell installs dependencies
├── results/                 # system_info/, precision/, bandwidth/, attention/, thermal/ — empty until executed
├── figures/                 # generated PNGs — empty until executed
├── reports/METRICS.md, RUN_LOG.txt
├── provenance/reservation_record.md, gpu_hours.md
└── AI_USE.md
```

## Reproducibility and Honesty Notes

- Every CSV/JSON row the notebook produces carries the real GPU UUID returned by `nvidia-smi` at run time; it is never hardcoded.
- Vendor-documented GPU specifications (architecture, memory type/bandwidth, tensor-core generation, supported reduced precisions, theoretical peak TFLOPS per precision) live in one editable `VENDOR_SPECS` dict in the notebook's Part A section, clearly separated from anything measured. Its citation explains how it was sourced — the pre-filled RTX 4090 figures were not re-verified against a live NVIDIA page because outbound network access was unavailable while authoring this notebook. Confirm them on the workstation (which has internet access) before treating Part B's "% of theoretical peak" column as final.
- No benchmark result, GPU UUID, temperature, power draw, or OOM boundary in this repository is invented. Anything GPU-dependent is either absent or explicitly marked "to be measured on the GPU workstation" until the notebook is actually run there.

## Files Intentionally Excluded from Git

- Python virtual environments (`.venv/`)
- Large raw thermal logs beyond the committed run, if multiple attempts are made
- Any `.ipynb_checkpoints/` or `__pycache__/` directories
