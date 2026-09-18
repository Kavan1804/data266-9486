"""Part D — the cost of attention.

Compares a from-scratch naive scaled dot-product attention implementation
(which materializes the full sequence-by-sequence attention matrix) against
PyTorch's fused/memory-efficient `torch.nn.functional.scaled_dot_product_attention`,
at batch size 1, one attention head, head dimension 64, across sequence
lengths 512 through 16384.

Fixed configuration (documented here, not re-derived per run):
  * batch size B = 1
  * number of attention heads = 1
  * head dimension d = 64
  * precision = bfloat16 (native Tensor Core support across the whole Ada /
    RTX 40-series lineup, RTX 4090 down to RTX 4060, and more numerically
    forgiving than float16 for the un-scaled score matrix the naive path
    materializes at long sequence lengths)
  * inference mode: all forward passes run under `torch.inference_mode()`,
    so no autograd graph or gradient buffers are allocated.

For every sequence length, CUDA peak-memory stats are reset and measurement
is synchronized before and after timing. `torch.cuda.OutOfMemoryError` is
caught per sequence length so a failure at one length does not abort the
sweep. After the fixed grid, the boundary between the largest succeeding and
smallest failing length is refined with a bounded bisection search using
additional probe points -- the exact single sequence length at which memory
runs out is never claimed, only the bracket in which it falls. On a
smaller-VRAM card (e.g. an 8 GB RTX 4060 versus a 24 GB RTX 4090) the naive
implementation's OOM boundary will simply land at a shorter sequence length --
the benchmark itself does not assume a particular GPU or VRAM size.

Run on the GPU workstation:

    python3 -m src.attention_benchmarks
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.system_info import get_gpu_uuid

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "attention"

BATCH_SIZE = 1
NUM_HEADS = 1
HEAD_DIM = 64
DTYPE_NAME = "bfloat16"

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
REPS, WARMUP = 10, 3
MAX_BOUNDARY_PROBES = 4

FUSED_KERNEL_EXPLANATION = (
    "PyTorch's fused scaled_dot_product_attention backend (Flash/mem-efficient "
    "attention) never materializes the full [seq_len, seq_len] attention "
    "probability matrix in GPU memory. It processes the query, key, and value "
    "tensors in tiles, computing partial attention scores and an online "
    "(running) softmax so only small per-tile score buffers and the running "
    "output/normalization statistics are kept resident. Because the O(seq_len^2) "
    "score and probability matrices are avoided entirely -- along with the "
    "corresponding extra read/write traffic to and from GPU memory for that "
    "matrix -- peak memory for the fused path scales close to O(seq_len) "
    "instead of O(seq_len^2), which is why it can run at sequence lengths where "
    "the naive implementation runs out of memory."
)


@dataclass
class AttentionResult:
    timestamp_utc: str
    gpu_uuid: str
    gpu_name: str
    implementation: str  # "naive" or "fused"
    dtype: str
    batch_size: int
    num_heads: int
    head_dim: int
    seq_len: int
    warmup_iters: int
    repetitions: int
    mean_latency_ms: float
    peak_memory_mb: float
    status: str  # "success" or "oom"
    error_message: str
    probe_type: str  # "grid" or "boundary_refinement"


def _make_qkv(batch, heads, seq_len, head_dim, dtype, device, torch):
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    return q, k, v


def _naive_attention(q, k, v, torch):
    """Textbook scaled dot-product attention that materializes the full
    [batch, heads, seq_len, seq_len] score/probability matrix."""
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def _fused_attention(q, k, v, torch):
    import torch.nn.functional as F  # noqa: PLC0415

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v)
    except ImportError:
        # Older PyTorch: fall back to the context-manager API, still steering
        # away from the naive math backend so this is a genuine fused-kernel run.
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ):
            return F.scaled_dot_product_attention(q, k, v)


def _run_single(
    seq_len: int,
    implementation: str,
    gpu_uuid: str,
    gpu_name: str,
    probe_type: str,
    torch,
) -> AttentionResult:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    status, error_message = "success", ""
    mean_latency_ms = float("nan")
    peak_memory_mb = float("nan")

    try:
        q, k, v = _make_qkv(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, dtype, device, torch)
        fn = _naive_attention if implementation == "naive" else _fused_attention

        with torch.inference_mode():
            for _ in range(WARMUP):
                _ = fn(q, k, v, torch)
            torch.cuda.synchronize()

            latencies_ms = []
            for _ in range(REPS):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = fn(q, k, v, torch)
                torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
        mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        del q, k, v
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        status = "oom"
        error_message = f"{type(exc).__name__}: {exc}"
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        status = "oom"
        error_message = f"{type(exc).__name__}: {exc}"
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return AttentionResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        implementation=implementation,
        dtype=DTYPE_NAME,
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        seq_len=seq_len,
        warmup_iters=WARMUP,
        repetitions=REPS,
        mean_latency_ms=mean_latency_ms,
        peak_memory_mb=peak_memory_mb,
        status=status,
        error_message=error_message,
        probe_type=probe_type,
    )


def _refine_oom_boundary(rows: list[AttentionResult], implementation, gpu_uuid, gpu_name, torch):
    """Bisect between the largest tested success and smallest tested failure.

    Only runs when the fixed grid actually contains both a success and a
    failure for this implementation; adds at most MAX_BOUNDARY_PROBES extra
    rows. This narrows the OOM bracket without ever asserting a single exact
    failing sequence length.
    """
    successes = sorted(r.seq_len for r in rows if r.status == "success")
    failures = sorted(r.seq_len for r in rows if r.status == "oom")
    if not successes or not failures:
        return []
    low = max(successes)
    high = min(failures)
    if low >= high:
        return []

    refined_rows = []
    for _ in range(MAX_BOUNDARY_PROBES):
        if high - low <= 64:
            break
        midpoint = ((low + high) // 2 // 64) * 64  # round to a multiple of 64
        if midpoint <= low or midpoint >= high:
            break
        result = _run_single(midpoint, implementation, gpu_uuid, gpu_name, "boundary_refinement", torch)
        refined_rows.append(result)
        if result.status == "success":
            low = midpoint
        else:
            high = midpoint
    return refined_rows


def fit_memory_curve(rows: list[AttentionResult]) -> dict:
    """Fit peak memory (MB) vs seq_len with a quadratic polynomial on the
    successful naive-implementation runs and return the measured coefficients.
    """
    import numpy as np  # noqa: PLC0415

    naive_success = sorted(
        ((r.seq_len, r.peak_memory_mb) for r in rows if r.implementation == "naive" and r.status == "success"),
        key=lambda pair: pair[0],
    )
    if len(naive_success) < 3:
        return {
            "status": "insufficient_data",
            "note": "Need at least 3 successful naive-attention runs to fit a quadratic curve.",
        }

    xs = np.array([p[0] for p in naive_success], dtype=float)
    ys = np.array([p[1] for p in naive_success], dtype=float)
    coeffs = np.polyfit(xs, ys, deg=2)  # [a, b, c] for a*x^2 + b*x + c
    return {
        "status": "fit",
        "degree": 2,
        "quadratic_coefficient_a_mb_per_seqlen_sq": float(coeffs[0]),
        "linear_coefficient_b_mb_per_seqlen": float(coeffs[1]),
        "intercept_c_mb": float(coeffs[2]),
        "fitted_on_seq_lengths": [p[0] for p in naive_success],
        "note": "peak_memory_mb ~= a * seq_len^2 + b * seq_len + c, fit with numpy.polyfit.",
    }


def compute_speedups(rows: list[AttentionResult]) -> list[dict]:
    """Fused-vs-naive speedup at every sequence length where both succeeded."""
    naive_by_len = {r.seq_len: r for r in rows if r.implementation == "naive" and r.status == "success"}
    fused_by_len = {r.seq_len: r for r in rows if r.implementation == "fused" and r.status == "success"}
    speedups = []
    for seq_len in sorted(set(naive_by_len) & set(fused_by_len)):
        naive_ms = naive_by_len[seq_len].mean_latency_ms
        fused_ms = fused_by_len[seq_len].mean_latency_ms
        speedups.append(
            {
                "seq_len": seq_len,
                "naive_latency_ms": naive_ms,
                "fused_latency_ms": fused_ms,
                "speedup_naive_over_fused": naive_ms / fused_ms if fused_ms else float("nan"),
            }
        )
    return speedups


def _oom_boundary_summary(rows: list[AttentionResult], implementation: str) -> dict:
    subset = [r for r in rows if r.implementation == implementation]
    successes = sorted(r.seq_len for r in subset if r.status == "success")
    failures = sorted(r.seq_len for r in subset if r.status == "oom")
    return {
        "implementation": implementation,
        "largest_tested_success": max(successes) if successes else None,
        "smallest_tested_failure": min(failures) if failures else None,
        "note": (
            "This brackets the out-of-memory boundary between the largest "
            "sequence length that succeeded and the smallest one that failed; "
            "it does not claim an exact single-token failure point."
        ),
    }


def run_all(output_csv: Path | None = None) -> None:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this benchmark on the GPU workstation, "
            "not on the development machine."
        )

    output_csv = output_csv or (RESULTS_DIR / "attention_benchmark_results.csv")
    gpu_uuid = get_gpu_uuid()
    gpu_name = torch.cuda.get_device_name(0)

    all_rows: list[AttentionResult] = []
    for implementation in ("naive", "fused"):
        grid_rows = []
        for seq_len in SEQ_LENGTHS:
            print(f"Running implementation={implementation} seq_len={seq_len} ...")
            result = _run_single(seq_len, implementation, gpu_uuid, gpu_name, "grid", torch)
            print(f"  status={result.status} peak_memory_mb={result.peak_memory_mb:.1f} "
                  f"latency_ms={result.mean_latency_ms}")
            grid_rows.append(result)
        all_rows.extend(grid_rows)

        refined = _refine_oom_boundary(grid_rows, implementation, gpu_uuid, gpu_name, torch)
        for r in refined:
            print(f"  [boundary refinement] seq_len={r.seq_len} status={r.status} "
                  f"peak_memory_mb={r.peak_memory_mb:.1f}")
        all_rows.extend(refined)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))
    print(f"\nSaved {len(all_rows)} rows to {output_csv}")

    summary = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_uuid": gpu_uuid,
        "oom_boundaries": [
            _oom_boundary_summary(all_rows, "naive"),
            _oom_boundary_summary(all_rows, "fused"),
        ],
        "memory_curve_fit_naive": fit_memory_curve(all_rows),
        "speedups_fused_over_naive": compute_speedups(all_rows),
        "fused_kernel_explanation": FUSED_KERNEL_EXPLANATION,
    }
    summary_path = RESULTS_DIR / "attention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved analysis summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()
    run_all(args.output_csv)


if __name__ == "__main__":
    main()
