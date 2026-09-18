"""Part B — precision and achieved throughput.

Dense square matrix multiplication (N x N) @ (N x N) on a single CUDA device,
measured at N in {1024, 4096, 8192, 16384} for FP32, TF32, FP16, and BF16.

Every measurement:
  * uses CUDA tensors created directly on the device (no host round-trip);
  * runs a warm-up pass before timing;
  * synchronizes CUDA immediately before and after the timed region;
  * repeats for a recorded number of repetitions;
  * reports latency and achieved TFLOPS using the standard dense matmul FLOP
    count (2 * N^3, one multiply and one add per output-element partial
    product);
  * reports achieved TFLOPS as a percentage of the connected GPU's
    vendor-documented theoretical peak for that precision, auto-detected via
    `src.system_info.get_active_vendor_specs()` (currently supports the RTX
    4090 and RTX 4060 -- see `KNOWN_GPU_VENDOR_SPECS` in `src/system_info.py`);
  * is tagged with the real GPU UUID queried from the driver at run time;
  * catches a CUDA out-of-memory error per (N, precision) combination instead
    of aborting the whole sweep, which matters most on smaller-VRAM cards
    (e.g. an 8 GB RTX 4060) at N = 16384.

Run on the GPU workstation:

    python3 -m src.precision_benchmarks
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.system_info import get_active_vendor_specs, get_gpu_uuid

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "precision" / "precision_benchmark_results.csv"
DEFAULT_FP8_LOG = REPO_ROOT / "results" / "precision" / "fp8_availability_probe.json"

MATRIX_SIZES = [1024, 4096, 8192, 16384]
PRECISIONS = ["fp32", "tf32", "fp16", "bf16"]

# Repetitions and warm-up scaled down for the largest matrices so the sweep
# still finishes in a reasonable wall-clock time; the exact counts actually
# used are recorded in every output row (never assume a fixed count later).
REPS_BY_SIZE = {1024: 50, 4096: 30, 8192: 15, 16384: 8}
WARMUP_BY_SIZE = {1024: 10, 4096: 8, 8192: 5, 16384: 3}


@dataclass
class PrecisionResult:
    timestamp_utc: str
    gpu_uuid: str
    gpu_name: str
    precision: str
    dtype: str
    matrix_n: int
    warmup_iters: int
    repetitions: int
    status: str  # "success" or "oom"
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    achieved_tflops: float
    theoretical_peak_tflops: float
    pct_of_theoretical_peak: float
    error_message: str = ""
    flop_formula: str = "2 * N^3 (one multiply-add pair per output-element partial product)"


def _configure_backend(precision: str, torch) -> None:
    """Select the matmul math mode for the requested precision.

    FP32 and TF32 both use float32-resident tensors; the distinction is
    whether TensorFloat-32 reduced-precision Tensor Core math is enabled for
    the matmul, which is a global backend switch in PyTorch rather than a
    tensor dtype.
    """
    is_tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = is_tf32
    torch.backends.cudnn.allow_tf32 = is_tf32


def _dtype_for(precision: str, torch):
    return {
        "fp32": torch.float32,
        "tf32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def benchmark_matmul(n: int, precision: str, gpu_uuid: str, gpu_name: str, vendor_specs: dict) -> PrecisionResult:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This benchmark must be run on the GPU workstation."
        )

    _configure_backend(precision, torch)
    dtype = _dtype_for(precision, torch)
    reps = REPS_BY_SIZE[n]
    warmup = WARMUP_BY_SIZE[n]
    peak_tflops = vendor_specs["theoretical_peak_tflops_dense"][precision]

    device = torch.device("cuda")
    torch.cuda.empty_cache()

    try:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)

        for _ in range(warmup):
            _ = a @ b
        torch.cuda.synchronize()

        latencies_ms = []
        for _ in range(reps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = a @ b
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

        mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
        flops = 2 * (n ** 3)
        achieved_tflops = flops / (mean_latency_ms / 1000.0) / 1e12
        pct_of_peak = (achieved_tflops / peak_tflops) * 100.0 if peak_tflops else float("nan")

        result = PrecisionResult(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            precision=precision,
            dtype=str(dtype),
            matrix_n=n,
            warmup_iters=warmup,
            repetitions=reps,
            status="success",
            mean_latency_ms=mean_latency_ms,
            min_latency_ms=min(latencies_ms),
            max_latency_ms=max(latencies_ms),
            achieved_tflops=achieved_tflops,
            theoretical_peak_tflops=peak_tflops,
            pct_of_theoretical_peak=pct_of_peak,
        )
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        result = PrecisionResult(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            precision=precision,
            dtype=str(dtype),
            matrix_n=n,
            warmup_iters=warmup,
            repetitions=reps,
            status="oom",
            mean_latency_ms=float("nan"),
            min_latency_ms=float("nan"),
            max_latency_ms=float("nan"),
            achieved_tflops=float("nan"),
            theoretical_peak_tflops=peak_tflops,
            pct_of_theoretical_peak=float("nan"),
            error_message=f"{type(exc).__name__}: {exc}",
        )
    finally:
        torch.cuda.empty_cache()

    return result


def attempt_lower_precision_probe(n: int = 4096, reps: int = 20, warmup: int = 5) -> dict:
    """Part B (additional lower precision) — probe FP8 support in the installed stack.

    The RTX 40-series' 4th-generation Tensor Cores (RTX 4090, RTX 4060, etc.)
    support FP8 (E4M3/E5M2), but
    PyTorch exposes FP8 GEMM support only through `torch._scaled_mm`, which is
    version- and build-dependent. This function attempts a real FP8 matmul and
    returns either genuine measurements or the exact unavailability/failure
    message raised by the installed stack -- it never fabricates a result.
    """
    attempted_api = "torch._scaled_mm with torch.float8_e4m3fn inputs"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "precision_attempted": "fp8_e4m3",
        "api_attempted": attempted_api,
        "matrix_n": n,
    }
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError(
                "This PyTorch build has no torch.float8_e4m3fn dtype; FP8 is not "
                "exposed by the installed software stack."
            )
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError(
                "This PyTorch build has no torch._scaled_mm; FP8 scaled GEMM is not "
                "exposed by the installed software stack."
            )

        device = torch.device("cuda")
        a = torch.randn(n, n, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)

        for _ in range(warmup):
            _ = torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(reps):
            _ = torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / reps

        flops = 2 * (n ** 3)
        achieved_tflops = flops / (elapsed_ms / 1000.0) / 1e12

        record.update(
            {
                "status": "available",
                "gpu_uuid": get_gpu_uuid(),
                "repetitions": reps,
                "warmup_iters": warmup,
                "mean_latency_ms": elapsed_ms,
                "achieved_tflops": achieved_tflops,
                "note": (
                    "Theoretical FP8 peak TFLOPS is intentionally not included in "
                    "KNOWN_GPU_VENDOR_SPECS, so no percent-of-peak figure is computed "
                    "for this exploratory precision."
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - we want to record any failure verbatim
        record.update(
            {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
    return record


def run_all(output_csv: Path = DEFAULT_OUTPUT_CSV, fp8_log: Path = DEFAULT_FP8_LOG) -> None:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this benchmark on the GPU workstation, "
            "not on the development machine."
        )

    gpu_uuid = get_gpu_uuid()
    gpu_name = torch.cuda.get_device_name(0)
    vendor_specs = get_active_vendor_specs()
    print(f"Detected GPU: {gpu_name}  (vendor specs matched: {vendor_specs['gpu_name']})")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in MATRIX_SIZES:
        for precision in PRECISIONS:
            print(f"Running N={n} precision={precision} ...")
            result = benchmark_matmul(n, precision, gpu_uuid, gpu_name, vendor_specs)
            rows.append(result)
            if result.status == "success":
                print(
                    f"  latency={result.mean_latency_ms:.3f} ms  "
                    f"tflops={result.achieved_tflops:.2f}  "
                    f"pct_peak={result.pct_of_theoretical_peak:.1f}%"
                )
            else:
                print(f"  OOM at N={n} precision={precision}: {result.error_message}")

    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nSaved {len(rows)} rows to {output_csv}")

    fp8_log.parent.mkdir(parents=True, exist_ok=True)
    import json  # noqa: PLC0415

    fp8_result = attempt_lower_precision_probe()
    fp8_log.write_text(json.dumps(fp8_result, indent=2) + "\n")
    print(f"Saved FP8 availability probe ({fp8_result['status']}) to {fp8_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--fp8-log", type=Path, default=DEFAULT_FP8_LOG)
    args = parser.parse_args()
    run_all(args.output_csv, args.fp8_log)


if __name__ == "__main__":
    main()
