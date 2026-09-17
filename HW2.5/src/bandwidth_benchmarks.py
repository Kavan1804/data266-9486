"""Part C — bandwidth-bound versus compute-bound.

Two operations are benchmarked on the same device:

  1. A memory-bound operation: large elementwise addition `C = A + B` on 1-D
     float32 tensors. Effective bandwidth is computed from the bytes moved
     (read A, read B, write C) and reported as a percentage of the RTX 4090's
     vendor-documented memory bandwidth.
  2. A compute-bound operation: a large dense square FP32 matrix
     multiplication, reusing the same timing methodology as Part B.

For both operations, arithmetic intensity (FLOPs / byte) is computed and
compared against the RTX 4090's roofline ridge point (theoretical peak FLOPS /
theoretical peak bandwidth) to classify the operation as bandwidth-bound or
compute-bound. Raw measurements and derived values are saved to CSV with the
real GPU UUID attached to every row.

Run on the RTX 4090 workstation:

    python3 -m src.bandwidth_benchmarks
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.system_info import VENDOR_SPECS_RTX_4090, get_gpu_uuid

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "bandwidth" / "bandwidth_benchmark_results.csv"

# ~1.5 GiB per tensor at float32 -> large enough to be bandwidth-limited and
# not fit in the RTX 4090's L2 cache.
ELEMENTWISE_ADD_NUM_ELEMENTS = 400_000_000
COMPUTE_BOUND_MATRIX_N = 8192

ADD_REPS, ADD_WARMUP = 20, 5
MATMUL_REPS, MATMUL_WARMUP = 15, 5

BYTES_PER_FLOAT32 = 4


@dataclass
class BandwidthResult:
    timestamp_utc: str
    gpu_uuid: str
    gpu_name: str
    operation: str
    category: str  # "memory_bound" or "compute_bound"
    dtype: str
    problem_size: str
    warmup_iters: int
    repetitions: int
    mean_latency_ms: float
    bytes_moved: int
    flops: int
    achieved_gbps: float
    pct_of_spec_bandwidth: float
    arithmetic_intensity_flops_per_byte: float
    roofline_ridge_point_flops_per_byte: float
    roofline_classification: str


def _roofline_ridge_point() -> float:
    peak_flops = VENDOR_SPECS_RTX_4090["theoretical_peak_tflops_dense"]["fp32"] * 1e12
    peak_bytes_per_s = VENDOR_SPECS_RTX_4090["memory_bandwidth_gbps_spec"] * 1e9
    return peak_flops / peak_bytes_per_s


def _classify(intensity: float, ridge_point: float) -> str:
    return "compute_bound" if intensity >= ridge_point else "bandwidth_bound"


def benchmark_elementwise_add(gpu_uuid: str, gpu_name: str) -> BandwidthResult:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on the RTX 4090 workstation.")

    device = torch.device("cuda")
    n = ELEMENTWISE_ADD_NUM_ELEMENTS
    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)

    for _ in range(ADD_WARMUP):
        c = a + b
    torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(ADD_REPS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        c = a + b
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
    bytes_moved = 3 * n * BYTES_PER_FLOAT32  # read a, read b, write c
    flops = n  # one add per element
    achieved_gbps = (bytes_moved / (mean_latency_ms / 1000.0)) / 1e9
    pct_of_spec = (achieved_gbps / VENDOR_SPECS_RTX_4090["memory_bandwidth_gbps_spec"]) * 100.0
    intensity = flops / bytes_moved
    ridge = _roofline_ridge_point()

    del a, b, c
    torch.cuda.empty_cache()

    return BandwidthResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        operation="elementwise_add",
        category="memory_bound",
        dtype="torch.float32",
        problem_size=f"N={n}",
        warmup_iters=ADD_WARMUP,
        repetitions=ADD_REPS,
        mean_latency_ms=mean_latency_ms,
        bytes_moved=bytes_moved,
        flops=flops,
        achieved_gbps=achieved_gbps,
        pct_of_spec_bandwidth=pct_of_spec,
        arithmetic_intensity_flops_per_byte=intensity,
        roofline_ridge_point_flops_per_byte=ridge,
        roofline_classification=_classify(intensity, ridge),
    )


def benchmark_compute_bound_matmul(gpu_uuid: str, gpu_name: str) -> BandwidthResult:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on the RTX 4090 workstation.")

    torch.backends.cuda.matmul.allow_tf32 = False  # keep this a clean FP32 reference point
    device = torch.device("cuda")
    n = COMPUTE_BOUND_MATRIX_N
    a = torch.randn(n, n, device=device, dtype=torch.float32)
    b = torch.randn(n, n, device=device, dtype=torch.float32)

    for _ in range(MATMUL_WARMUP):
        _ = a @ b
    torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(MATMUL_REPS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = a @ b
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
    flops = 2 * (n ** 3)
    bytes_moved = 3 * (n ** 2) * BYTES_PER_FLOAT32  # read A, read B, write C (naive lower bound)
    achieved_gbps = (bytes_moved / (mean_latency_ms / 1000.0)) / 1e9
    pct_of_spec = (achieved_gbps / VENDOR_SPECS_RTX_4090["memory_bandwidth_gbps_spec"]) * 100.0
    intensity = flops / bytes_moved
    ridge = _roofline_ridge_point()

    del a, b
    torch.cuda.empty_cache()

    return BandwidthResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        operation="square_matmul",
        category="compute_bound",
        dtype="torch.float32",
        problem_size=f"N={n}",
        warmup_iters=MATMUL_WARMUP,
        repetitions=MATMUL_REPS,
        mean_latency_ms=mean_latency_ms,
        bytes_moved=bytes_moved,
        flops=flops,
        achieved_gbps=achieved_gbps,
        pct_of_spec_bandwidth=pct_of_spec,
        arithmetic_intensity_flops_per_byte=intensity,
        roofline_ridge_point_flops_per_byte=ridge,
        roofline_classification=_classify(intensity, ridge),
    )


def run_all(output_csv: Path = DEFAULT_OUTPUT_CSV) -> None:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this benchmark on the RTX 4090 GPU lab "
            "workstation, not on the development machine."
        )

    gpu_uuid = get_gpu_uuid()
    gpu_name = torch.cuda.get_device_name(0)

    print("Running memory-bound elementwise addition ...")
    add_result = benchmark_elementwise_add(gpu_uuid, gpu_name)
    print(
        f"  latency={add_result.mean_latency_ms:.3f} ms  "
        f"bandwidth={add_result.achieved_gbps:.1f} GB/s  "
        f"pct_spec={add_result.pct_of_spec_bandwidth:.1f}%  "
        f"classification={add_result.roofline_classification}"
    )

    print("Running compute-bound square matmul ...")
    matmul_result = benchmark_compute_bound_matmul(gpu_uuid, gpu_name)
    print(
        f"  latency={matmul_result.mean_latency_ms:.3f} ms  "
        f"achieved_bandwidth={matmul_result.achieved_gbps:.1f} GB/s  "
        f"intensity={matmul_result.arithmetic_intensity_flops_per_byte:.2f} FLOPs/byte  "
        f"classification={matmul_result.roofline_classification}"
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [add_result, matmul_result]
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nSaved {len(rows)} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()
    run_all(args.output_csv)


if __name__ == "__main__":
    main()
