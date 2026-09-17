"""Part A — onboarding and provenance: GPU identification and `nvidia-smi -q` capture.

This module is the single place where MEASURED hardware facts (queried live from
the workstation's driver/runtime) and VENDOR-DOCUMENTED specifications (copied
from NVIDIA's published RTX 4090 materials) meet. The two are kept in clearly
separate structures on purpose:

- `VENDOR_SPECS_RTX_4090` — static, manufacturer-published numbers. Never
  measured on this machine. Used only as the denominator for "percent of
  theoretical peak" calculations in Parts B and C.
- `get_measured_gpu_info()` / `capture_nvidia_smi_full()` — everything else.
  These call `nvidia-smi` / the CUDA runtime directly and fail loudly if no
  GPU is present, rather than inventing placeholder numbers.

Run this file directly on the RTX 4090 workstation to populate
`results/system_info/`:

    python3 -m src.system_info --capture
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# VENDOR-DOCUMENTED SPECIFICATIONS (NOT MEASURED)
#
# Source: NVIDIA's official GeForce RTX 4090 product specification page and
# the "NVIDIA Ada GPU Architecture" whitepaper (4th-generation Tensor Cores).
# Citation: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
#
# IMPORTANT: this repository was assembled in an environment whose outbound
# network access to nvidia.com was blocked, so these figures could not be
# re-fetched at repository-creation time. They reproduce the manufacturer's
# published reference-card specifications as commonly cited in NVIDIA's own
# materials, but MUST be spot-checked by the student against the live NVIDIA
# page (open internet access is available on the GPU lab workstation) before
# `reports/METRICS.md` is finalized. If a number here is wrong, correct it in
# this one dictionary; every "% of theoretical peak" calculation reads from it.
# ---------------------------------------------------------------------------
VENDOR_SPECS_RTX_4090 = {
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "architecture": "Ada Lovelace (AD102)",
    "tensor_core_generation": "4th generation (Ada)",
    "reduced_precisions_supported_by_tensor_cores": [
        "FP16",
        "BF16",
        "TF32",
        "INT8",
        "INT4",
        "FP8 (E4M3 / E5M2)",
    ],
    "memory_type": "GDDR6X",
    "memory_bus_width_bits": 384,
    "vram_capacity_gb_spec": 24,
    "memory_bandwidth_gbps_spec": 1008.0,
    "reference_board_power_limit_w_spec": 450,
    # Dense (no structured-sparsity) theoretical peak throughput, TFLOPS.
    # FP32 is the non-Tensor CUDA-core rate; TF32/FP16/BF16 are Tensor Core rates.
    "theoretical_peak_tflops_dense": {
        "fp32": 82.6,
        "tf32": 82.6,
        "fp16": 330.3,
        "bf16": 165.2,
    },
    "vendor_documentation_citation": (
        "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/"
    ),
    "verification_note": (
        "Verify these figures against the live NVIDIA RTX 4090 specification page "
        "and the NVIDIA Ada GPU Architecture whitepaper before finalizing "
        "reports/METRICS.md. They were not re-fetched at repository-creation time "
        "because outbound network access to nvidia.com was blocked in that "
        "environment. Record the verification date/source once checked."
    ),
}


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def require_nvidia_smi() -> None:
    """Fail loudly (no fabricated data) if nvidia-smi is not reachable."""
    try:
        _run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "nvidia-smi is not available. This module must be run on the RTX "
            "4090 GPU lab workstation, not on the development machine."
        ) from exc


def get_gpu_uuid() -> str:
    """Return the real GPU UUID reported by the driver. Never fabricated."""
    require_nvidia_smi()
    return _run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).splitlines()[0].strip()


def get_measured_gpu_info() -> dict:
    """Query live, measured hardware/software facts from nvidia-smi and torch.

    Every value here comes from the driver or the installed software stack at
    call time. Nothing is hardcoded.
    """
    require_nvidia_smi()
    fields = [
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "power.limit",
        "power.max_limit",
        "clocks.max.sm",
        "clocks.max.memory",
        "compute_cap",
    ]
    query = _run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    )
    values = [v.strip() for v in query.splitlines()[0].split(",")]
    smi_info = dict(zip(fields, values))

    info = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_uuid": smi_info["uuid"],
        "gpu_name_reported": smi_info["name"],
        "driver_version": smi_info["driver_version"],
        "vram_total_mib_measured": smi_info["memory.total"],
        "power_limit_w_measured": smi_info["power.limit"],
        "power_max_limit_w_measured": smi_info["power.max_limit"],
        "sm_max_clock_mhz_measured": smi_info["clocks.max.sm"],
        "memory_max_clock_mhz_measured": smi_info["clocks.max.memory"],
        "compute_capability_measured": smi_info["compute_cap"],
    }

    try:
        import torch  # noqa: PLC0415

        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["torch_device_name"] = props.name
            info["torch_total_memory_bytes"] = props.total_memory
            info["torch_multi_processor_count"] = props.multi_processor_count
    except ImportError:
        info["torch_version"] = None
        info["torch_import_error"] = "PyTorch is not installed in this environment."

    return info


def capture_nvidia_smi_full(output_dir: Path) -> Path:
    """Save the complete `nvidia-smi -q` output verbatim. Raises if unavailable."""
    require_nvidia_smi()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nvidia_smi_full_query.txt"
    result = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, check=True)
    output_path.write_text(result.stdout)
    return output_path


def capture_all(output_dir: Path | None = None) -> None:
    output_dir = output_dir or (REPO_ROOT / "results" / "system_info")
    smi_path = capture_nvidia_smi_full(output_dir)
    print(f"Saved full nvidia-smi -q output to {smi_path}")

    measured = get_measured_gpu_info()
    measured_path = output_dir / "measured_hardware_info.json"
    measured_path.write_text(json.dumps(measured, indent=2) + "\n")
    print(f"Saved measured hardware info to {measured_path}")

    vendor_path = output_dir / "vendor_specs_rtx_4090.json"
    vendor_path.write_text(json.dumps(VENDOR_SPECS_RTX_4090, indent=2) + "\n")
    print(f"Saved vendor-documented specification snapshot to {vendor_path}")

    print(f"\nGPU UUID for this run: {measured['gpu_uuid']}")
    print("Record this UUID in provenance/reservation_record.md and reports/RUN_LOG.txt.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture nvidia-smi -q, measured hardware info, and the vendor spec "
        "snapshot into results/system_info/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory (default: results/system_info/).",
    )
    args = parser.parse_args()

    if not args.capture:
        parser.print_help()
        sys.exit(1)

    capture_all(args.output_dir)


if __name__ == "__main__":
    main()
