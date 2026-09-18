"""Part E — sustained load and thermal behaviour.

Runs a sustained GPU compute load (default 20 minutes) while a background
sampling thread queries `nvidia-smi` every 5 seconds for elapsed time, SM
clock, memory clock, temperature, power draw, GPU utilization, and the real
GPU UUID. Samples are appended to a CSV and flushed to disk immediately, so
the log survives an interruption instead of being lost.

This script is meant to be run directly from the command line on the GPU
workstation (RTX 4090 or RTX 4060), e.g.:

    python3 -m src.thermal_benchmark --duration-seconds 1200 --interval-seconds 5 \\
        --output results/thermal/thermal_log.csv

Afterward, run with --analyze to compute throttling/throughput statistics
from the CSV that was actually recorded (never invented ahead of the run):

    python3 -m src.thermal_benchmark --analyze results/thermal/thermal_log.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "thermal" / "thermal_log.csv"
DEFAULT_ANALYSIS_JSON = REPO_ROOT / "results" / "thermal" / "thermal_analysis.json"

CSV_FIELDS = [
    "elapsed_s",
    "timestamp_utc",
    "gpu_uuid",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "temperature_c",
    "power_draw_w",
    "utilization_pct",
    "cumulative_matmuls",
    "interval_throughput_matmuls_per_s",
]

SMI_QUERY_FIELDS = "uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"


class _MatmulCounter:
    """Simple shared counter the compute loop increments and the sampler reads.

    Reads/writes of a Python int under the GIL are atomic enough for this
    approximate, human-scale (5-second cadence) throughput sampling.
    """

    def __init__(self) -> None:
        self.count = 0


def _query_nvidia_smi_sample() -> dict:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={SMI_QUERY_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    uuid, sm_clock, mem_clock, temp, power, util = [v.strip() for v in result.stdout.splitlines()[0].split(",")]
    return {
        "gpu_uuid": uuid,
        "sm_clock_mhz": float(sm_clock),
        "memory_clock_mhz": float(mem_clock),
        "temperature_c": float(temp),
        "power_draw_w": float(power),
        "utilization_pct": float(util),
    }


def _sampler_loop(
    stop_event: threading.Event,
    interval_s: float,
    start_time: float,
    counter: _MatmulCounter,
    csv_writer,
    csv_file,
) -> None:
    last_count = 0
    last_sample_time = start_time
    next_sample_at = start_time + interval_s
    while not stop_event.is_set():
        now = time.time()
        if now < next_sample_at:
            time.sleep(min(0.25, next_sample_at - now))
            continue
        try:
            smi_sample = _query_nvidia_smi_sample()
        except subprocess.CalledProcessError as exc:
            smi_sample = {
                "gpu_uuid": "UNKNOWN",
                "sm_clock_mhz": float("nan"),
                "memory_clock_mhz": float("nan"),
                "temperature_c": float("nan"),
                "power_draw_w": float("nan"),
                "utilization_pct": float("nan"),
            }
            print(f"[thermal_benchmark] nvidia-smi sample failed: {exc}")

        now = time.time()
        current_count = counter.count
        dt = now - last_sample_time
        throughput = (current_count - last_count) / dt if dt > 0 else float("nan")

        row = {
            "elapsed_s": round(now - start_time, 3),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_uuid": smi_sample["gpu_uuid"],
            "sm_clock_mhz": smi_sample["sm_clock_mhz"],
            "memory_clock_mhz": smi_sample["memory_clock_mhz"],
            "temperature_c": smi_sample["temperature_c"],
            "power_draw_w": smi_sample["power_draw_w"],
            "utilization_pct": smi_sample["utilization_pct"],
            "cumulative_matmuls": current_count,
            "interval_throughput_matmuls_per_s": throughput,
        }
        csv_writer.writerow(row)
        csv_file.flush()

        last_count = current_count
        last_sample_time = now
        next_sample_at += interval_s


def run_sustained_load(
    duration_s: int = 1200,
    interval_s: float = 5.0,
    matmul_n: int = 8192,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
) -> Path:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run the thermal benchmark on the GPU "
            "workstation, not on the development machine."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.float16
    a = torch.randn(matmul_n, matmul_n, device=device, dtype=dtype)
    b = torch.randn(matmul_n, matmul_n, device=device, dtype=dtype)

    counter = _MatmulCounter()
    stop_event = threading.Event()

    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        start_time = time.time()
        sampler_thread = threading.Thread(
            target=_sampler_loop,
            args=(stop_event, interval_s, start_time, counter, writer, csv_file),
            daemon=True,
        )
        sampler_thread.start()

        print(
            f"Starting sustained load: duration={duration_s}s, sampling every "
            f"{interval_s}s, matmul_n={matmul_n}, dtype={dtype}."
        )
        end_time = start_time + duration_s
        try:
            while time.time() < end_time:
                _ = a @ b
                counter.count += 1
        except KeyboardInterrupt:
            print("\n[thermal_benchmark] Interrupted; saving partial log and stopping cleanly.")
        finally:
            torch.cuda.synchronize()
            stop_event.set()
            sampler_thread.join(timeout=interval_s * 2)

    print(f"Sustained load complete. Log saved continuously to {output_csv}")
    return output_csv


def analyze_thermal_log(csv_path: Path) -> dict:
    """Compute throttling/throughput statistics from an already-recorded log.

    Every number here is derived from the CSV rows written during the actual
    run; nothing about throttling, temperature, or throughput is assumed in
    advance.
    """
    import csv as csv_module  # noqa: PLC0415

    rows = []
    with csv_path.open(newline="") as fh:
        reader = csv_module.DictReader(fh)
        for r in reader:
            rows.append(
                {
                    "elapsed_s": float(r["elapsed_s"]),
                    "sm_clock_mhz": float(r["sm_clock_mhz"]) if r["sm_clock_mhz"] not in ("", "nan") else float("nan"),
                    "temperature_c": float(r["temperature_c"]) if r["temperature_c"] not in ("", "nan") else float("nan"),
                    "power_draw_w": float(r["power_draw_w"]) if r["power_draw_w"] not in ("", "nan") else float("nan"),
                    "throughput": float(r["interval_throughput_matmuls_per_s"])
                    if r["interval_throughput_matmuls_per_s"] not in ("", "nan")
                    else float("nan"),
                }
            )

    if not rows:
        return {"status": "no_data", "note": f"{csv_path} contains no rows yet."}

    total_duration = max(r["elapsed_s"] for r in rows)
    first_30s = [r for r in rows if r["elapsed_s"] <= 30]
    last_5min = [r for r in rows if r["elapsed_s"] >= total_duration - 300]

    peak_sm_clock_first_30s = max((r["sm_clock_mhz"] for r in first_30s if r["sm_clock_mhz"] == r["sm_clock_mhz"]), default=float("nan"))
    max_temperature = max((r["temperature_c"] for r in rows if r["temperature_c"] == r["temperature_c"]), default=float("nan"))
    max_power = max((r["power_draw_w"] for r in rows if r["power_draw_w"] == r["power_draw_w"]), default=float("nan"))

    # Throttling heuristic: SM clock sustained at or below 97% of the early
    # (first-30s) peak clock for at least two consecutive samples, after the
    # first 30 seconds have elapsed.
    throttle_onset_s = None
    if peak_sm_clock_first_30s == peak_sm_clock_first_30s:  # not NaN
        threshold = 0.97 * peak_sm_clock_first_30s
        consecutive_low = 0
        for r in rows:
            if r["elapsed_s"] <= 30:
                continue
            if r["sm_clock_mhz"] == r["sm_clock_mhz"] and r["sm_clock_mhz"] <= threshold:
                consecutive_low += 1
                if consecutive_low >= 2 and throttle_onset_s is None:
                    throttle_onset_s = r["elapsed_s"]
            else:
                consecutive_low = 0

    valid_first_30s_throughput = [r["throughput"] for r in first_30s if r["throughput"] == r["throughput"]]
    valid_last_5min_throughput = [r["throughput"] for r in last_5min if r["throughput"] == r["throughput"]]
    peak_throughput = max(valid_first_30s_throughput) if valid_first_30s_throughput else float("nan")
    steady_state_throughput = (
        sum(valid_last_5min_throughput) / len(valid_last_5min_throughput) if valid_last_5min_throughput else float("nan")
    )
    steady_state_pct_of_peak = (
        (steady_state_throughput / peak_throughput) * 100.0
        if peak_throughput and peak_throughput == peak_throughput and peak_throughput != 0
        else float("nan")
    )

    return {
        "source_csv": str(csv_path),
        "total_logged_duration_s": total_duration,
        "num_samples": len(rows),
        "peak_sm_clock_mhz_first_30s": peak_sm_clock_first_30s,
        "max_temperature_c_observed": max_temperature,
        "max_power_draw_w_observed": max_power,
        "throttling_detected": throttle_onset_s is not None,
        "throttle_onset_time_s": throttle_onset_s if throttle_onset_s is not None else "none",
        "peak_throughput_matmuls_per_s_first_30s": peak_throughput,
        "steady_state_throughput_matmuls_per_s_last_5min": steady_state_throughput,
        "steady_state_pct_of_peak_throughput": steady_state_pct_of_peak,
        "note": (
            "Throttling is flagged heuristically as the SM clock staying at or "
            "below 97% of its first-30-second peak for two or more consecutive "
            "5-second samples after the first 30 seconds. Confirm against the "
            "figures/thermal_clock_temperature.png plot before writing report "
            "conclusions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=1200)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--matmul-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--analyze",
        type=Path,
        default=None,
        help="Skip running the load; analyze an already-recorded thermal CSV instead.",
    )
    parser.add_argument("--analysis-output", type=Path, default=DEFAULT_ANALYSIS_JSON)
    args = parser.parse_args()

    if args.analyze is not None:
        analysis = analyze_thermal_log(args.analyze)
        args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
        args.analysis_output.write_text(json.dumps(analysis, indent=2) + "\n")
        print(json.dumps(analysis, indent=2))
        print(f"\nSaved analysis to {args.analysis_output}")
        return

    run_sustained_load(
        duration_s=args.duration_seconds,
        interval_s=args.interval_seconds,
        matmul_n=args.matmul_size,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
