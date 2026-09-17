"""Figure generation for Parts B, D, and E.

Each function reads a results CSV produced by the corresponding benchmark
module and renders one figure. If the CSV does not exist yet or is empty
(i.e., the benchmark has not been run on the RTX 4090 workstation yet), the
function prints a clear message and returns without writing a figure --
it never fabricates a plot from placeholder numbers.

Run all figures at once:

    python3 -m src.plotting --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"

PRECISION_CSV = REPO_ROOT / "results" / "precision" / "precision_benchmark_results.csv"
ATTENTION_CSV = REPO_ROOT / "results" / "attention" / "attention_benchmark_results.csv"
ATTENTION_SUMMARY_JSON = REPO_ROOT / "results" / "attention" / "attention_summary.json"
THERMAL_CSV = REPO_ROOT / "results" / "thermal" / "thermal_log.csv"


def _read_csv_rows(csv_path: Path) -> list[dict]:
    import csv

    if not csv_path.exists():
        print(f"[plotting] {csv_path} does not exist yet. Run the benchmark on the "
              f"RTX 4090 workstation first; skipping this figure.")
        return []
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"[plotting] {csv_path} is empty; skipping this figure.")
    return rows


def plot_precision_scaling(csv_path: Path = PRECISION_CSV, out_path: Path | None = None) -> Path | None:
    rows = _read_csv_rows(csv_path)
    if not rows:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or (FIGURES_DIR / "precision_tflops_vs_matrix_size.png")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    by_precision: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        by_precision.setdefault(row["precision"], []).append(
            (float(row["matrix_n"]), float(row["achieved_tflops"]))
        )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for precision, points in sorted(by_precision.items()):
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=precision.upper())

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix size N (N x N)")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title("HW2.5 Part B — Achieved TFLOPS vs. Matrix Size by Precision")
    ax.legend(title="Precision")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plotting] Saved {out_path}")
    return out_path


def plot_attention_memory(
    csv_path: Path = ATTENTION_CSV,
    summary_path: Path = ATTENTION_SUMMARY_JSON,
    out_path: Path | None = None,
) -> Path | None:
    rows = _read_csv_rows(csv_path)
    if not rows:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = out_path or (FIGURES_DIR / "attention_peak_memory_vs_seqlen.png")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for implementation, marker in (("naive", "o"), ("fused", "s")):
        success_points = sorted(
            (
                (float(r["seq_len"]), float(r["peak_memory_mb"]))
                for r in rows
                if r["implementation"] == implementation and r["status"] == "success"
            ),
            key=lambda p: p[0],
        )
        if success_points:
            xs = [p[0] for p in success_points]
            ys = [p[1] for p in success_points]
            ax.plot(xs, ys, marker=marker, label=f"{implementation} (success)")

        failure_points = sorted(
            float(r["seq_len"]) for r in rows if r["implementation"] == implementation and r["status"] == "oom"
        )
        for x in failure_points:
            ax.axvline(x, color="red", linestyle=":", alpha=0.3)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        fit = summary.get("memory_curve_fit_naive", {})
        if fit.get("status") == "fit":
            a, b, c = (
                fit["quadratic_coefficient_a_mb_per_seqlen_sq"],
                fit["linear_coefficient_b_mb_per_seqlen"],
                fit["intercept_c_mb"],
            )
            xs_fit = np.array(sorted(fit["fitted_on_seq_lengths"]))
            xs_dense = np.linspace(xs_fit.min(), xs_fit.max(), 200)
            ys_fit = a * xs_dense ** 2 + b * xs_dense + c
            ax.plot(xs_dense, ys_fit, linestyle="--", color="black", alpha=0.6, label="naive quadratic fit")

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Peak allocated GPU memory (MB)")
    ax.set_title("HW2.5 Part D — Peak Memory vs. Sequence Length (naive vs. fused attention)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plotting] Saved {out_path}")
    return out_path


def plot_thermal(csv_path: Path = THERMAL_CSV, out_path: Path | None = None) -> Path | None:
    rows = _read_csv_rows(csv_path)
    if not rows:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or (FIGURES_DIR / "thermal_clock_temperature_vs_time.png")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    elapsed = [float(r["elapsed_s"]) for r in rows]
    sm_clock = [float(r["sm_clock_mhz"]) for r in rows]
    mem_clock = [float(r["memory_clock_mhz"]) for r in rows]
    temperature = [float(r["temperature_c"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(elapsed, sm_clock, color="tab:blue", label="SM clock (MHz)")
    ax1.plot(elapsed, mem_clock, color="tab:cyan", label="Memory clock (MHz)")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Clock (MHz)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(elapsed, temperature, color="tab:red", label="Temperature (C)")
    ax2.set_ylabel("Temperature (C)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center")

    ax1.set_title("HW2.5 Part E — Clock and Temperature vs. Time (sustained load)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plotting] Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--precision", action="store_true")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--thermal", action="store_true")
    args = parser.parse_args()

    run_all = args.all or not any([args.precision, args.attention, args.thermal])
    if run_all or args.precision:
        plot_precision_scaling()
    if run_all or args.attention:
        plot_attention_memory()
    if run_all or args.thermal:
        plot_thermal()


if __name__ == "__main__":
    main()
