#!/usr/bin/env bash
# Parts B, C, D — run the precision, bandwidth, and attention benchmarks and
# generate their figures. Run this on the GPU workstation (RTX 4090 or RTX 4060), after
# scripts/capture_gpu_info.sh has already recorded the hardware/UUID.
#
# Usage (from the HW2.5/ directory, on the GPU workstation):
#   ./scripts/run_benchmarks.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Part B: precision and achieved throughput ==="
python3 -m src.precision_benchmarks

echo
echo "=== Part C: bandwidth-bound vs. compute-bound ==="
python3 -m src.bandwidth_benchmarks

echo
echo "=== Part D: cost of attention (naive vs. fused) ==="
python3 -m src.attention_benchmarks

echo
echo "=== Generating figures for Parts B and D ==="
python3 -m src.plotting --precision --attention

echo
echo "Done. Review results/precision/, results/bandwidth/, results/attention/,"
echo "and figures/, then update reports/METRICS.md and reports/RUN_LOG.txt with"
echo "the observed numbers. Part E (thermal) is run separately with"
echo "scripts/run_thermal_test.sh because it takes ~20 minutes."
