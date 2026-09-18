#!/usr/bin/env bash
# Part E — sustained load and thermal behaviour. Runs for ~20 minutes by
# default. Run this on the GPU workstation (RTX 4090 or RTX 4060); the CSV log is
# flushed continuously, so an interruption (Ctrl-C) still leaves a usable
# partial log.
#
# Usage (from the HW2.5/ directory, on the GPU workstation):
#   ./scripts/run_thermal_test.sh [duration_seconds]
set -euo pipefail
cd "$(dirname "$0")/.."

DURATION_SECONDS="${1:-1200}"
OUTPUT_CSV="results/thermal/thermal_log.csv"

echo "=== Part E: sustained load for ${DURATION_SECONDS}s, sampling every 5s ==="
python3 -m src.thermal_benchmark \
  --duration-seconds "${DURATION_SECONDS}" \
  --interval-seconds 5 \
  --output "${OUTPUT_CSV}"

echo
echo "=== Analyzing the recorded thermal log ==="
python3 -m src.thermal_benchmark --analyze "${OUTPUT_CSV}"

echo
echo "=== Generating the clock/temperature figure ==="
python3 -m src.plotting --thermal

echo
echo "Done. Review results/thermal/thermal_analysis.json and"
echo "figures/thermal_clock_temperature_vs_time.png, then fill in the Part E"
echo "sections of reports/METRICS.md from those observed values."
