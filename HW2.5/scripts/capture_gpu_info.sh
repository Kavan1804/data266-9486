#!/usr/bin/env bash
# Part A — capture nvidia-smi -q and measured hardware info on the GPU
# workstation (RTX 4090 or RTX 4060). Run this FIRST, before any other script
# in this repo.
#
# Usage (from the HW2.5/ directory, on the GPU workstation):
#   ./scripts/capture_gpu_info.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Verifying nvidia-smi is reachable ==="
nvidia-smi --query-gpu=name,uuid --format=csv,noheader

echo "=== Capturing nvidia-smi -q, measured hardware info, and vendor spec snapshot ==="
python3 -m src.system_info --capture

echo
echo "Done. Review results/system_info/, then record the reported GPU UUID and"
echo "workstation session details in provenance/reservation_record.md and"
echo "reports/RUN_LOG.txt before running scripts/run_benchmarks.sh."
