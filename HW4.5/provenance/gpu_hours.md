# GPU-Hours Used — HW4.5

GPU used for the reported results: **Tesla T4** (Google Colab), as reported by `nvidia-smi`.

Durations are from the notebooks' own start and end timestamps (`results/gpu/env_hw4a.json`, `env_hw45.json` and the run
logs). They don't include package installation at the start of each Colab session, so the real session time is a little longer.

| Date | Session start | Session end | Duration (hours) | Activity | GPU UUID |
| --- | --- | --- | ---: | --- | --- |
| 2026-10-27 | not recorded in the repo | 22:54:40Z (15:54 PDT) | not recorded | RTX 4090, Colab local Docker runtime: HW4.a attempt 1, vLLM failed (FlashInfer needed `nvcc`); not used for results | not in the repo |
| 2026-10-28 | 00:57:46Z (Oct 27 17:57 PDT) | 01:05:14Z (Oct 27 18:05 PDT) | 0.124 | RTX 4090, Colab local Docker runtime: HW4.a attempt 2, vLLM failed (GPU memory held by a test cell); not used for results | not in the repo |
| 2026-11-02 | 23:27:57Z (15:27 PST) | 23:48:50Z (15:48 PST) | 0.348 | Tesla T4, Colab: HW4.a full run (fine-tune + serving benchmark) | GPU-585cb49b-329b-b3d3-3094-9d792639bfd4 |
| 2026-11-03 | 00:08:29Z (Nov 2 16:08 PST) | 01:31:24Z (Nov 2 17:31 PST) | 1.382 | Tesla T4, Colab: HW4.5 Parts A, B, D, E | GPU-d44fecb0-9de2-0bc4-8f0f-edca5b8c66f2 |

**Total GPU-hours for the reported results (T4):** 0.348 + 1.382 = **1.730 GPU-hours**.

**Earlier RTX 4090 attempts:** at least 0.124 GPU-hours (attempt 1's start time is not recorded in the repo). The
executed notebook from attempt 2 is commit 0323ebe.

## Notes

- The two T4 runs were in separate Colab sessions and got two different physical T4s (different UUIDs).
- Both RTX 4090 attempts ran only HW4.a and stopped at the vLLM step; their errors and fixes are in `RUN_LOG.txt` (Step 2b).
