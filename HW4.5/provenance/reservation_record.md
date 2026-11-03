# GPU Reservation Record — HW4.5

The reported results were run on a hosted Google Colab T4 runtime. Colab assigns GPUs on demand, so there was no
booking for these runs. Fields that were not recorded anywhere are marked for me to fill in by hand.

| Field | Value |
| --- | --- |
| Student | Kavan Siddesh Kumar (SID4 9486) |
| Workstation identifier / hostname | Google Colab hosted runtime (VM hostname not recorded) |
| GPU model (reported by `nvidia-smi`) | Tesla T4, 14.56 GiB, driver 580.82.07, CUDA 13.0 |
| GPU UUID (reported by `nvidia-smi`) | HW4.a: GPU-585cb49b-329b-b3d3-3094-9d792639bfd4 (`nvidia_smi_q_hw4a_2026-11-02T232757Z.txt`); HW4.5: GPU-d44fecb0-9de2-0bc4-8f0f-edca5b8c66f2 (`nvidia_smi_q_hw45_2026-11-03T000829Z.txt`, `nvidia_smi_L_hw45_2026-11-03T000829Z.txt`) |
| Reservation system / booking reference | None (Colab on-demand GPU). Colab plan: free |
| Reservation start / end | Not applicable |
| Actual session start (date, time, timezone) | HW4.a: 2026-11-02 23:27:57 UTC (15:27 PST); HW4.5: 2026-11-03 00:08:29 UTC (Nov 2 16:08 PST), from the notebook logs |
| Actual session end (date, time, timezone) | HW4.a: 2026-11-02 23:48:50 UTC (15:48 PST); HW4.5: 2026-11-03 01:31:24 UTC (Nov 2 17:31 PST) |
| Purpose | DATA 266 HW4.a baseline and HW4.5 GPU Assignment II (Parts A, B, D, E; Part C skipped as optional) |
| Second GPU / pairing | None. Part C skipped (optional) per the instructor's clarification |

## Session Notes

- HW4.a and HW4.5 ran in two separate Colab sessions, on two different physical T4s.
- Before the T4 runs, HW4.a was attempted twice on an RTX 4090 desktop through a Colab local Docker runtime
  (2026-10-27/28). Both attempts stopped at the vLLM step and are not used for any result; see `RUN_LOG.txt` Step 2b
  and `gpu_hours.md`. Reservation details for that desktop: _fill in if the lab requires them_.
