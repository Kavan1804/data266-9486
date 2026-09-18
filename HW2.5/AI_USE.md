# DATA 266 HW2.5 — AI Use Disclosure

## Question 1

What AI assistance, if any, did you use for HW2.5? Describe the tasks assisted and the work you personally reviewed, ran, or decided.

**Response:** AI assistance (Claude Code) was used to write the single self-contained `HW2.5.ipynb` notebook (Part A hardware capture, Part B precision benchmark, Part C bandwidth/roofline benchmark, Part D naive-vs-fused attention with OOM handling, Part E sustained-load thermal logger, and the analysis/plotting cells); draft `README.md`, `reports/METRICS.md`/`RUN_LOG.txt` templates, and `provenance/` templates; and help debug issues found while reviewing the generated code. The assignment specifies a GPU Lab RTX 5090 or RTX 4090 workstation; I chose to run this on my own RTX 4060 instead, and AI assistance kept the vendor-spec section as a single editable dictionary (rather than a multi-card lookup system) so I can point it at whichever card I actually use.

AI assistance did **not** run any GPU benchmark, capture any `nvidia-smi` output, or verify any measured result — the repository was assembled on a Mac with no NVIDIA GPU, and every GPU-dependent value is left empty or marked "to be measured on the GPU workstation." I am personally responsible for: cloning this repository onto my RTX 4060 workstation, running `HW2.5.ipynb` top to bottom myself, confirming the reported GPU UUID and hardware fields against the actual workstation, filling in `reports/METRICS.md` and `reports/RUN_LOG.txt` from the real saved CSV/JSON outputs, and reviewing the `VENDOR_SPECS` dictionary in the notebook against NVIDIA's official documentation before treating it as final — the RTX 4060's TF32/FP16/BF16 TFLOPS figures were derived rather than read directly off a single vendor table, so they need the closest scrutiny.

## Question 2

Provide one specific incorrect, incomplete, or misleading AI-generated suggestion or output encountered during HW2.5.

**Response:**

<!-- Fill this in from your own experience running the assignment, e.g. a
     specific AttributeError from an older PyTorch build, an inaccurate
     assumption in a first draft of the code, or a vendor-spec number that
     needed correcting after checking NVIDIA's documentation. -->

## Question 3

How did you discover that the suggestion or output in Question 2 was incorrect, incomplete, or misleading?

**Response:**

<!-- Fill this in: e.g. a traceback from running the benchmark on the
     workstation, a mismatch between a code comment and the actual installed
     PyTorch version, or a discrepancy found while cross-checking the vendor
     specification page. -->

## Question 4

What correction did you make, and how did you verify the correction?

**Response:**

<!-- Fill this in: describe the fix and how you confirmed it worked, e.g. by
     re-running the affected benchmark module and checking the resulting CSV. -->

## Additional Notes

<!-- Space for any additional prompts, corrections, or context you want to
     record about how you used AI assistance for this assignment. -->
