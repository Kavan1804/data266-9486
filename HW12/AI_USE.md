# DATA 266 Homework 12 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write yourself?

### AI-Assisted Work

- Helped write and polish documentation: Markdown in `HW12_GPTQ_Distillation.ipynb`, plus `README.md`, `RUN_LOG.txt`, `METRICS.md`, and this file.
- Helped debug errors while building and running the notebook (see Questions 2–4).
- Reviewed and refined code I had already written (GPTQ, distillation loss / training loop, sweeps, measurements, and figures), including clearer structure and small implementation fixes.
- Helped turn run outputs into findings, limitations, and conclusion write-ups (Sections 7, 15, and 16) and the tables in `METRICS.md`.

### Work I Completed Myself

- Obtained the teacher model and assignment materials, and kept HW12 separate from the other homeworks.
- Implemented the core notebook logic: GPTQ, distillation training, sweeps, and evaluation.
- Chose settings such as OPT-125m, group size 128, the validation split, and training epochs.
- Ran the notebook on Kaggle (smoke test, then full run) and downloaded the executed notebook and outputs.
- Found the GPTQ link in the assignment for checking the base model, and decided to keep `opt-125m` and the PyTorch implementation.

## Question 2

Give one specific thing it produced that was wrong, such as an error, incorrect output, or implementation issue.

**Response:** While reviewing my teacher-logit caching code, the assistant flagged that I had wrapped the function in `@torch.inference_mode()`. Those logits are reused later in the student training loss, and PyTorch does not allow inference tensors to be saved for the backward pass.

## Question 3

How did you find out? What did the failure look like?

**Response:** The issue was caught during code review before the full training run, so there was no crash in the final Kaggle job. If it had reached training unchanged, it could have failed with an error like "Inference tensors cannot be saved for backward."

## Question 4

What did you change, and why does your version work?

**Response:** I switched the function to `torch.no_grad()`. That still skips gradient computation for the teacher (which is what we want) but returns normal tensors that can participate in the student training loss.
