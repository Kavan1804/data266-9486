# DATA 266 Homework 7 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write yourself?

### AI-Assisted Work

- Wrote and organized the Markdown in `HW7_GAN_AE_VAE.ipynb`, including the GAN answers, AE/VAE explanations, comparison, limitations, and conclusion.
- Created `RUN_LOG.txt` and `METRICS.md`.
- Fixed issues in the demo code and reran the notebook successfully.
- Adapted the FC-AE and VAE code from MNIST to FashionMNIST and added the required checks, evaluations, figures, and seeding.

### Work I Completed Myself

- Selected the final experiment settings, including the dataset, latent dimensions, batch size, epochs, and random seed.
- Ran the notebook twice from beginning to end and confirmed that both runs produced identical results.
- Reviewed the GAN answers and checked the generated figures and outputs.
- Reviewed the final notebook and supporting files to confirm that everything was organized correctly.

## Question 2

Give one specific thing it produced that was wrong, such as an error, incorrect output, or implementation issue.

**Response:** When the notebook was run a second time to check reproducibility, the assistant compared the two runs' timestamped logs with a shell command that was supposed to ignore only the timing values. Instead, it used a filter (`grep -v` with the pattern `\| [0-9.]+s$`) that threw away every line ending in a timing like "| 5.1s". Every per-epoch training line ends that way, so all 40 epoch lines were removed before the comparison, and the command still printed "IDENTICAL (excluding timings)". It reported a reproducibility check as passed without actually comparing any of the training losses.

## Question 3

How did you find out? What did the failure look like?

**Response:** Nothing failed on screen. The command ran and printed "IDENTICAL". The problem showed up when looking at what the filter actually matched: the lines it was supposed to keep, the `[FC-AE epoch ..]` and `[VAE epoch ..]` lines with the losses, are exactly the ones that end in "| x.xs". So the "identical" result only covered the handful of lines that were left, like the shape checks and the final test metrics, not the 20 + 20 epochs of losses.

## Question 4

What did you change, and why does your version work?

**Response:** The check was rewritten so that, instead of deleting lines, it edits each line to remove only the parts that are supposed to differ between runs: the leading timestamp, the trailing "| x.xs" epoch time, and the "done in Xs" / "run times" values. Then it compares everything that is left. It also prints how many lines were compared (61 in total, including all 40 epoch lines plus the batches-per-epoch line), so it is obvious if lines go missing again. With that version, the two runs were still identical, so the reproducibility claim in `RUN_LOG.txt` is now actually backed up by a comparison of every loss value. The error is also recorded in `RUN_LOG.txt` (Step 4).

