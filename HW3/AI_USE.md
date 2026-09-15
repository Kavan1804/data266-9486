# DATA 266 Homework 3 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write yourself?

**Response:** I used Claude (Claude Code) mainly to debug errors, to write and organize the Markdown explanations in HW3.ipynb, and to help create and organize the prompt examples for the six prompt-engineering techniques. It also helped write the documentation — RUN_LOG.txt, METRICS.md, and this findings PDF — based on whatever the notebook actually output. I ran the notebook myself, including running Part 1 in Google Colab with my own `GOOGLE_API_KEY` to get the real model responses, reviewed the generated code before trusting it, checked the printed outputs against what I expected, and verified the attention assertions myself rather than just assuming they passed.

## Question 2

Give one specific thing it produced that was wrong, such as an error, incorrect output, or implementation issue.

**Response:** When Claude first executed the self-attention section of the notebook (Part 2) with `nbclient`, the shell environment it ran the command in had `MPLBACKEND=Agg` set. Both `plot_attention()` cells (the unmasked and masked heatmaps) ran with no exception, but instead of saving an image they only produced this warning:

```
FigureCanvasAgg is non-interactive, and thus cannot be shown
```

No `image/png` output was captured in the notebook for either cell, so the two heatmaps were just missing even though the attention code itself was correct.

## Question 3

How did you find out? What did the failure look like?

**Response:** The execution finished without any error, so at a glance it looked like it had worked. It was only when I looked at the saved notebook and the two heatmap cells had no image output at all — just that one warning line each — that I noticed something was off. Claude then ran a small isolated test (plotting a simple figure with and without `MPLBACKEND=Agg` set) and confirmed the environment variable was the difference: with it set, `ipykernel` never captured the matplotlib figure as a cell output.

## Question 4

What did you change, and why does your version work?

**Response:** The fix was to drop `MPLBACKEND=Agg` from the environment before re-running the executor, so `ipykernel`'s default inline backend could capture the matplotlib figures automatically — nothing in `plot_attention()` or the model itself needed to change. Re-running the same notebook code with that override removed produced real `image/png` outputs for both heatmap cells. I then checked that these were genuinely post-training results (matching the printed attention shapes and the loss values from the same run, not leftover/random-weight images), and confirmed the causal-masking assertions — attention rows summing to about 1, the causal mask being exactly lower-triangular, and zero attention weight above the diagonal — all passed against those actual trained outputs before I considered Part 2 done.
