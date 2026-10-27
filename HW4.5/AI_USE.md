# DATA 266 HW4.a + HW4.5 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write or verify yourself?

**Response:** I used Claude (Claude Code) for error debugging, for documentation and Markdown writing (the notebook explanations, `RUN_LOG.txt`, `METRICS.md`, this file and the provenance templates), for organizing the structure of the two notebooks, and for refining code that I reviewed and tested. The assistant drafted the notebook code from the assignment requirements using a notebook-building script, as recorded in `RUN_LOG.txt`. I reviewed that code and checked it with the local smoke test before using it.

I made the assignment-specific decisions myself:
- splitting the work into an HW4.a notebook and an HW4.5 notebook, since I hadn't done HW4.a before,
- choosing TweetEval sentiment after comparing the class balance of my slice with the alternatives,
- skipping Part C as allowed by the instructor's clarification,
- reporting only the one RTX 4090 I actually had, instead of a two-card comparison.

I run the experiments on the lab RTX 4090, check the outputs against the sanity checks built into the notebooks, and interpret the measured results in the conclusions.

## Question 2

Give one specific thing it produced that was wrong.

**Response:** The first version of the Part B explanation said that gradient accumulation (physical batch 8, four micro-batches per optimizer step) uses the same samples and the same number of optimizer steps as the baseline. That part was true. But the text implied the result would effectively be the same training run as the baseline. It isn't, because dropout draws a different random mask for four forward passes of 8 than for one forward pass of 32. So the accumulated run is not numerically identical to the baseline, and its accuracy can differ.

## Question 3

How did you find out? What did the failure look like?

**Response:** In the local smoke test (48 training rows, CPU), the gradient accumulation run finished with test accuracy 0.583, while the baseline finished at 0.500. If the two runs were really equivalent, the accuracies should have matched. That mismatch made me look at what differs between the two code paths. Everything else matched: the data order, the sample-weighted loss, the clipping and the LR schedule. The dropout masks were the only difference.

## Question 4

What did you change, and why does your version work?

**Response:** The Part B Markdown now says that gradient accumulation keeps the same samples per optimizer step and the same number of steps as the baseline, so its wall-clock time is "time to the same number of optimizer steps". It also says that accuracy won't match exactly because the dropout masks differ. This describes what the code actually does, and it matches the smoke-test numbers instead of an assumption made before running.

In the same review I also found that the KV-cache sweep would have stopped at 16,384 tokens: 32,768 plus the 16 new tokens is over Qwen2.5-0.5B's 32,768-position limit. The last context length is now clamped to 32,752, so the run tests the model's real limit instead of under-reporting it.
