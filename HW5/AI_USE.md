# DATA 266 Homework 5 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write yourself?

**Response:** I mainly used Claude (Claude Code) for three things. The first was writing and organizing the Markdown explanations throughout `HW5_LoRA_Fine_Tuning.ipynb`, including the observations for Tasks 6 and 7 and the final observations. The second was the supporting documentation (`RUN_LOG.txt`, `METRICS.md`, `README.md`, `findings.pdf`, and this `AI_USE.md`). The third was error handling: tracking down and fixing problems that came up while building and running the notebook. The main ones were the duplicate evaluation dialogue described below, and the `ImportError` from Colab's preinstalled torchao 0.10, which the current peft release rejects. That fix ended up as an uninstall step in the install cell. It also helped me adapt the demo's starter code from causal language modeling to sequence-to-sequence summarization (`DataCollatorForSeq2Seq`, `Seq2SeqTrainer`, and the T5 `q`/`v` target modules). I ran the notebook myself on Google Colab with a T4 GPU, re-ran it in a fresh runtime so the submitted outputs come from one clean Run all, and checked the printed outputs (parameter counts, training losses, and the baseline, r=4, and r=16 summaries) against the Markdown before accepting the write-up.

## Question 2

Give one specific thing it produced that was wrong, such as an error, incorrect output, or implementation issue.

**Response:** The first version of the notebook picked the two evaluation dialogues as test indices `SLICE` and `SLICE + 1` (486 and 487), on the assumption that consecutive rows are different conversations. They are not. In the DialogSum test split, each dialogue appears in consecutive rows, once per reference summary (`test_162_1`, `test_162_2`, `test_162_3`), so both "evaluation dialogues" were the exact same conversation about Ernie starting a band, just with different reference summaries.

## Question 3

How did you find out? What did the failure look like?

**Response:** The notebook ran with no errors, so nothing flagged it automatically. It showed up in the printed output of the first full local run: the baseline cell printed the same dialogue under Example 1 and Example 2, the model produced identical summaries for both, and the ids were `test_162_1` and `test_162_2`. Printing test indices 480 to 498 showed the pattern clearly: every dialogue repeats three times in a row, and the test split only has 167 unique dialogues out of 499 rows.

## Question 4

What did you change, and why does your version work?

**Response:** The first evaluation dialogue is still test index `SLICE` (486), but the second one is now the next row after `SLICE` whose dialogue text is different, which turned out to be index 489 (`test_163_1`, a conversation about picking something to do in New Orleans). I also added an assertion that the two dialogues are different, next to the existing check that neither one appears in the training subset. This works because the selection now compares the actual dialogue text instead of assuming row order, so the qualitative before/after comparison really covers two different conversations. The Colab run confirms it: it prints `test indices [486, 489] -> ['test_162_1', 'test_163_1']`, and both checks pass.
