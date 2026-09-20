# DATA 266 Homework 4 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write yourself?

**Response:** I used Claude (Claude Code) to organize and write the Markdown explanations throughout `HW4_Mini_GPT.ipynb`, to produce `findings.pdf`, to write the supporting documentation (`RUN_LOG.txt`, `METRICS.md`, this `AI_USE.md`), to debug errors that came up while building and running the notebook, and to help me understand the assignment requirements (e.g. how multi-head attention should split the hidden dimension, what the causal mask needs to guarantee, and what each decoding method is supposed to do differently). I ran the notebook myself on Google Colab with a GPU runtime, reviewed the results, and checked the printed outputs — parameter count, causal-mask verification, training loss, and the six generated samples — against what I expected before accepting the write-up.

## Question 2

Give one specific thing it produced that was wrong, such as an error, incorrect output, or implementation issue.

**Response:** The first draft of the Part 5 analysis Markdown (written before the notebook had been executed) assumed the model would end up "largely memorizing" the short Shakespeare excerpt after training, since the dataset is only about 1,700 characters, and described the generated samples in those terms. After the notebook was actually run, none of the six generated samples (greedy, three temperatures, two top-k values) looked memorized at all — they were fairly garbled character sequences with only short recognizable word fragments ("the", "thy", "she", "nor"), and greedy decoding specifically fell into a repetitive loop ("nor nor nathe", "O nor nathame") rather than reproducing training text.

## Question 3

How did you find out? What did the failure look like?

**Response:** I compared the pre-written analysis text against the notebook's actual printed output for all six Part 4 samples. The "largely memorized" description didn't match what was on the page at all — the text was visibly undertrained rather than close to the source excerpt, and greedy decoding was stuck repeating itself, which the original draft hadn't accounted for.

## Question 4

What did you change, and why does your version work?

**Response:** I had the Part 5 analysis and Final Observations Markdown rewritten to describe what the six samples actually show: greedy decoding gets stuck in a repetition loop (a known failure mode of always taking `argmax` on an undertrained model), temperature sampling produces a clear, increasing progression from more repetitive/structured at `T=0.5` to most chaotic at `T=1.5`, and top-k sampling at `k=5` still shows visible looping while `k=20` is noticeably more varied. This version works because it's grounded in the notebook's own printed output rather than an assumption made before the notebook had been run.
