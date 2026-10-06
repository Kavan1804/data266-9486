# DATA 266 Homework 6 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write yourself?

**Response:** I mainly used Claude for three things. The first was writing and organizing the Markdown explanations in `HW6_STL10_SSL.ipynb`, including the Final Comparison, which was written only after the Colab run, from the actual accuracies, curves, and nearest-neighbor figures. The second was the supporting documentation (`RUN_LOG.txt`, `METRICS.md`, `README.md`, `findings.pdf`, and this `AI_USE.md`). The third was error handling: running a short local test of the notebook before the real run and fixing what it turned up, which is described below. It also helped me adapt the starter code from `Demo_6_Self_Supervised_Learning.ipynb` (the rotation dataset, the two-view dataset, the NT-Xent loss, and the nearest-neighbor code) from CIFAR-10 and the demo's tiny CNN to STL-10 and ResNet-18, and add the supervised baseline and the frozen-encoder linear evaluation. I kept the demo's four SimCLR augmentations and tau = 0.2. I ran the full notebook myself on Google Colab with a T4 GPU, brought the figures and results back into the repo, and checked the printed checks (500-image subset fingerprint, frozen encoders, cosine similarity, same query images) and the final accuracies before accepting the write-up.

## Question 2

Give one specific thing it produced that was wrong, such as an error, incorrect output, or implementation issue.

**Response:** The first version of the Part D nearest-neighbor figure was laid out wrong. Each figure has three rows (Supervised, Rotation SSL, SimCLR) with the query and its 5 neighbors, and every neighbor has a three-line title (test index, class, cosine score). The figure was created with `figsize=(15, 8.4)`, which was not tall enough for that. In rows 2 and 3, the neighbor titles were drawn on top of the bottom of the images in the row above, so the test index line was partly covered and hard to read.

## Question 3

How did you find out? What did the failure look like?

**Response:** The code ran without any error, so nothing flagged it. It showed up when looking at the saved figure `partD_nearest_neighbors_query3.png` from the short local test run (1 epoch per part, small unlabeled subset) done before the real Colab run. In the Rotation SSL and SimCLR rows, the red and green neighbor titles ran into the images above them, and labels like "idx=5605" were half hidden behind the Supervised row's pictures.

## Question 4

What did you change, and why does your version work?

**Response:** The figure height was changed from 8.4 to 10.5 inches (`figsize=(15, 10.5)`), with the same `tight_layout()` call. That gives each row enough vertical space for its three-line titles above the images. The new layout was checked with a test figure using the same titles before running the notebook on Colab, and the three figures from the real run (`figures/partD_nearest_neighbors_query1.png` to `query3.png`) show every title clearly, with no overlap. This matters because the class labels and cosine scores in those titles are what the nearest-neighbor comparison in the notebook is based on.
