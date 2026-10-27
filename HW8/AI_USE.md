# DATA 266 Homework 8 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write yourself?

### AI-Assisted Work

- Wrote and organized the Markdown in `HW8_Diffusion_DDPM_DDIM.ipynb`, including the explanations of the forward process, the beta schedule, timestep conditioning, DDPM and DDIM sampling, and the Section 13 discussion and conclusion.
- Created `RUN_LOG.txt`, `METRICS.md`, `README.md`, and `findings.pdf`.
- Found and fixed errors that came up while building and running the notebook. These included overlapping figure titles, a misleading comparison metric (see Question 2), and a log comparison that skipped the epoch lines.
- Helped structure the code on top of the demo: the larger time-conditioned CNN, the training loop, the DDPM and DDIM samplers, the comparison figures, and the start-noise diagnostic.

### Work I Completed Myself

- Ran the full notebook twice from beginning to end on my laptop (about 1 hour 38 minutes each) and confirmed that both runs gave the same losses and samples.
- Reviewed the generated figures and sample grids and judged the quality of the DDPM and DDIM samples.
- Reviewed the final notebook, the report, and the supporting files to confirm that everything was organized correctly and the numbers match the run.

## Question 2

Give one specific thing it produced that was wrong, such as an error, incorrect output, or implementation issue.

**Response:** The first version of the DDPM vs DDIM comparison table had an "outside [-1, 1]" column: the fraction of generated pixels outside the valid range before clamping. The notebook text described a high value as a sign that the sampler was overshooting. In the first full run, DDPM had 25.07% of its pixels outside [-1, 1], compared with 0.25% and 0.37% for DDIM. Read that way, DDPM looked like the worst sampler by a large margin, even though its images were the cleanest of the three.

## Question 3

How did you find out? What did the failure look like?

**Response:** The number did not match the pictures. The DDPM grid had the smoothest strokes and the cleanest black background, but the table said a quarter of its pixels were out of range. The log line for that run showed that DDPM's output range was only [-1.04, 1.04]. So the "out of range" pixels were almost all background pixels slightly below -1, which become exactly black after clamping. DDPM pushes the background a tiny bit past -1, while DDIM leaves it slightly above -1. The metric was measuring that, not overshoot or bad samples.

## Question 4

What did you change, and why does your version work?

**Response:** The column was replaced by a "gray fraction": the share of pixels between 0.1 and 0.9 after rescaling to [0, 1]. Real MNIST digits are almost entirely pure black or pure white (gray fraction 0.085 on 1,000 test images), so a high gray fraction means speckle or blur, which is the actual quality difference you can see in the grids. In the reported run, DDPM's gray fraction is 0.160 and both DDIM runs are about 0.25, which matches what the images show. The markdown claim about overshoot was removed. The change is recorded in `RUN_LOG.txt` (Step 4). The second full run confirmed that all the other numbers stayed the same.
