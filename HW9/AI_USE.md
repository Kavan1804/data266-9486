# DATA 266 Homework 9 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write yourself?

### AI-Assisted Work

- Wrote and organized the Markdown in `HW9_Latent_Diffusion_CFG.ipynb`, including the explanations of the VAE compression, the latent scaling, forward diffusion in latent space, the U-Net architecture, classifier-free guidance, and the limitations. It also drafted the observations, analysis, and conclusion from the outputs of my run.
- Helped organize the notebook structure around the five parts of the assignment and the two demo notebooks.
- Created `README.md`, `RUN_LOG.txt`, `METRICS.md`, and this file.
- Helped debug errors that came up while building and running the notebook. It also ran the short smoke test on my Mac (64 images, 2 epochs), which only checks that the cells run and isn't part of the reported results.
- Generated the notebook with a build script based on the demos and helped refine the code: adapting the CFG demo's U-Net to 4-channel VAE latents, adding the missing shortcut in the residual block, caching the VAE latents, the micro-batching, the checkpoint/resume logic, and the output zip cell.
- Helped clarify implementation details, for example why a 64×64 image gives an 8×8 latent with no 16×16 level in the U-Net, which is why I resized the images to 256×256.

### Work I Completed Myself

- Reviewed the code and made the decisions about the final implementation (the 256×256 input size, keeping the demo's per-batch label dropout, the batch size of 1000).
- Ran the full notebook on a Kaggle T4 GPU (about 2 hours 5 minutes) after my first attempt on Colab ran out of credits at epoch 39, and evaluated the results myself. The AI did not run the full experiment or produce the reported results.
- Reviewed the figures, the generated sample grids, and the final notebook and supporting files.

## Question 2

Give one specific thing it produced that was wrong, such as an error, incorrect output, or implementation issue.

**Response:** The first version of the settings cell decided it was on Colab by checking whether the `google.colab` package could be imported, and if so it mounted Google Drive for backups. That check is wrong on Kaggle, because Kaggle also has the `google.colab` package installed.

## Question 3

How did you find out? What did the failure look like?

**Response:** When I moved the run to Kaggle, the notebook crashed right at the settings cell with a `NotImplementedError` from `drive.mount()`. Nothing had trained yet, so it was an easy failure to see. The traceback pointed at the Drive mount, which only made sense on Colab, so the problem had to be the environment check and not the training code.

## Question 4

What did you change, and why does your version work?

**Response:** The check now looks for `/var/colab/hostname`, a file that only exists on a real Colab machine, and it also looks for Kaggle's `KAGGLE_KERNEL_RUN_TYPE` variable and `/kaggle` folder. The zip cell uses the same `IN_COLAB` flag, so it doesn't try to call Colab's file download on Kaggle either. This works because it checks which machine the notebook is running on instead of which packages are installed. The settings cell then printed `Colab: False | Kaggle: True | backup folder: None`, and the full run finished. The downside is that Kaggle has no Drive backup, so that run had to finish in one session.
