# DATA 266 Homework 11 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write yourself?

### AI-Assisted Work

- Wrote and organized the Markdown in `HW11_CLIP_Similarity_ZeroShot.ipynb`, including the explanations of CLIP's two encoders and shared embedding space, the preprocessing, the L2 normalization and cosine similarity, and how zero-shot classification and prompt ensembling work. It also drafted the findings, limitations, and conclusion from the outputs of the run.
- Helped organize the notebook around the steps in the assignment (load CLIP, define the caption, preprocess, extract and normalize the embeddings, compute similarities, visualize, document, then zero-shot CIFAR-10), starting from `Demo_11_CLIP.ipynb` (`CLIPModel` / `CLIPProcessor`, FashionMNIST, PIL inputs, the prompt list, `logits_per_image` with softmax) and the CIFAR-10 loading and evaluation loop from `Demo_11_Knowledge_Distillation.ipynb`.
- Created `README.md`, `RUN_LOG.txt`, `METRICS.md`, and this file.
- Helped debug errors that came up while building and running the notebook, for example a kernel crash on my Mac caused by the order of the imports (see Questions 2-4), and the change in transformers 5 where `get_image_features` returns an output object instead of a tensor.
- Generated the notebook with a build script and helped refine the code structure and readability: the `encode_images` / `encode_texts` helpers with the check against the demo's logits, the per-class score table, the figures, the cached CIFAR-10 embeddings, and the smoke-test switch.
- At my request it executed the smoke test and the full notebook on my Mac through Claude Code, so the outputs saved in the notebook come from that run.

### Work I Completed Myself

- Decided how to set up the experiment: the caption comes from my CLS_A (Shirt), 100 FashionMNIST test images per class starting at my SLICE, all 10,000 CIFAR-10 test images, and the three prompt sets to compare.
- Reviewed the code and checked that the similarity numbers come from normalized embeddings and match the demo's `logits_per_image`.
- Checked the outputs of the run: the top and bottom image grids, the per-class score table, the confusion matrix, and the misclassified examples, and whether the written findings agree with them.
- Reviewed the final notebook and the supporting files and made sure the numbers in the findings match the tables.

## Question 2

Give one specific thing it produced that was wrong, such as an error, incorrect output, or implementation issue.

**Response:** The first version of the import cell imported `sklearn` before `torch`. On my Mac (M1, Anaconda Python 3.13, torch 2.14) that crashed the Jupyter kernel as soon as CLIP ran its first forward pass on the MPS device. nbconvert only reported `DeadKernelError: Kernel died`, with no Python error.

## Question 3

How did you find out? What did the failure look like?

**Response:** The first smoke test stopped after about 16 seconds with "Kernel died". I ran the code cells one by one as a plain script with `python -X faulthandler`. It printed `Fatal Python error: Segmentation fault` from background threads in the cell that compares my helper functions with the demo's `model(**inputs)` call. The same CLIP call worked fine in a small separate script, so the model itself wasn't the problem. Removing the import lines one at a time showed that the crash only happened when `import sklearn` came before `import torch`. That points to two different OpenMP runtimes being loaded (scikit-learn's and PyTorch's), which is a known problem on macOS.

## Question 4

What did you change, and why does your version work?

**Response:** I moved `torch`, `torchvision` and `transformers` to the top of the import cell, with `sklearn` after them, and added a comment explaining why the order matters. With torch imported first, its OpenMP runtime is the one that gets used, and scikit-learn uses the same runtime instead of loading a second copy. After that change the smoke test ran through without crashing, and so did the full run. The code itself didn't change, only the import order, so the results are the same as they would have been without the crash. The problem and the fix are also recorded in `RUN_LOG.txt`.
