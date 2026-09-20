# DATA 266 HW4 Metrics

The numbers below are from the Google Colab GPU run of HW4_Mini_GPT.ipynb (see RUN_LOG.txt
for the full console output and how this run compares to the earlier local CPU run I used
while building the notebook).

## Personal Parameters

| Field | Value | HW4 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | Used for random, numpy, and torch (CPU and CUDA generators), applied before the model is created and again before every sampling call |
| SLICE | 486 | Reported only, not used in HW4 |
| HP_ID | 0 | Reported only, HW4 has no HP_ID arm |
| CLS_A | 6 | Reported only, not used in HW4 |
| CLS_B | 0 | Reported only, not used in HW4 |

## Part 1: Data Preparation

| Setting | Value |
| --- | --- |
| Dataset | shakespeare.txt, the Week 4 excerpt from Romeo and Juliet, loaded as given |
| Total characters | 1,738 |
| Tokenization | Character level, using char_to_idx and idx_to_char |
| Vocabulary size | 49 |
| Sequence length | 128 |
| Sliding window | input is tokens t through t+127, target is the same window shifted one character to the right |
| Number of training windows | 1,609 |
| Batch size | 32 |

I checked that the target really is the input shifted by one character with a direct tensor comparison in the notebook, and printed two example windows decoded back to text to see it directly.

## Part 2: GPT Architecture

None of the disallowed modules were used anywhere in the notebook. No nn.MultiheadAttention, no nn.Transformer, and no HuggingFace transformer classes. The attention, decoder block, and full model are all built from nn.Linear, nn.LayerNorm, nn.GELU, and nn.Embedding.

| Setting | Value |
| --- | --- |
| Hidden dimension | 128 |
| Number of attention heads | 4 |
| Head dimension | 32 (128 divided evenly by 4) |
| Number of decoder blocks | 4 |
| Feed forward network | Linear from 128 to 512, GELU, Linear back to 128 |
| Total parameter count | 822,321 |
| Model output shape on a real batch | (32, 128, 49), matching batch size by sequence length by vocab size |

### Causal mask check

| Check | Result |
| --- | --- |
| Attention row sums (should all be about 1) | minimum 0.9999998212, maximum 1.0000001192 |
| Total attention weight placed on future positions | 0.0 |
| Hidden dimension divisible by head count | yes, 128 mod 4 is 0 |

These were checked with actual assertions in the notebook, not just eyeballed. Every row of the attention matrix sums to about 1 like a proper softmax should, and not a single unit of attention weight leaks onto a future token.

## Part 3: Training

| Setting | Value |
| --- | --- |
| Optimizer | Adam with a learning rate of 3e-4 |
| Loss | cross entropy between the model's next character logits and the shifted targets |
| Epochs | 6 |
| Device | cuda, on Google Colab |
| Checkpoint | checkpoints/mini_gpt_seed_9486.pt, saved with torch.save right after training |

### Training loss by epoch

| Epoch | Average loss |
| ---: | ---: |
| 1 | 2.9655 |
| 2 | 2.3758 |
| 3 | 2.1977 |
| 4 | 2.0952 |
| 5 | 1.9809 |
| 6 | 1.8303 |

Loss went down every single epoch, and the curve is in figures/training_loss.png. Interestingly, these average loss values came out identical to four decimal places on this GPU run and on the earlier CPU run. That's not something I'd expect in general (CPU and GPU floating point math doesn't add up in exactly the same order), but with a model this small and only 300 total training steps, the tiny numerical differences between the two never grew large enough to shift the rounded loss.

One thing worth noting: the checkpoint file itself was saved on the Colab instance during that session but did not come back with the notebook when I pulled it down, since Colab's local disk isn't part of the git repo. I'll need to download it from Colab (or re-run the notebook) and add it to the repo separately before final submission.

## Part 4: Sampling

All samples start from the prompt "ROMEO:" and generate 300 new characters. I reset the seed to 9486 right before every one of the six generation calls.

### What each method actually produced

Greedy decoding came out exactly the same on the GPU as it did on my earlier CPU run: it gets stuck almost right away, repeating fragments like "nor nor nathe" and "O nor nathame" over and over. Since argmax has no randomness involved, this makes sense as long as the model's top choice at each step is a clear winner rather than a close call, which it seems to be here.

The three temperature runs and the two top-k runs, on the other hand, came out differently from the CPU versions, even with the same SEED. The reason is that torch.multinomial draws from the CUDA random generator when the logits live on the GPU, and from the CPU generator when they don't. Those are two separate random streams, so setting the same seed value does not make sampling identical across devices, only reproducible on the same device. This is a real, identifiable source of non-determinism (not a bug), and it only shows up in the sampling methods that actually use randomness, not in greedy decoding.

| Method | Setting | What I saw |
| --- | --- | --- |
| Greedy | none | falls into a repeating loop within the first few dozen characters |
| Temperature | 0.5 | still loops a fair amount but with more variety than greedy, some readable word starts |
| Temperature | 1.0 | noticeably less repetitive, more unusual letter combinations and odd capitalization |
| Temperature | 1.5 | the most scrambled of the three, shortest recognizable word fragments |
| Top-k | 5 | close to greedy, restricted enough that it loops in similar ways |
| Top-k | 20 | clearly more varied than k=5, similar overall feel to temperature 1.0 |

Full text for all six samples is in HW4_Mini_GPT.ipynb and findings.pdf. Based on these samples, the most coherent output came from the lower temperature and lower top-k settings, and the most varied came from temperature 1.5, with top-k 20 not far behind.

## Part 5: Failure and Error Analysis

The clearest failure mode across this whole run is greedy decoding's tendency to loop, which shows up identically whether the model runs on CPU or GPU, since it is a property of the trained weights and not of the hardware. The model was only trained for 6 epochs on a very small, repetitive excerpt, so it hasn't learned enough to break out of these loops or produce fluent text at any decoding setting. This is discussed in the notebook itself and in findings.pdf. A separate six item failure gallery was not included, since this assignment does not say one is required.
