# DATA 266 HW4.a + HW4.5 Metrics

All numbers below are from the GPU runs on a **Tesla T4** in Google Colab:

| Run | Time (UTC) | GPU UUID |
| --- | --- | --- |
| HW4.a | 2026-11-02 23:27:57 to 23:48:50 | `GPU-585cb49b-329b-b3d3-3094-9d792639bfd4` |
| HW4.5 | 2026-11-03 00:08:29 to 01:31:24 | `GPU-d44fecb0-9de2-0bc4-8f0f-edca5b8c66f2` |

Each Colab session got its own machine, so the two notebooks ran on two different physical T4s (same model). Raw files are in
`results/gpu/`, figures in `figures/gpu/`. The smoke-test numbers in `results/smoke/` are not used anywhere here.

## Hardware and Software

| Item | Value |
| --- | --- |
| GPU | NVIDIA Tesla T4, 14.56 GiB usable, driver 580.82.07 |
| Python / PyTorch | 3.13.16 / 2.14.0+cu130 |
| Transformers / vLLM / bitsandbytes | 5.20.2 / 0.33.0 / 0.50.2 |
| Native bf16 | No (T4). Serving and mixed precision use fp16 instead of bf16 |
| GPUs detected | 1 |

## Personal Parameters

| Field | Value | Use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | Python, NumPy and PyTorch seeds, the 70/15/15 split, the training data order, and set again before every BERT model is created |
| SLICE | 486 | First dataset row used |
| HP_ID | 0 | Reported only; HW4.5 defines no HP_ID arm |
| CLS_A | 6 | Reported only |
| CLS_B | 0 | Reported only |

## Dataset and Row Range

| Field | Value |
| --- | --- |
| Dataset | TweetEval sentiment (`cardiffnlp/tweet_eval`, config `sentiment`, split `train`) |
| Dataset revision | `b3a375baf0f409c77e6bc7aa35102b7b3534f8be` |
| Rows available | 45,615 |
| Rows used | 486 to 5485 inclusive (5,000 consecutive rows) |
| Classes in the slice | negative 772 (15.4%), neutral 2,288 (45.8%), positive 1,940 (38.8%) |
| SHA-256 of the selected rows | `24886c35fad68ff9...` (full hash in `results/hw4a_split.json`; same on the T4 run as in the smoke test) |
| Tweets longer than 128 tokens | 0 of 5,000 (longest is 89 tokens) |

HW4.a and HW4.5 use exactly the same rows and split. HW4.5 recomputes both and asserts they match the saved `results/hw4a_split.json`.

## Split Sizes

| Split | Rows |
| --- | ---: |
| Train | 3,500 |
| Validation | 750 |
| Test | 750 |

70/15/15, stratified by label, `random_state = SEED`.

## Baseline Settings (HW4.a fine-tune)

| Setting | Value |
| --- | --- |
| Model | `google-bert/bert-base-uncased` (`BertForSequenceClassification`, 3 labels, 109,484,547 parameters) |
| Tokenizer | `BertTokenizer`, max length 128, padded to max length |
| Optimizer | `torch.optim.AdamW`, lr 2e-5, eps 1e-8, weight decay 0.01 |
| Schedule | linear decay, 0 warm-up steps |
| Batch size / epochs | 32 / 4 (110 steps per epoch, 440 optimizer steps) |
| Gradient clipping | 1.0 |
| Precision | FP32 |
| Model selection | epoch with the lowest validation loss |

## HW4.a Results

### Per-epoch training

| Epoch | Train loss | Val loss | Val accuracy | Epoch training time (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9412 | 0.8019 | 0.6520 | 66.0 |
| 2 | 0.6903 | **0.6946** | 0.6973 | 71.1 |
| 3 | 0.5283 | 0.6993 | 0.7013 | 76.6 |
| 4 | 0.4180 | 0.7165 | 0.7027 | 76.4 |

**Where validation stopped improving:** validation loss was lowest after epoch 2 (0.6946) and went up in epochs 3 and 4,
while the training loss kept falling from 0.69 to 0.42. That is the model starting to overfit. Validation accuracy still
crept up by 0.5 points (0.697 to 0.703), so the extra epochs mostly made the model more confident, not more correct. I trained
the full 4 epochs, which cost about 164 s more (153 s of training plus 11 s of validation). That is roughly half of the
313 s run. The checkpoint kept is the epoch-2 one, so the test numbers are not hurt; the cost was only compute.

### Test set (epoch-2 weights)

Test accuracy **0.6907** (518 of 750), test loss 0.6812.

Confusion matrix (rows = true class, columns = predicted):

| | pred negative | pred neutral | pred positive |
| --- | ---: | ---: | ---: |
| **true negative** | 66 | 40 | 10 |
| **true neutral** | 39 | 234 | 70 |
| **true positive** | 16 | 57 | 218 |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.5455 | 0.5690 | 0.5570 | 116 |
| neutral | 0.7069 | 0.6822 | 0.6944 | 343 |
| positive | 0.7315 | 0.7491 | 0.7402 | 291 |

**Worst class: negative** (F1 0.557). It is also the smallest class (15.4% of the slice), and most of its errors go to
neutral (40 of the 50 mistakes).

### Five misclassified negative tweets

| Row | Predicted (prob.) | Tweet | Diagnosis |
| ---: | --- | --- | --- |
| 559 | neutral (0.52) | BNP opposes Companies Act change: The BNP on Tuesday asked the government to drop the move to change Companies A... | The tweet reports that the BNP opposes a proposed change to the Companies Act, but it does not contain explicit negative sentiment words. Its negativity is implicit in the political opposition and the demand to stop the policy. Because the statement is written like a factual news headline, the model likely interpreted it as neutral. |
| 904 | positive (0.62) | it\u2019s a sunday\u002c i think in the spirit of douglas adams and feeling like shit i will have a second bath | The phrase “feeling like shit” indicates a negative emotional state, but the tweet also describes taking a bath in a somewhat humorous and self-care-oriented way. This combination creates mixed sentiment. The informal wording, possible text encoding issues, and sarcastic tone may have caused the model to focus on the lighter interpretation and predict positive. |
| 1105 | positive (0.38) | Chris brown is going to be in atx on Wednesday &amp; I'm so so sad I can't see my baby :/ | The tweet expresses sadness because the user cannot see Chris Brown, using the phrases “so so sad” and “can’t see.” However, the affectionate phrase “my baby” introduces a positive emotional signal. Since the model’s confidence was low, it appears to have been uncertain between the positive language and the clearly negative disappointment. This is an example of mixed sentiment and competing emotional cues. |
| 1110 | neutral (0.84) | Dr. going to see lot of people of stomach pain due to yakub's hanging tomorrow @user @user @user @user | The tweet mentions a doctor seeing many people with stomach pain because of an upcoming hanging, but the intended sentiment is difficult to determine without additional context. It may be sarcastic, ironic, or referring to a disturbing event in an indirect way. The unusual wording and lack of standard sentiment expressions likely caused the model to classify it as neutral. The high confidence suggests that the model did not recognize the implied negative meaning. |
| 1118 | neutral (0.81) | @user @user That's Ashley in the 1st pic not Briana but the pregnancy is still fake af | The tweet includes negative language such as “fake af,” accusing someone of pretending to be pregnant. However, the message also contains multiple usernames, a correction about the person shown in a picture, and informal slang. The tokenizer may not have represented “fake af” effectively, especially in combination with the noisy social-media format. As a result, the model likely treated the tweet as neutral despite its strongly negative and accusatory tone. |

**Overall diagnosis:** The errors suggest that the model had difficulty identifying implicit negativity, sarcasm, slang, and mixed emotional signals. Forty of the fifty misclassified negative tweets were predicted as neutral, indicating that the model frequently defaulted to the majority neutral class when negative sentiment was expressed indirectly or through informal social-media language. All 50 are in `results/gpu/hw4a_worst_class_misclassified.csv`.

### Training memory and time (kept as the HW4.5 baseline)

| Metric | Value |
| --- | ---: |
| Peak GPU memory, allocated / reserved | 3.78 GB / 4.05 GB |
| Training wall-clock (training steps only) | 290.1 s |
| Training wall-clock including per-epoch validation | 312.8 s |
| Training throughput | 48.3 samples/s |

### Table HW4.1 — Serving comparison

Model `Qwen/Qwen2.5-0.5B-Instruct`, **fp16** (the T4 has no native bf16 and vLLM refuses bf16 on it), my own prompt asking
what the writer of a tweet is feeling and why, 128 prompts from the test split, exactly 64 generated tokens per request,
2 warm-up calls, 2 timed repeats (256 requests per configuration). vLLM used its Triton attention backend, because
FlashAttention-2 needs a newer GPU than the T4.

| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) | Requests/sec |
| --- | ---: | ---: | ---: | ---: |
| HF pipeline, batch 1 | 29.5 | 1967.6 | 2724.9 | 0.46 |
| vLLM, batch 1 | 164.4 | 389.8 | 399.8 | 2.57 |
| vLLM, batch 8 | 1174.5 | 435.0 | 445.8 | 18.35 |
| vLLM, batch 32 | 3214.0 | 634.6 | 651.4 | 50.22 |

**Serving answer.** Compared with its own batch-1 latency, vLLM's per-request latency gets worse from batch 8 on (p50 390 ms
to 435 ms, then 635 ms at batch 32) while total throughput is still improving strongly (164 to 1,175 to 3,214 tokens/s).
Against the HF pipeline baseline (p50 1,968 ms), vLLM is never worse at any batch size I tested; even batch 32 is about
3x faster per request. For an interactive application the per-request latency (especially p95) is the number that matters,
because a user waits for one answer; total throughput matters for offline or bulk jobs.

## Part A — Batch Scaling (Tesla T4)

HW4 fine-tune unchanged (FP32). Each batch size: fresh model, 3 untimed warm-up steps, then one timed epoch over 3,500 rows.

| Batch size | Samples/s | Time per epoch (s) | Optimizer steps | Peak alloc (GB) | Peak reserved (GB) | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 12.5 | 280.0 | 3500 | 2.11 | 2.32 | ok |
| 2 | 21.8 | 160.3 | 1750 | 2.11 | 2.40 | ok |
| 4 | 31.0 | 113.0 | 875 | 2.11 | 2.31 | ok |
| 8 | 37.8 | 92.5 | 438 | 2.13 | 2.34 | ok |
| 16 | 43.5 | 80.5 | 219 | 2.52 | 2.85 | ok |
| 32 | 47.9 | 73.0 | 110 | 3.78 | 4.05 | ok |
| 64 | 49.1 | 71.2 | 55 | 6.15 | 6.33 | ok |
| 128 | 49.7 | 70.4 | 28 | 11.00 | 11.13 | ok |
| 256 | — | — | — | — | — | OOM (tried to allocate 384 MiB with 14.39 GiB in use) |

| Summary | Value |
| --- | --- |
| Largest batch that ran | 128 |
| First OOM batch | 256 |
| Throughput stops improving after | **batch 32**: doubling to 64 adds only 2.5% samples/s, and 128 adds another 1.2% (rule: less than 5% per doubling) |
| Second card | N/A — not run, only one GPU was available |
| Max-batch ratio vs VRAM ratio | N/A — needs two cards (explanation below from this card's numbers) |

**Why the two ratios would not be equal.** Peak memory is not proportional to batch size. Up to batch 8 it stays flat at
about 2.1 GB: that part is fixed (FP32 weights, gradients and the two Adam states are 4 × 0.41 GB = 1.63 GB, plus workspace),
and every card pays it before any activations. Above that, each extra sample adds about 0.075 GB of activations. So the max
batch is set by (VRAM − CUDA context − fixed cost) ÷ per-sample cost, which grows faster than VRAM itself, and the
power-of-two steps make it coarser still: extending the line through batches 64 and 128 to the T4's 14.56 GiB gives
roughly 170 samples (an estimate, not measured), but the largest power of two below that is 128.

Figures: `figures/gpu/batch_size_vs_throughput.png`, `figures/gpu/batch_size_vs_memory.png`.

## Part B — Memory Techniques (Table HW4.5.1, Tesla T4)

Each technique is applied on its own at the HW4 batch size (32) for the full 4-epoch fine-tune (440 optimizer steps each).
Max batch comes from a separate doubling probe (8 up to 4096). Mixed precision on the T4 is fp16 autocast with
`torch.amp.GradScaler`.

| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy (test) |
| --- | ---: | ---: | ---: | ---: |
| Baseline (HW4 settings) | 128 | 47.8 | 3.78 | 0.6907 |
| Mixed precision (fp16 + GradScaler) | 256 | 148.4 | 3.06 | 0.6907 |
| Gradient checkpointing | 512 | 36.1 | 2.13 | 0.6907 |
| Gradient accumulation (8 × 4 = effective 32) | 128 physical / 512 effective | 43.9 | 2.35 | 0.6867 |
| 8-bit optimiser (bitsandbytes AdamW8bit) | 128 | 48.1 | 3.14 | 0.6867 |
| Best combination (mixed precision + gradient accumulation + 8-bit optimiser) | 256 physical / 1024 effective | 116.3 | 1.70 | 0.6773 |

- **Mixed precision:** 3.1x the baseline throughput (148.4 vs 47.8 samples/s) because the T4's tensor cores only accelerate
  fp16. Peak memory fell by 0.72 GB (19%) and the max batch doubled to 256. Test accuracy was the same, 0.6907; the
  validation losses differ from the baseline at every epoch, so this is a genuinely different run that happens to make the
  same number of correct predictions.
- **Gradient checkpointing:** the lowest single-technique memory (2.13 GB, −44%) and a 4x larger max batch (512), but
  throughput dropped by **24.6%** (47.8 to 36.1 samples/s) because the encoder activations are recomputed in the backward
  pass. Its training history is identical to the baseline, as expected, since the gradient check before Part B showed the
  same gradients (max difference 0.0).
- **Gradient accumulation:** physical batch 8, 4 micro-batches per step, so the effective batch is 32 and the number of
  optimizer steps is the same 440 as the baseline. Time to those 440 steps: 318.7 s vs 292.7 s for the baseline (+8.9%),
  or 340.5 s vs 314.4 s including validation. Accuracy 0.6867 (−0.4 points; dropout masks differ from one pass of 32). The
  probe reached a physical batch of 128, i.e. an effective batch of 512.
- **8-bit optimiser:** 0.64 GB (17%) less peak memory from the smaller Adam states, with no throughput change
  (48.1 vs 47.8 samples/s, which is within noise). The max batch stayed at 128 because at large batches activations, not
  optimizer state, are what run out. Accuracy 0.6867.
- **Best combination:** the rule picked before the run (saves memory, loses at most 10% throughput, accuracy within 1 point,
  each judged on its own) selected mixed precision, gradient accumulation and the 8-bit optimiser. Together they gave the
  lowest peak memory of all (1.70 GB, −55%), but throughput was 22% below mixed precision alone because of the small
  physical batch of 8, and test accuracy dropped 1.3 points to 0.6773. Each technique passed the rule alone, but the
  combination did not stay within the 1-point accuracy limit.

### Technique Ranking

Score = (baseline peak allocated GB − technique peak allocated GB) ÷ (baseline samples/s − technique samples/s).
A technique that saves memory with no throughput loss gets an infinite score and ranks first.

| Rank | Technique | Memory saved (GB) | Throughput lost (samp/s) | Score (GB per samp/s) |
| --- | --- | ---: | ---: | ---: |
| 1 | Mixed precision (fp16 + GradScaler) | 0.72 | −100.5 (gained) | ∞ |
| 2 | 8-bit optimiser | 0.64 | −0.2 (no change) | ∞ |
| 3 | Gradient accumulation | 1.43 | 3.9 | 0.37 |
| 4 | Gradient checkpointing | 1.65 | 11.8 | 0.14 |

The 8-bit optimiser's infinite score comes from a 0.2 samples/s difference, which is noise; in practice it is "free memory,
no speed change".

**Which first, and which not to bother with on a 32 GB card.** I would apply mixed precision first: it is the only technique
that saves memory and makes training much faster, with no accuracy loss. On a 32 GB card I would not bother with gradient
checkpointing (or gradient accumulation): the HW4 fine-tune peaks at 3.8 GB, so there is no memory problem to solve, and
checkpointing would cost about a quarter of the throughput for nothing.

Figure: `figures/gpu/technique_comparison.png`.

## Part C — Two GPUs

Part C skipped (optional). The instructor's HW4.5 clarification made Part C optional. Only one GPU was used (one Tesla T4
detected), so no DDP run, communication overhead, interconnect bandwidth or scaling efficiency was measured.

## Part D — Serving on the Tesla T4

Same model, prompts, token budget, warm-up and repeats as HW4.a, fp16.

| Configuration | Tokens/sec | Requests/sec | p50 (ms) | p95 (ms) | Total requests | Total tokens | vs HW4.a tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HF pipeline, batch 1 | 32.0 | 0.50 | 1903.4 | 2388.8 | 256 | 16,384 | +8.7% |
| vLLM, batch 1 | 161.9 | 2.53 | 394.0 | 406.5 | 256 | 16,384 | −1.5% |
| vLLM, batch 8 | 1191.2 | 18.61 | 429.1 | 438.8 | 256 | 16,384 | +1.4% |
| vLLM, batch 32 | 3284.7 | 51.32 | 622.6 | 638.0 | 256 | 16,384 | +2.2% |

Batching: the HF pipeline gets one request per call. vLLM is batched natively: a list of prompts is passed to
`LLM.generate`, and each request in a batch is given the batch's wall time as its latency. The second card is N/A (not run).

**Comparison with HW4.a.** The vLLM numbers are within ±2.2% of HW4.a and the HF pipeline is 8.7% faster. Nothing
substantial changed: both runs used the same GPU model (T4), the same precision (fp16) and the same vLLM version (0.33.0).
They did run on two different physical T4s (different UUIDs). The HF pipeline varies more because each generated token is a
separate Python-level step (its p95 also moved, 2,725 to 2,389 ms), so I read these differences as run-to-run variation
rather than a measurement flaw.

### KV Cache vs Context Length (batch 8)

Qwen2.5-0.5B: 24 layers × 2 KV heads × head dim 64, so the KV cache takes 12,288 bytes per token in fp16.
HF: 8 random-token sequences per context, 16 new tokens, model weights 0.99 GB.

| Context | HF peak above weights (GB) | Theoretical KV (GB) | HF latency (ms) | vLLM latency (ms) | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 0.19 | 0.05 | 702 | 423 | ok / ok |
| 1,024 | 0.37 | 0.10 | 914 | 1,112 | ok / ok |
| 2,048 | 0.75 | 0.19 | 1,476 | 3,112 | ok / ok |
| 4,096 | 1.49 | 0.38 | 2,893 | 10,954 | ok / ok |
| 8,192 | 2.99 | 0.75 | 6,450 | 40,447 | ok / ok |
| 16,384 | 5.97 | 1.50 | 16,543 | 154,570 | ok / ok |
| 32,752 | — | 3.00 | — | 605,923 | HF OOM (tried to allocate 2.38 GiB) / vLLM ok |

The HF memory grows linearly with context but is about 4x the pure KV size, because the prefill activations for 8 long
sequences dominate.

| Summary | Value |
| --- | --- |
| Longest context completed at batch 8, HF | **16,384 tokens** (OOM at 32,752) |
| Longest context completed at batch 8, vLLM | **32,752 tokens**, the model's limit (32,768 positions − 16 new tokens) |
| vLLM KV pool | 57,982 blocks × 16 = 927,712 tokens = 10.62 GB |

vLLM serves twice the context because its KV pool is pre-allocated and paged, and prefill is chunked. On the T4 it is much
slower at long contexts than HF (155 s vs 17 s at 16,384), which I attribute to the Triton attention fallback vLLM has to
use on this GPU (FlashAttention-2 is not supported on the T4).

Figures: `figures/gpu/serving_latency_comparison.png`, `figures/gpu/serving_throughput_comparison.png`,
`figures/gpu/kv_cache_memory_vs_context.png`.

## Part E — GPU-Hours

| Quantity | Value |
| --- | --- |
| One HW4 fine-tune (4 epochs × 3,500 rows, including validation) | 314.4 s ÷ 3600 × 1 GPU = **0.087 GPU-hours** (HW4.5 baseline run; HW4.a measured 312.8 s = 0.087) |
| Same fine-tune with mixed precision | 100.9 s ÷ 3600 = 0.028 GPU-hours |
| Estimate for the full 45,615-row TweetEval train split (not run) | 4 × 45,615 ÷ 47.8 samples/s ÷ 3600 = **1.06 GPU-hours** FP32, or 4 × 45,615 ÷ 148.4 ÷ 3600 = 0.34 GPU-hours with mixed precision (training steps only) |

**Configuration I would use for a full fine-tune:** mixed precision alone (fp16 with GradScaler on a T4; bf16 on a card that
supports it) at the HW4 batch size of 32. It had the highest throughput (148.4 samples/s, 3.1x the baseline), the shortest
wall-clock (100.9 s including validation), the same test accuracy as the baseline (0.6907), and it needs only 3.06 GB, so it
fits on any lab card. I would add the 8-bit optimiser or gradient checkpointing only if memory became the limit, for example
with a larger model or longer sequences, since the combination with gradient accumulation lost accuracy and speed here.

## HW4.5 Summary (Tesla T4)

This is the HW4.5 summary only. It is not an HW5 baseline.

| Category | Configuration | Value |
| --- | --- | --- |
| Best peak memory | Mixed precision + gradient accumulation + 8-bit optimiser | 1.70 GB peak allocated (test accuracy 0.6773) |
| Best throughput | Mixed precision (fp16 + GradScaler) | 148.4 samples/s (test accuracy 0.6907) |
| Best wall-clock | Mixed precision (fp16 + GradScaler) | 100.9 s for the 4-epoch fine-tune including validation (test accuracy 0.6907) |
| Recommended configuration for a full fine-tune | Mixed precision (fp16 + GradScaler), batch 32 | see Part E |
