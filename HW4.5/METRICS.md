# DATA 266 HW4.a + HW4.5 Metrics

Status: **the RTX 4090 run is pending.** Only the local smoke test has been run so far (a CPU check on
48/16/24 rows), and none of its numbers are used here. Every cell marked "Pending" will be filled from
`results/gpu/hw4a_tables.md`, `results/gpu/hw45_tables.md` and `results/gpu/hw45_summary.json` after
the lab run.

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
| SHA-256 of the selected rows | `24886c35fad68ff9...` (full hash in `results/hw4a_split.json`) |
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
| Batch size / epochs | 32 / 4 |
| Gradient clipping | 1.0 |
| Precision | FP32 |
| Model selection | epoch with the lowest validation loss |

## HW4.a Results

| Metric | RTX 4090 |
| --- | --- |
| Per-epoch train loss / val loss / val accuracy | Pending |
| Epoch where validation stopped improving | Pending |
| Test accuracy | Pending |
| Worst class (lowest F1) | Pending |
| Peak GPU memory, allocated / reserved (GB) | Pending |
| Training wall-clock (training steps / incl. validation) | Pending |

### Table HW4.1 — Serving comparison

Model `Qwen/Qwen2.5-0.5B-Instruct`, bf16, 128 prompts, 64 generated tokens per request, 2 warm-up calls, 2 timed repeats.

| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |
| --- | ---: | ---: | ---: |
| HF pipeline, batch 1 | Pending | Pending | Pending |
| vLLM, batch 1 | Pending | Pending | Pending |
| vLLM, batch 8 | Pending | Pending | Pending |
| vLLM, batch 32 | Pending | Pending | Pending |

## Part A — Batch Scaling (RTX 4090)

FP32 HW4 settings. Each batch size gets a fresh model, 3 untimed warm-up steps, then one timed epoch over 3,500 rows. Doubling continues until a real OOM or the 1024 safety limit.

| Batch size | Samples/s | Time per epoch (s) | Optimizer steps | Peak alloc (GB) | Peak reserved (GB) | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pending | | | | | | |

| Summary | Value |
| --- | --- |
| Largest batch that ran | Pending |
| First OOM batch | Pending |
| Throughput stops improving after | Pending (rule: the next doubling adds less than 5% samples/s) |
| Second card | N/A — not run, only one RTX 4090 was available |
| Max-batch ratio vs VRAM ratio | N/A — needs two cards |

## Part B — Memory Techniques (Table HW4.5.1, RTX 4090)

Each technique runs on its own at the HW4 batch size (32) for the full 4-epoch fine-tune. Max batch comes from a separate doubling probe (8 up to 4096).

| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy |
| --- | ---: | ---: | ---: | ---: |
| Baseline (HW4 settings) | Pending | Pending | Pending | Pending |
| Mixed precision (bf16 autocast) | Pending | Pending | Pending | Pending |
| Gradient checkpointing | Pending | Pending | Pending | Pending |
| Gradient accumulation (8 x 4 = effective 32) | Pending | Pending | Pending | Pending |
| 8-bit optimiser (bitsandbytes AdamW8bit) | Pending | Pending | Pending | Pending |
| Best combination | Pending | Pending | Pending | Pending |

### Technique Ranking

Score = (baseline peak allocated GB − technique peak allocated GB) ÷ (baseline samples/s − technique samples/s).
If a technique saves memory with no throughput loss, its score is infinite and it ranks first. Techniques
that save no memory, or did not run, are not ranked.

| Rank | Technique | Memory saved (GB) | Throughput lost (samp/s) | Score (GB per samp/s) |
| --- | --- | ---: | ---: | ---: |
| Pending | | | | |

## Part C — Two GPUs

Part C skipped (optional). The instructor's HW4.5 clarification made Part C optional. Only one RTX 4090
was reserved, so no DDP run, communication overhead, interconnect bandwidth or scaling efficiency was
measured.

## Part D — Serving on the RTX 4090

| Configuration | Tokens/sec | Requests/sec | p50 (ms) | p95 (ms) | Total requests | Total tokens | vs HW4.a tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HF pipeline, batch 1 | Pending | | | | | | |
| vLLM, batch 1 | Pending | | | | | | |
| vLLM, batch 8 | Pending | | | | | | |
| vLLM, batch 32 | Pending | | | | | | |

Batching: the HF pipeline gets one request per call. vLLM is batched natively: a list of prompts is
passed to `LLM.generate`, and each request in a batch is given the batch's wall time as its latency.
The second card is N/A (not run).

### KV Cache vs Context Length (batch 8)

Qwen2.5-0.5B: 24 layers × 2 KV heads × head dim 64, so the KV cache takes 12,288 bytes per token in bf16.

| Context | HF peak above weights (GB) | Theoretical KV (GB) | HF latency (ms) | vLLM latency (ms) | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| Pending | | | | | |

| Summary | Value |
| --- | --- |
| Longest context completed at batch 8 | Pending |
| vLLM KV pool (tokens / GB) | Pending |

## Part E — GPU-Hours

| Quantity | Value |
| --- | --- |
| One HW4 fine-tune (4 epochs × 3,500 rows, incl. validation) | Pending: wall-clock seconds ÷ 3600 × 1 GPU |
| Estimate for the full 45,615-row TweetEval train split (not run) | Pending: 4 × 45,615 ÷ samples/s ÷ 3600 |

## HW4.5 Summary (RTX 4090)

This is the HW4.5 summary only. It is not an HW5 baseline.

| Category | Configuration | Value |
| --- | --- | --- |
| Best peak memory | Pending | |
| Best throughput | Pending | |
| Best wall-clock | Pending | |
| Recommended configuration for a full fine-tune | Pending | |
