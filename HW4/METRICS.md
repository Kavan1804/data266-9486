# DATA 266 HW4 Metrics

**Note:** the numbers below come from the local CPU run logged in `RUN_LOG.txt` (Step 4),
used while building and verifying the notebook. This assignment will be re-run on Google
Colab with a GPU runtime before final submission; `RUN_LOG.txt`, this file, `figures/`,
and `findings.pdf` should be regenerated from that run's console output.

## Personal Parameters

| Field | Value | HW4 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | Reproducibility: `random`, `numpy`, `torch` (incl. CUDA generator) seeding throughout the notebook, re-applied before model creation and before every sampling call |
| SLICE | 486 | Reported only; no HW4 mapping |
| HP_ID | 0 | Reported only; no HW4 mapping (HW4 has no HP_ID arm) |
| CLS_A | 6 | Reported only; no HW4 mapping |
| CLS_B | 0 | Reported only; no HW4 mapping |

## Part 1 — Data Preparation

| Setting | Value |
| --- | --- |
| Dataset | `shakespeare.txt` (Week 4 dataset, given — not downloaded or substituted) |
| Total characters | 1,738 |
| Tokenization | Character-level; `char_to_idx` / `idx_to_char` |
| Vocabulary size | 49 |
| Sequence length (`seq_len`) | 128 |
| Sliding-window input/target | `x = tokens[t : t+128]`, `y = tokens[t+1 : t+129]` (shifted by exactly 1 character, verified with `torch.equal`) |
| Number of training windows | 1,609 (`len(text) - seq_len - 1`) |
| Batch size | 32 |

## Part 2 — GPT Architecture

| Item | Status |
| --- | --- |
| Forbidden components used | None — no `nn.MultiheadAttention`, `nn.Transformer`, or HuggingFace transformer classes anywhere. Only `nn.Linear`, `nn.LayerNorm`, `nn.GELU`, `nn.Embedding`, and explicit tensor ops. |
| Attention | Manual multi-head masked self-attention; separate `nn.Linear` Q/K/V projections; hidden dimension split into heads via `.view(...).transpose(1, 2)`; causal mask applied with `masked_fill` before softmax |
| Hidden dimension (`d_model`) | 128 |
| Number of heads (`n_heads`) | 4 |
| Head dimension | 32 (`128 / 4`, exact — divisibility asserted in the notebook) |
| Number of decoder blocks (`n_layers`) | 4 |
| Feed-forward network | `Linear(128 → 512) → GELU → Linear(512 → 128)` |
| Total parameter count | 822,321 |
| Model output shape (on a real training batch) | `(32, 128, 49)` = `[batch_size, seq_len, vocab_size]` |

### Causal Mask Verification

| Check | Result |
| --- | --- |
| Softmax row sums | min = 0.9999998807907104, max = 1.0000001192092896 |
| Total attention weight on future positions (`j > i`) | 0.0 |
| Hidden dimension divisible by head count | Passed (`128 % 4 == 0`) |
| Model output shape matches `[B, T, vocab_size]` | Passed |

## Part 3 — Training

| Setting | Value |
| --- | --- |
| Optimizer | Adam, `lr = 3e-4` |
| Loss | Cross-entropy over next-character logits, `[B, T, vocab_size]` vs. shifted `[B, T]` targets |
| Epochs | 6 (within the required 5–8 range, asserted in the notebook) |
| Device (this run) | CPU (`torch.cuda.is_available()` was `False`; auto-selects CUDA when run on a GPU, e.g. Colab) |
| Reproducibility | `SEED = 9486`, re-applied immediately before model creation |
| Checkpoint | `checkpoints/mini_gpt_seed_9486.pt` (`torch.save(model.state_dict(), ...)`) — cell added after this run; not yet produced, will be written the next time the notebook executes top to bottom |
| Exact command that trained it | Running `HW4_Mini_GPT.ipynb` top to bottom (no CLI args); training loop: `for epoch in range(1, epochs + 1): ... optimizer.step()` over the full `DataLoader(dataset, batch_size=32, shuffle=True)` |

### Training Loss per Epoch

| Epoch | Avg. Loss |
| ---: | ---: |
| 1 | 2.9655 |
| 2 | 2.3758 |
| 3 | 2.1977 |
| 4 | 2.0952 |
| 5 | 1.9809 |
| 6 | 1.8303 |

Loss decreased every epoch; see `figures/training_loss.png` for the plotted curve.

## Part 4 — Sampling

| Setting | Value |
| --- | --- |
| Prompt | `"ROMEO:"` |
| Generated length | 300 new characters per sample |
| Decoding methods | Greedy; temperature sampling at `T ∈ {0.5, 1.0, 1.5}`; top-k sampling at `k ∈ {5, 20}` (with `T = 1.0`) |
| Reproducibility | `SEED = 9486` re-applied immediately before each of the 6 generation calls |

### Decoding Method Summary (from the actual generated samples)

| Method | Setting | Observed behavior |
| --- | --- | --- |
| Greedy | — | Fell into a tight repetition loop almost immediately (`"nor nor nathe"`, `"O nor nathame"`) — no randomness, fully deterministic |
| Temperature | `T=0.5` | Least random of the three; still some repeated letters, closest to greedy |
| Temperature | `T=1.0` | More erratic capitalization and punctuation than `T=0.5` |
| Temperature | `T=1.5` | Most chaotic of all six samples; rare/odd characters appear mid-word |
| Top-k | `k=5` | Still visibly loops (`"bullllll'de"`, `"bureeort"`), close to greedy's failure mode |
| Top-k | `k=20` | Noticeably more varied than `k=5`; closely resembles the `T=1.0` sample |

Most coherent output (relative, given all six samples are undertrained): `T=0.5` and `k=5`.
Most diverse output: `T=1.5`, with `k=20` second. Full generated text for all six samples is in
`HW4_Mini_GPT.ipynb` and `findings.pdf`.

## Part 5 — Failure / Error Analysis

The assignment's main observed failure mode is greedy decoding's repetition loop on this
undertrained model — a well-known consequence of always taking `argmax` once the model becomes
confident about a short high-probability cycle. This is discussed directly in the notebook's
Part 5 analysis and in `findings.pdf`; no separate six-item failure gallery is included, since
this assignment does not state "Failure gallery required."
