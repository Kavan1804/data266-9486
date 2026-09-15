# DATA 266 HW3 Metrics

## Personal Parameters

| Field | Value | HW3 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | Reproducibility: `random`, `numpy`, and `torch` seeding throughout the notebook |
| SLICE | 486 | Reported only; no HW3 mapping |
| HP_ID | 0 | Reported only; no HW3 mapping |
| CLS_A | 6 | Reported only; no HW3 mapping |
| CLS_B | 0 | Reported only; no HW3 mapping |

## Part 1 — Prompt Engineering

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Complete: 6 techniques x 2 named prompt examples = 12 prompts, each called through `get_completion()` (LangChain `ChatGoogleGenerativeAI`), reusing the `set_params`/`get_completion` pattern from `Prompt_Engineering.ipynb`. |
| Techniques implemented | Zero-Shot, Few-Shot, Chain-of-Thought, Zero-Shot CoT, Meta-Prompting, Tree of Thoughts |
| Structural validation | `nbformat.validate()` passed; every code cell parses with `ast.parse()`. |
| Execution | **Not executed.** Requires Google Colab (`google.colab.userdata`) and a configured `GOOGLE_API_KEY` secret, neither of which is available in this environment. Cells are clearly marked in the notebook as requiring Colab execution. This is a missing-credentials limitation, not an implementation defect. |
| Next step | Run Part 1 top-to-bottom in Colab with `GOOGLE_API_KEY` set; outputs will populate under each of the 12 prompt cells. |

## Part 2 — Self-Attention and Causal Masking

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Complete; single-head scaled dot-product self-attention from scratch, matching `Demo_3_Attention_Mechanism.ipynb`. |
| Execution | Executed locally end-to-end via `nbclient` (kernel: `python3`, `/opt/anaconda3/bin/python`, `torch 2.14.0`). No GPU/API dependency for this part. |
| Forbidden components used | None — no `nn.MultiheadAttention`, `nn.Transformer`, HuggingFace transformer classes, positional encoding, or multi-head attention. Only `nn.Embedding`, `nn.Linear`, matrix multiplication, softmax, and `F.cross_entropy`. |
| Reproducibility | `SEED = 9486` applied to `random`, `numpy`, and `torch` (re-applied immediately before model creation). |

### Dataset and Tokenization

| Setting | Value |
| --- | --- |
| Dataset text | Fixed 5-sentence assignment text ("Neural networks are powerful models...one token at a time.") |
| Tokenization | Word-level, `re.findall(r"[A-Za-z]+", text.lower())` |
| Num tokens | 46 |
| Vocabulary size | 39 |
| Autoregressive input/target | `x = ids[:-1]` (45 tokens), `y = ids[1:]` (45 tokens) |

### Model Configuration

| Setting | Value |
| --- | --- |
| Architecture | `SingleHeadSelfAttentionLM`: `nn.Embedding` + `Wq`/`Wk`/`Wv`/`Wo` (`nn.Linear`, no bias) + `lm_head` (`nn.Linear`, no bias) |
| `d_model` | 64 |
| `d_k` | 64 |
| Optimizer | Adam, `lr = 0.01` |
| Epochs | 600 |
| Loss | Cross-entropy over next-token logits |
| Training mask | Causal (decoder-style), `use_causal_mask=True` during training |
| Device | CPU |

### Learning-Rate Selection (pre-training check)

| Learning rate | 600-epoch outcome |
| ---: | --- |
| 0.005 | Unstable; final-10-epoch losses oscillate between 0.29 and 0.75 |
| 0.01 | Stable; converges and flattens at loss ≈ 0.031 |
| 0.02 | Unstable; oscillates around loss ≈ 0.61 |

`lr = 0.01` was selected for the final notebook based on this comparison.

### Training Progress

| Epoch | Loss |
| ---: | ---: |
| 100 | 0.0326 |
| 200 | 0.0310 |
| 300 | 0.0309 |
| 400 | 0.0309 |
| 500 | 0.0309 |
| 600 | 0.0309 |

### Attention Matrices (computed after training, full 46-token sequence)

| Configuration | Shape | Row-sum min | Row-sum max |
| --- | --- | ---: | ---: |
| Unmasked (Configuration 1) | `(1, 46, 46)` | 0.9999998807907104 | 1.000000238418579 |
| Causal-masked (Configuration 2) | `(1, 46, 46)` | 0.9999998807907104 | 1.000000238418579 |

Both configurations share the same trained embeddings and Q/K/V weights (see `HW3.ipynb` §2.11); the only
difference is whether the causal mask is applied at inference time.

### Causal Masking Verification

| Check | Result |
| --- | --- |
| Causal mask is exactly lower-triangular (`T=5` example and full `T=46`) | Passed (`torch.equal` against `torch.tril(...)`) |
| Attention matrix shape is `[1, T, T]` for both configurations | Passed |
| Every attention row sums to ≈ 1 (softmax correctness) | Passed, `atol=1e-4` |
| Total masked attention mass above the diagonal | `0.0` |
| Max single future-position attention weight | `0.0` |
| Masked attention assigns zero weight to all future-token positions | Passed |

### Generated Figures

- `figures/part2_unmasked_attention.png` — unmasked self-attention heatmap after training, full word-token labels on both axes
- `figures/part2_masked_attention.png` — causal-masked self-attention heatmap after training; upper-triangular half is uniformly zero, visually confirming the masking verification above
