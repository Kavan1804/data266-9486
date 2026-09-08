# DATA 266 HW2 Metrics

## Personal Parameters

| Field | Value | HW2 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | Stochastic operations |
| SLICE | 486 | Reported only |
| HP_ID | 0 | Reported only; no HW2 mapping |
| CLS_A | 6 | Reported only |
| CLS_B | 0 | Reported only |

## Part 1 — Embedding-Based Transfer Learning

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Complete; guarded local-only notebook functions are present. |
| Execution | Not executed in this repository setup. No numerical Part 1 results are recorded. |
| IMDB dataset source | First valid read-only CSV among `../IMDB Dataset.csv`, `../../IMDB Dataset.csv`, and `data/hw2/IMDB Dataset.csv`; actual selected path is recorded only when run. |
| Pretrained model | `word2vec-google-news-300`, loaded only from a supplied local path or inspected standard Gensim cache; automatic download is disabled. |
| Default review count | All available reviews (`MAX_REVIEWS=None`); any explicit limit will be recorded when run. |
| Reproducibility | `SEED=9486`; Gensim seed 9486, one worker, sorted vocabulary, and t-SNE `random_state=9486`. |

### Planned Configuration

| Setting | Value |
| --- | --- |
| Target words | `cast`, `score`, `plot`, `screen`, `review` |
| Vector size | 300 |
| Context window | 5 |
| Minimum count | 2 |
| Epochs | 5 |
| Gensim workers | 1 |
| Tokenization | Lowercase; HTML break tags to spaces; punctuation separated; whitespace word tokens |
| Shared vocabulary initialized from pretrained vectors | TBD — actual count only after execution |

| Target word | Top 3 neighbors before | Top 3 neighbors after | Cosine similarity before vs. after |
| --- | --- | --- | ---: |
| cast | TBD | TBD | TBD |
| score | TBD | TBD | TBD |
| plot | TBD | TBD | TBD |
| screen | TBD | TBD | TBD |
| review | TBD | TBD | TBD |

| Required summary | Observed result |
| --- | --- |
| IMDB source and review count | TBD |
| Pretrained model provenance | TBD |
| Fine-tuning configuration and seed | TBD |
| Most-shifted word | TBD |
| Least-shifted word | TBD |
| 2D visualization artifact | TBD |

## Part 2 — LangChain RAG

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Complete; explicit guarded loader, splitter, embedding, FAISS, retriever, context formatter, prompt, and LLM functions are present. |
| Execution | Not executed in this repository setup. No retrieval, answer, ranking, success-rate, or failure values are recorded. |
| Document loader | `langchain_community.document_loaders.WikipediaLoader`; ten exact configured movie titles. |
| Primary chunking | 500 characters / 50-character overlap. |
| Alternate chunking | 800 characters / 100-character overlap; exactly the Inception and The Godfather questions. |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` — instantiated only when `run_part2()` is called. |
| Vector store | FAISS. |
| LLM | `google/flan-t5-base` through a local Transformers text-to-text-generation pipeline — instantiated only when `run_part2()` is called. |
| Reproducibility | `SEED=9486`; Transformers `set_seed(9486)` and deterministic (`do_sample=False`) generation. |
| Manual evaluation | Required after inspection of actual complete retrieved passages; keyword matching alone is not accepted. |

### Required Source Documents

1. Inception
2. The Godfather
3. Titanic (1997 film)
4. The Dark Knight
5. Pulp Fiction
6. Forrest Gump
7. The Matrix
8. Interstellar
9. Parasite (2019 film)
10. Gladiator (2000 film)

| Configuration | Value |
| --- | --- |
| Document loader and ten source documents | TBD |
| Primary chunk size / overlap | 500 / 50 |
| Embedding model | TBD |
| Vector store | TBD |
| PromptTemplate, retriever, and LLM | TBD |
| Alternate chunk size / overlap | TBD |

| Question | Retrieved evidence manually verified? | First relevant rank | Retrieval success | Generated answer | Notes |
| --- | --- | ---: | --- | --- | --- |
| Q1 | TBD | TBD | TBD | TBD | TBD |
| Q2 | TBD | TBD | TBD | TBD | TBD |
| Q3 | TBD | TBD | TBD | TBD | TBD |
| Q4 | TBD | TBD | TBD | TBD | TBD |
| Q5 | TBD | TBD | TBD | TBD | TBD |

| Retrieval success metric | Observed result |
| --- | --- |
| Successful questions / total questions | TBD |
| Success rate | TBD |

| Required RAG failure analysis | Evidence and analysis |
| --- | --- |
| Failure 1 | TBD |
| Failure 2 | TBD |

## Part 3 — Optimization Techniques

| Technique | Baseline / comparison | Device | Time | Memory (where available) | Loss or performance measure | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| Tensor creation CPU vs. GPU | TBD | TBD | TBD | TBD | TBD | TBD |
| Weight initialization | TBD | TBD | TBD | TBD | TBD | TBD |
| Activation checkpointing | TBD | TBD | TBD | TBD | TBD | TBD |
| Gradient accumulation | TBD | TBD | TBD | TBD | TBD | TBD |
| Mixed precision training | TBD | TBD | TBD | TBD | TBD | TBD |

| Measurement context | Observed value |
| --- | --- |
| Seed | 9486 |
| PyTorch version | TBD |
| TensorFlow version, if used | TBD |
| Hardware and runtime | TBD |
| Warm-up / trial protocol | TBD |
| Memory-measurement method and limitations | TBD |
