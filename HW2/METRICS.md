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
| Implementation | Complete; sequential Colab-style notebook cells. |
| Execution | Executed in Google Colab; artifacts saved under `artifacts/part1/` and `figures/hw2/`. |
| IMDB dataset source | Colab path `/content/IMDB Dataset.csv` with `engine="python"`; configuration records `data/IMDB Dataset.csv`. Shape `(50000, 2)`. |
| Pretrained model | `word2vec-google-news-300` (vocab 3,000,000; vector size 300). |
| Reviews used | 10,000 |
| Reproducibility | `SEED=9486`; Word2Vec `seed=9486`, `workers=4`; t-SNE `random_state=9486`. |

### Configuration

| Setting | Value |
| --- | --- |
| Target words | `cast`, `score`, `plot`, `screen`, `review` |
| Vector size | 300 |
| Context window | 5 |
| Minimum count | 2 |
| Epochs | 5 |
| Gensim workers | 4 |
| Tokenization | Lowercase; HTML break tags to spaces; non-alphanumeric (except `'`) removed; whitespace tokens |
| Shared vocabulary initialized from pretrained vectors | 26,556 |

### Nearest Neighbors (Before)

| Word | Neighbor | Similarity |
| --- | --- | ---: |
| cast | casts | 0.721888 |
| cast | casting | 0.718846 |
| cast | Cast | 0.663812 |
| score | scoring | 0.719680 |
| score | scores | 0.659572 |
| score | scored | 0.638416 |
| plot | plots | 0.762485 |
| plot | Plot | 0.652418 |
| plot | plotting | 0.632763 |
| screen | screens | 0.772881 |
| screen | onscreen | 0.611545 |
| screen | LCD_screen | 0.559932 |
| review | reviewed | 0.663041 |
| review | reviewing | 0.660956 |
| review | reviews | 0.637956 |

### Nearest Neighbors (After)

| Word | Neighbor | Similarity |
| --- | --- | ---: |
| cast | casting | 0.660974 |
| cast | actors | 0.637971 |
| cast | performances | 0.627096 |
| score | soundtrack | 0.694458 |
| score | cinematography | 0.646970 |
| score | music | 0.625545 |
| plot | storyline | 0.712091 |
| plot | story | 0.691996 |
| plot | script | 0.622696 |
| screen | screens | 0.580450 |
| screen | stage | 0.486031 |
| screen | lebowski | 0.439573 |
| review | comment | 0.658783 |
| review | reviews | 0.637579 |
| review | recommendation | 0.585404 |

### Vector Shifts

| Word | Cosine similarity original vs fine-tuned | Shift `1 − cosine` |
| --- | ---: | ---: |
| review | 0.609045 | 0.390955 |
| score | 0.622409 | 0.377591 |
| screen | 0.676173 | 0.323827 |
| plot | 0.701515 | 0.298485 |
| cast | 0.781051 | 0.218949 |

| Required summary | Observed result |
| --- | --- |
| IMDB source and review count | `/content/IMDB Dataset.csv` (repo path `data/IMDB Dataset.csv`); 10,000 reviews |
| Pretrained model provenance | Gensim downloader `word2vec-google-news-300` |
| Fine-tuning configuration and seed | window 5, min_count 2, epochs 5, workers 4, seed 9486 |
| Most-shifted word | review (similarity 0.609045, shift 0.390955) |
| Least-shifted word | cast (similarity 0.781051, shift 0.218949) |
| 2D visualization artifact | `figures/hw2/part1_embedding_shift_tsne.png` |

## Part 2 — LangChain RAG

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Explicit PromptTemplate + retriever + context formatter + LLM (no RetrievalQA). |
| Execution | Executed in Google Colab on CUDA (`cuda:0`). |
| Document loader | `WikipediaLoader` failed with `JSONDecodeError`; replaced by `WebBaseLoader` + official Wikipedia URLs. |
| Primary chunking | 500 / 50 → 3,411 chunks |
| Alternate chunking | 800 / 100; Inception and The Godfather questions only |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| LLM | `google/flan-t5-base` |
| Reproducibility | `SEED=9486`; Transformers `set_seed(9486)`; `do_sample=False` |

### Retrieval Ranks and Success

| Question | Expected keyword aid | First relevant rank (manual) | Success | Generated answer |
| --- | --- | ---: | --- | --- |
| Who directed Inception, and what is its central plot concept? | christopher nolan | 1 | Yes | Christopher Nolan |
| Who played Don Vito Corleone in The Godfather? | marlon brando | 1 | Yes | Marlon Brando |
| What ship is central to the story of Titanic? | rms titanic | — | No | Titanic |
| Who is the main antagonist in The Dark Knight? | joker | — | No | Gambol |
| What major Oscar award did Parasite win? | best picture | 2 | Yes | Best Picture |

| Retrieval success metric | Observed result |
| --- | --- |
| Successful questions / total questions | 3 / 5 |
| Success rate | 60.00% |

### Alternate Chunking Comparison

| Question | Original (500/50) answer | Alternate (800/100) answer | Top chunk original (preview) | Top chunk alternate (preview) |
| --- | --- | --- | --- | --- |
| Inception director/plot | Christopher Nolan | Christopher Nolan | Inception is a 2010 science fiction heist film written and directed by Christopher Nolan… | “…emotional journey of his [DiCaprio's] character was the driving force…” |
| Godfather / Vito Corleone | Marlon Brando | Marlon Brando | Casting: Marlon Brando chosen to portray Vito Corleone… | Similar casting passage about Brando / Pacino / Caan |

### RAG Failure Analyses

| Failure | Evidence and analysis |
| --- | --- |
| Failure 1 — Dark Knight | Top-3 were reference/sequel snippets mentioning Gambol; answer was “Gambol” instead of Joker. |
| Failure 2 — Titanic | Top-3 were footnote/reference lines without a clear ship-identity passage; retrieval did not provide useful grounding despite the short answer “Titanic.” |

## Part 3 — Optimization Techniques

### Implementation and Execution Status

| Item | Status |
| --- | --- |
| Implementation | Five techniques in sequential PyTorch cells. |
| Execution | Google Colab; CUDA True; GPU Tesla T4; PyTorch 2.11.0+cu128. |
| Seed | 9486 |
| Shared dataset | Synthetic classification; input 128, hidden 256, output 10, batch 64, 40 steps, Adam lr 0.001 |

### Timing, Memory, and Final Loss

| Technique | Configuration | Device | Time (s) | Peak memory (MB) | Final loss |
| --- | --- | --- | ---: | ---: | ---: |
| Tensor creation | CPU tensor creation | cpu | 0.323885 | — | — |
| Tensor creation | GPU tensor creation | cuda | 0.012795 | 1059.211426 | — |
| Weight initialization | PyTorch default initialization | cuda | 0.286958 | 1061.304688 | 2.309902 |
| Weight initialization | Xavier uniform initialization | cuda | 0.062525 | 1061.304688 | 2.318031 |
| Activation checkpointing | No checkpointing | cuda | 0.134194 | 1070.088867 | 2.303427 |
| Activation checkpointing | Activation checkpointing | cuda | 0.260050 | 1071.088867 | 2.303427 |
| Gradient accumulation | Standard training | cuda | 0.069160 | 1062.304688 | 2.309902 |
| Gradient accumulation | 4 x 16 micro-batches | cuda | 0.212517 | 1062.304688 | 2.328997 |
| Mixed precision | Full precision | cuda | 0.066441 | 1062.304688 | 2.309902 |
| Mixed precision | CUDA mixed precision | cuda | 0.319262 | 1062.306152 | 2.310822 |

| Measurement context | Observed value |
| --- | --- |
| Seed | 9486 |
| PyTorch version | 2.11.0+cu128 |
| Hardware and runtime | Google Colab; Tesla T4; CUDA available |
| Memory-measurement method | `torch.cuda.max_memory_allocated` after synchronize; CPU tensor-creation memory recorded as unavailable |
| Loss plot | `figures/hw2/part3_final_loss_comparison.png` |
| CUDA / mixed precision | Both available and measured on this runtime |
