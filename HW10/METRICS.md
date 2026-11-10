# DATA 266 HW10 Metrics

All numbers come from the full run of `HW10_RAG_Reranking_Chunking_Memory.ipynb` on my Apple M1 (see `results/hw10_metrics.json`, `results/metrics_per_run.csv`, and `results/retrieval_runs.csv`, which lists every retrieved chunk with its relevance label). K = 3 and 5 for Precision@K and Recall@K, and MRR is computed over the top 10. Recall@K is fact-level: it's the share of the hand-written answer facts for a query that are covered by relevant chunks in the top K (see notebook Section 15). A chunk counts as relevant only if it comes from the query's target paper and contains at least one of the facts.

## Personal Parameters

| Field | Value | How it is calculated | HW10 use |
| --- | ---: | --- | --- |
| SID4 | 9486 | given | Reported |
| SEED | 9486 | SID4 | Python / NumPy / PyTorch seeds, and Ollama `seed` (with temperature 0) |
| SLICE | 486 | SID4 % 1000 | paper used for the chunk examples: SLICE % 100 = 86 |
| HP_ID | 0 | SID4 % 6 | Reported only |
| CLS_A | 6 | SID4 % 10 | Reported only |
| CLS_B | 0 | (CLS_A + 1 + (SID4 // 10) % 9) % 10 | Reported only |

## Setup

| Item | Value |
| --- | --- |
| Data | `ccdv/arxiv-summarization`, config `document`, split `train`, first 100 rows (streamed with `take(100)`), `article` field, 541,857 words after cleaning |
| Strategy 1 | fixed 200-word windows, 40-word overlap: 3,411 chunks, mean 197.7 words, 7.0% end at a sentence |
| Strategy 2 | `RecursiveCharacterTextSplitter`, at most 1,200 chars, overlap up to 200 chars, separators `". "`, `"; "`, `", "`, `" "`: 3,500 chunks, mean 172.2 words, 97.6% end at a sentence, 21.2% of neighbouring pairs have no overlap |
| Embeddings | `mixedbread-ai/mxbai-embed-large-v1` (335M parameters, 1024-dim, normalized, query instruction prefix), max 512 tokens: 5.25% of S1 and 3.91% of S2 chunks are truncated |
| Vector DB | ChromaDB 1.5.9, persistent, cosine HNSW, two collections (`hw10-s1-fixed-words`, `hw10-s2-recursive`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` (22.7M parameters), reorders the same top 20 ChromaDB candidates |
| Memory | `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`. Variants: no memory / previous user question + follow-up (concat) / LLM rewrite with `llama3.2:3b` (temperature 0, seed 9486) |
| Queries | 8 standalone (Q1-Q6, C1-T1, C2-T1) and 2 follow-ups (C1-T2, C2-T2) |
| Hardware | Apple M1, 8 GB, MPS. Python 3.13.9, torch 2.14.0, transformers 5.21.0, sentence-transformers 6.2.0, langchain-core 1.7.7, datasets 5.1.0 |

## Runtime

| Stage | Time |
| --- | ---: |
| Load 100 rows (streaming) | 3.7 s |
| Embed + index S1 (3,411 chunks) | 1,026.5 s (3.3 chunks/s) |
| Embed + index S2 (3,500 chunks) | 1,039.0 s (3.4 chunks/s) |
| Retrieval, 16 lists of 20 | 1.4 s |
| Reranking, 16 lists of 20 | 5.6 s |
| Two conversations with Ollama (2 rewrites, 6 answers) | 142.1 s |
| Whole notebook | 37.4 min |

## Standalone Queries: Four Conditions (mean over 8 queries)

| Condition | P@3 | P@5 | R@3 | R@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 fixed words, no rerank | 0.833 | 0.850 | 0.885 | 0.958 | 0.938 |
| S1 fixed words, rerank | 0.875 | 0.825 | 0.917 | 0.958 | 0.917 |
| S2 recursive sentences, no rerank | 0.917 | 0.850 | 0.844 | 0.958 | 1.000 |
| S2 recursive sentences, rerank | 0.833 | 0.800 | 0.885 | 0.958 | 0.938 |

In all 32 standalone ranked lists, all top 5 chunks came from the correct paper.

## Chunking: S2 − S1

| Setting | P@3 | P@5 | R@3 | R@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no rerank | +0.083 | 0.000 | −0.042 | 0.000 | +0.062 |
| rerank | −0.042 | −0.025 | −0.031 | 0.000 | +0.021 |

Per query, MRR@10 was the same in 14 of 16 (query, rerank) pairs, and S2 was better in 2.

## Reranking: rerank − no rerank

| Strategy | P@3 | P@5 | R@3 | R@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 | +0.042 | −0.025 | +0.031 | 0.000 | −0.021 |
| S2 | −0.083 | −0.050 | +0.042 | 0.000 | −0.062 |

Per (query, strategy) pair: P@5 was better in 3, the same in 9 and worse in 4. MRR@10 was better in 1, the same in 13 and worse in 2 (Q5 dropped from 1.000 to 0.333 for S1 and to 0.500 for S2).

## Conversational Memory (follow-ups C1-T2 and C2-T2, mean over 2 strategies × 2 rerank settings)

| Variant | Retriever query (C1-T2) | P@3 | P@5 | R@3 | R@5 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no memory | "Why does that method work better than probing the cavity with ordinary laser light?" | 0.667 | 0.500 | 0.625 | 0.708 | 0.854 |
| memory (concat) | previous user question + follow-up | 0.833 | 0.800 | 0.750 | 0.917 | 0.938 |
| memory (LLM rewrite) | "Why does injecting a squeezed vacuum field into a detuned cavity work better than probing the cavity with ordinary laser light?" | 0.667 | 0.550 | 0.708 | 0.917 | 0.875 |

The C2-T2 rewrite was "Does the gravity-type model require a prior origin-destination matrix to estimate the number of passengers traveling between each pair of stations?". The gravity model came from the LLM's own turn-1 answer and isn't the paper's method. The worst no-memory case was C2-T2 on S2 without reranking: P@5 0.200, MRR 0.333, with paper 93 in the top 5. With the concat memory, the same setting got P@5 1.000 and MRR 1.000.

## Chunk-Level Recall (extra, textbook definition)

| Condition | chunkR@3 | chunkR@5 |
| --- | ---: | ---: |
| S1, no rerank | 0.186 | 0.320 |
| S1, rerank | 0.193 | 0.296 |
| S2, no rerank | 0.228 | 0.344 |
| S2, rerank | 0.187 | 0.303 |
