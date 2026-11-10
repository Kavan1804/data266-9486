# DATA 266 Homework 10 — Improving RAG with Reranking, Chunking, and Conversational Memory

## Overview

A local RAG retrieval pipeline built from `Demo_10_RAG_Chatbot_Local.ipynb`. It loads the first 100 papers of `ccdv/arxiv-summarization` (config `document`, split `train`), splits them with two chunking strategies (fixed 200-word windows with a 40-word overlap, and recursive sentence-aware chunks of at most 1,200 characters with up to 200 characters of overlap), and embeds them with `mixedbread-ai/mxbai-embed-large-v1` into two separate ChromaDB collections. It then retrieves the top 20 chunks per query, reranks them with the cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2`, and runs two two-turn conversations with chat memory (`InMemoryChatMessageHistory` + `RunnableWithMessageHistory`, with `llama3.2:3b` in Ollama for the query rewrite). Everything is evaluated with Precision@K, Recall@K (K = 3, 5) and MRR@10 against relevance rules I wrote by hand.

Results from the full run on my Apple M1 (8 GB, MPS), 37.4 minutes in total:

- **Chunking:** sentence-aware chunking (S2) was slightly better at the top of the ranking without reranking (P@3 0.917 vs. 0.833, MRR 1.000 vs. 0.938). P@5 and R@5 were tied.
- **Reranking:** did not help overall. P@5 and MRR went down for both strategies (S2: P@5 0.850 → 0.800, MRR 1.000 → 0.938), while R@3 went up a little.
- **Memory:** helped the follow-up questions. Adding the previous user question raised P@5 from 0.500 to 0.800 and R@5 from 0.708 to 0.917. The LLM rewrite also reached R@5 0.917, but it carried a mistake from its own earlier answer into one of the rewritten queries.

The full numbers are in `METRICS.md`, and the discussion is in Sections 17-18 of the notebook.

## Personal Parameters

SID4 9486 → SEED 9486, SLICE 486, HP_ID 0, CLS_A 6, CLS_B 0.

## Folder Structure

```
HW10/
├── Demo_10_RAG_Chatbot_Local.ipynb          professor's demo (unchanged)
├── Demo_10_RAG_Chatbot.ipynb                professor's demo (unchanged)
├── Demo_10_cagdemo.ipynb                    professor's demo (unchanged)
├── HW10_RAG_Reranking_Chunking_Memory.ipynb main notebook with the outputs of the M1 run
├── METRICS.md                               results in tables
├── RUN_LOG.txt                              how the notebook was built and run
├── AI_USE.md                                AI use disclosure
├── results/                                 hw10_metrics.json, metrics_per_run.csv, retrieval_runs.csv, run_log_raw.txt
└── chroma_db/                               the two persistent ChromaDB collections (about 100 MB, rebuilt by the notebook; not committed)
```

## Run Instructions

Only `HW10_RAG_Reranking_Chunking_Memory.ipynb` needs to be run. The three demo notebooks are reference material.

1. Install the packages (the notebook also installs any that are missing):
   ```bash
   pip install datasets chromadb langchain-core langchain-chroma langchain-huggingface langchain-text-splitters langchain-ollama sentence-transformers
   ```
2. Install [Ollama](https://ollama.com/), start it, and pull the model: `ollama pull llama3.2:3b`. Without Ollama, the notebook still runs everything except the LLM rewrite and the answers, and it prints that those were skipped.
3. Open the notebook in the `HW10` folder and Run All. No API keys or Hugging Face login are needed.

The first run downloads the dataset rows (streaming, only 100 rows), the embedding model (about 670 MB) and the reranker (about 90 MB). It then embeds 6,911 chunks, which took about 35 minutes on my M1 (3.3-3.4 chunks/s) and should take a few minutes on a CUDA GPU. The collections are saved in `chroma_db/`, so a re-run only embeds missing chunks and finishes in a few minutes.

**Smoke test (checks that every cell runs, about 2-5 minutes):** every paper is cut to 300 words and stored in a separate `chroma_db_smoke/` folder. Run it on a copy so it does not overwrite the reported outputs:

```bash
mkdir -p /tmp/hw10_smoke && cp HW10_RAG_Reranking_Chunking_Memory.ipynb /tmp/hw10_smoke/ && cd /tmp/hw10_smoke
HW10_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 HW10_RAG_Reranking_Chunking_Memory.ipynb --output smoke_run.ipynb
```

The device is chosen automatically (CUDA, then MPS, then CPU). Set `HW10_DEVICE` to force one, `HW10_EMBED_MODEL` to use a different embedding model, and `HW10_LLM` to use a different Ollama model.
