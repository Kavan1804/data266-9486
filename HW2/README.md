# DATA 266 HW2

## Scope

Homework 2 covers:

1. Embedding-based transfer learning with `word2vec-google-news-300` fine-tuned on IMDB reviews.
2. A LangChain RAG pipeline over ten movie Wikipedia pages using MiniLM embeddings, FAISS, and `google/flan-t5-base`.
3. Controlled PyTorch measurements for five optimization techniques (CPU/GPU tensor creation, weight initialization, activation checkpointing, gradient accumulation, mixed precision).

## Personal Parameters

| Field | Value | HW2 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | All stochastic operations |
| SLICE | 486 | Reported only; not used for experiment selection |
| HP_ID | 0 | Reported only; HW2 defines no HP_ID mapping |
| CLS_A | 6 | Reported only |
| CLS_B | 0 | Reported only |

## Run Instructions

1. Open `hw2_nlp_optimization.ipynb` (the main submission notebook).
2. Run the personal-parameters and reproducibility cells first (`SEED=9486`).
3. **Part 1:** Install/import packages, load Google News Word2Vec, compute baseline neighbors, load IMDB, tokenize 10,000 reviews, fine-tune Word2Vec, compare neighbors/shifts, save the t-SNE figure and Part 1 artifacts.
4. **Part 2:** Install RAG packages, load `google/flan-t5-base`, load the ten Wikipedia pages (use `WebBaseLoader` URLs if `WikipediaLoader` fails), chunk 500/50, embed with MiniLM into FAISS, answer five questions with an explicit PromptTemplate/retriever/LLM pipeline, re-run two questions at 800/100, evaluate retrieval, save Part 2 artifacts.
5. **Part 3:** Run the five optimization experiments on the shared synthetic setup, save the results table and loss plot.
6. Update `METRICS.md`, `RUN_LOG.txt`, and `AI_USE.md` from observed outputs only.

`Hw2_demo.ipynb` is the professor reference only; do not submit it as your solution.

## Colab Requirements

- Internet access for Gensim Google News vectors, Hugging Face models, and Wikipedia pages.
- GPU (e.g., Colab Tesla T4) recommended for Part 2 LLM and Part 3 CUDA/mixed-precision measurements.
- Upload or mount the IMDB CSV. If the default CSV engine fails, use `engine="python"`.
- Pin Transformers to a release that supports `pipeline('text2text-generation', model='google/flan-t5-base')` (this run used `transformers==4.57.1`).

## Dataset Location

- Repository path used in configuration: `HW2/data/IMDB Dataset.csv`
- Colab execution path used in the completed run: `/content/IMDB Dataset.csv`
- Professor reference notebook: `HW2/Hw2_demo.ipynb`

## Artifact Locations

- `artifacts/part1/` — neighbor tables, vector-shift metrics, Part 1 configuration/summary
- `artifacts/part2/` — chunk manifest, retrieval results/evaluation, chunking comparison, RAG configuration
- `artifacts/part3/` — optimization results CSV and configuration JSON
- `figures/hw2/` — `part1_embedding_shift_tsne.png`, `part3_final_loss_comparison.png`
- `checkpoints/` — empty for this run (no model checkpoints intentionally saved)
- `report.md`, `HW2_Report_9486.pdf` — written report

## Reproducibility

Use `SEED=9486` for NumPy/PyTorch, Word2Vec, Transformers `set_seed`, and t-SNE. Exact Word2Vec and Wikipedia retrieval results can still vary with library versions, worker parallelism, live page HTML, and hardware. Record observed numbers from a specific run rather than inventing values.

## Files Intentionally Excluded from Git

- Large model caches (Gensim Google News, Hugging Face transformers/sentence-transformers weights)
- Full downloaded embedding binaries
- Temporary Colab `/content` copies of large datasets if duplicated elsewhere
- PDF preview scratch directories such as `_pdf_preview_hw1/` and `_pdf_preview_hw2/` used only for local visual QA
