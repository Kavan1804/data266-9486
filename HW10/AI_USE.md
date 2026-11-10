# DATA 266 Homework 10 — AI Use Disclosure

## Question 1

Which parts did you use an assistant for, and which parts did you write yourself?

### AI-Assisted Work

- Wrote and organized the Markdown in `HW10_RAG_Reranking_Chunking_Memory.ipynb`, including the explanations of the two chunking strategies, the cross-encoder reranker, the memory design, the relevance rules, and the metric definitions. It also drafted the discussion, limitations, and conclusion from the outputs of the run.
- Helped organize the notebook into the 18 sections of the assignment, starting from `Demo_10_RAG_Chatbot_Local.ipynb` (ChromaDB, `HuggingFaceEmbeddings` with mxbai, `ChatOllama`, the batched indexing loop, `augment_prompt`, and the chat-history classes the demo imports).
- Created `README.md`, `RUN_LOG.txt`, `METRICS.md`, and this file.
- Helped debug errors that came up while building and running the notebook, for example a pandas error where `results.query` pointed to the DataFrame's `query()` method instead of my `query` column, and a dependency conflict after installing chromadb (pip upgraded `python-dateutil` to 4.0, which I pinned back to 2.9).
- Generated the notebook with a build script and helped refine the code structure and readability: the persistent and resumable ChromaDB collections, the shared retrieve/rerank/judge functions, the `RUNS` dictionary that every table is computed from, and the smoke-test switch.
- Helped clarify implementation details, for example why mxbai needs the query instruction prefix only on queries, and why I count facts instead of chunks for Recall@K when the two strategies cut the papers differently.
- At my request it executed the smoke test and the full notebook on my Mac through Claude Code, so the outputs saved in the notebook come from that run.

### Work I Completed Myself

- Chose the queries and the two conversations, and decided what counts as relevant for each question (the target paper and the facts a good answer needs).
- Reviewed the code and made the decisions about the final setup: chunk sizes that keep both strategies at about 200 words, the 20-candidate pool, K = 3 and 5, and using `llama3.2:3b` because `phi3:mini` wasn't installed and my disk was almost full.
- Checked the outputs of the run: the chunk statistics, the before/after reranking tables, the memory rewrites, and whether the `R`/`.` labels in the verification tables match the chunks that were actually retrieved.
- Reviewed the final notebook and the supporting files and made sure the numbers in the discussion match the tables.

## Question 2

Give one specific thing it produced that was wrong, such as an error, incorrect output, or implementation issue.

**Response:** The first version of the query-rewriting step for the memory part didn't rewrite the question. It answered it. I gave `llama3.2:3b` the first turn about measuring a cavity's linewidth with squeezed vacuum, then the follow-up "Why does that method work better than probing the cavity with ordinary laser light?", and asked it to turn the follow-up into a standalone search query. It returned "That method works better than probing the cavity with ordinary laser light because the squeezed vacuum field has a reduced noise floor, ...". That's an answer, and it still says "that method", so it's useless as a search query.

## Question 3

How did you find out? What did the failure look like?

**Response:** I tested the rewrite step on its own before putting it into the notebook and printed what came back. The output was a full sentence that started with "That method works better ..." instead of a question. It still didn't name the method, and it made up a reason ("reduced noise floor") that hadn't come from any retrieved chunk. If that string had been sent to ChromaDB, the "with memory" retrieval would have been tested with a query that is no better than the original follow-up.

## Question 4

What did you change, and why does your version work?

**Response:** The rewrite now uses its own `ChatPromptTemplate`. The system message says the model only rewrites questions for a search engine and never answers them. The chat history goes into a `MessagesPlaceholder`, and the final human message says exactly what to do: replace "that method" / "this approach" / "it" with the thing they refer to, keep it a short question, don't answer it, and output only the rewritten question. With that prompt the same follow-up became "Why does injecting a squeezed vacuum field into a detuned cavity work better than probing the cavity with ordinary laser light?", and the second conversation's follow-up came out as a question that names the origin-destination matrix too. The model runs with `temperature=0` and a fixed seed, so it gives the same rewrite every time, and the notebook prints the exact query sent to the retriever for every memory variant so I could check it.
