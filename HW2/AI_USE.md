# DATA 266 Homework 2 — AI Use Disclosure

## Question 1

What AI assistance, if any, did you use for Homework 2? Describe the tasks assisted and the work you personally reviewed, ran, or decided.

**Response:** AI assistance (Cursor) was used to organize the HW2 notebook and repository structure; write and simplify implementation code to match the professor demo style; draft documentation, markdown explanations, and comments; set up personal-parameter and reproducibility cells; debug Colab execution errors from pasted traces; and organize tables, figures, metrics, and report formatting for the final PDF.

I personally executed the notebook in Google Colab, supplied/verified the IMDB dataset upload, inspected retrieved Wikipedia passages, verified generated answers, identified the RAG failures (Dark Knight → Gambol; Titanic reference-only retrieval), reviewed the actual numerical Part 1–3 results against saved artifacts, and checked the final submission files under `HW2/`.

## Question 2

Provide one specific incorrect, incomplete, or misleading AI-generated suggestion or output encountered during Homework 2.

**Response:** Several concrete debugging failures occurred during Colab execution. Representative examples:

1. **IMDB `ParserError`.** Default `pd.read_csv(...)` with the C engine failed on the IMDB CSV (quoted multiline reviews). Detection: traceback while loading the dataset. Correction: reread with `engine="python"` from `/content/IMDB Dataset.csv`. Why it worked: the Python engine tolerates the CSV’s quoting/multiline fields.

2. **Missing `WINDOW` variable.** An early Word2Vec construction referenced `WINDOW` before it was defined in the active kernel state. Detection: `NameError`. Correction: define `WINDOW = 5` (with `MIN_COUNT`, `EPOCHS`, `workers`, `seed`) in the fine-tuning cell before constructing `Word2Vec`. Why it worked: the hyperparameter existed in scope for `window=WINDOW`.

3. **Transformers unsupported `text2text-generation` task.** A newer Transformers install rejected or failed the FLAN-T5 `pipeline('text2text-generation', ...)`. Detection: pipeline construction error in Part 2. Correction: uninstall Transformers and pin `transformers==4.57.1`, then recreate the pipeline. Why it worked: 4.57.1 still supports the text2text-generation task used with `google/flan-t5-base`.

4. **Incorrect `acceleratefrom` installation text.** A pip/install suggestion was malformed as concatenated package text (`acceleratefrom` rather than a proper `accelerate` install line). Detection: invalid package / install failure. Correction: install `accelerate` as its own package in the Part 2 pip cell. Why it worked: the dependency name must be exactly `accelerate`.

5. **Incorrect `WikipediaLoader` import / API failure.** `WikipediaLoader` was imported from `langchain_community.document_loaders`, but live loads failed with `JSONDecodeError` from the Wikipedia API response. Detection: exception while iterating movie titles. Correction: switch to `WebBaseLoader` with direct official Wikipedia URLs and install `beautifulsoup4`/`lxml`. Why it worked: HTML page fetch bypassed the broken API JSON path and still produced LangChain Documents with source metadata.

6. **Missing `RecursiveCharacterTextSplitter` import.** Chunking called the splitter before importing it from `langchain_text_splitters`. Detection: `NameError`/`ImportError` on the split cell. Correction: `from langchain_text_splitters import RecursiveCharacterTextSplitter` in the chunking cell. Why it worked: the class became available for 500/50 and 800/100 splits.

## Question 3

How did you discover that the suggestion or output in Question 2 was incorrect, incomplete, or misleading?

**Response:** Each issue was discovered from Colab cell tracebacks or failed outputs during the live run (ParserError on CSV read; NameError for undefined names; pipeline/task error for Transformers; pip failure for the malformed accelerate line; JSONDecodeError from WikipediaLoader; missing splitter import). I also manually read retrieved passages and saw that automatic keyword misses were not the only failure signal—Dark Knight returned Gambol-centered citations and Titanic returned footnote-only chunks.

## Question 4

What correction did you make, and how did you verify the correction?

**Response:** Corrections are listed under Question 2. Verification: IMDB loaded with shape `(50000, 2)` and 10,000 tokenized reviews; Word2Vec trained and neighbor/shift tables saved; `flan-t5-base` printed `LLM: google/flan-t5-base` on `cuda:0`; ten Wikipedia documents loaded via WebBaseLoader; 3,411 chunks embedded into FAISS; five answers and retrieval JSON written; Part 3 optimization CSV/plot saved on Tesla T4. Final numeric values were cross-checked against `artifacts/part1`, `artifacts/part2`, and `artifacts/part3` before writing the report.
