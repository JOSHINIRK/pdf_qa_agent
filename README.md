# PDF Q&A Agent

An LLM agent that reads PDFs (research papers, manuals, contracts, ...), breaks them into their sections, and answers questions with **citations to the exact document, section and page**. Click a citation to see the PDF page with the passage highlighted.

It uses **OpenAI function calling** directly, with no LangChain layer. The model decides which tools to call: search, read the outline, read a section or read pages. Every citation it makes is checked against the passages the tools actually returned.

```
INGESTION                                     QUESTION
                                              
PDF ──► PyMuPDF: text + fonts + bookmarks     question ──► agent loop (OpenAI function calling, ≤ 8 model calls)
    ──► strip headers/footers/page numbers                  │   tools:  list_documents
    ──► detect sections (bookmarks ▸ fonts ▸ pages)         │           search_documents  ──► hybrid search
    ──► chunk per section (≤ 400 tokens, overlap)           │           get_document_outline       (vectors + BM25, RRF)
    ──► embed (text-embedding-3-small)                      │           read_section
    ──► ChromaDB  +  documents/*.json  +  pdfs/*.pdf         │           read_pages
                                                            ▼
                                              answer citing [S3] ──► validated, renumbered [1] ──► UI:
                                                                     answer · sources · highlighted page · agent steps
```

## Contents

1. [Quick start](#1-quick-start)
2. [Using the app](#2-using-the-app)
3. [How it works](#3-how-it-works)
4. [Configuration](#4-configuration)
5. [Command-line tools](#5-command-line-tools)
6. [Testing and evaluation](#6-testing-and-evaluation)
7. [Deployment](#7-deployment)
8. [Project structure](#8-project-structure)
9. [Troubleshooting](#9-troubleshooting)
10. [Limitations and security](#10-limitations-and-security)

---

## 1. Quick start

You need Python 3.11+ and an OpenAI API key with credit. ChatGPT subscriptions don't include API access; see section 1 of [../rag_chatbot/README.md](../rag_chatbot/README.md) for how to create a key.

```powershell
cd D:\Projects\projects\pdf_qa_agent
py -3 -m venv .venv                                  # skip if .venv already exists
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
notepad .env                                         # paste your key after OPENAI_API_KEY=
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at http://localhost:8501. Click **Load sample documents** in the sidebar, then try one of the example questions.

> **Windows:** start the app with `python -m streamlit`, not `streamlit.exe`. On this laptop, Smart App Control blocks the unsigned `streamlit.exe` launcher that pip creates.

## 2. Using the app

**Sidebar**

- **Add PDFs:** upload one or more files, then click **Ingest**. Each file is parsed, split into sections and chunks, and indexed. Uploading the same file again is detected and skipped.
- **Load sample documents:** generates and ingests three fictional test PDFs: a research paper, a product manual and a service contract.
- **Library:** each document shows its page, section and chunk counts, plus any warnings (for example, scanned pages with no text). **Remove** deletes it from the index.
- **Search in:** restricts questions to selected documents. Leave it empty to search all of them.

**Ask view**

- Answers stream in while the status line shows what the agent is doing ("Searching...", "Reading section...").
- **Sources:** each `[n]` in the answer is listed with document, section and page, the quoted passage, and a **Show page** toggle that renders the PDF page with the passage highlighted.
- **Agent steps:** every tool call with its arguments and result, plus time and token use.
- An answer with no citations is flagged. **Download answer** saves the answer and its sources as Markdown.
- Follow-up questions work: the last 4 question/answer pairs are sent as context.

**Documents view:** the section breakdown of each document (titles, nesting, page ranges, text of each section), the source of the structure (bookmarks, headings or pages), and a page viewer.

## 3. How it works

### 3.1 Parsing and section detection ([pdfagent/parsing.py](pdfagent/parsing.py))

PyMuPDF extracts every line with its font size, boldness and position. Then:

1. **Clean-up:** lines in the top or bottom 8% of a page that repeat on at least half the pages (running headers and footers) are removed, as are page numbers. Hyphenated line breaks are joined and ligatures (ﬁ → fi) expanded.
2. **Section structure**, from the best available source:
   | Source | When | How |
   |---|---|---|
   | `outline` | The PDF has 2+ bookmarks | Each bookmark is matched to its heading line. If a bookmark has the wrong page number, it is looked up elsewhere. |
   | `layout` | No bookmarks | Headings are lines in a font ≥ 12% larger than the body text, or short stand-alone bold lines such as `8. TERM AND TERMINATION`. Numbering (`3.1`) sets the depth; otherwise font size does. |
   | `pages` | No headings found | One section per page. |
3. **Tables:** cells on the same baseline are joined with ` | `, one row per line, so rows stay readable for the model.
4. **Title:** the PDF metadata title if it looks real, otherwise the largest text at the top of page 1, otherwise the file name.

### 3.2 Chunking ([pdfagent/chunking.py](pdfagent/chunking.py))

Chunks never cross a section boundary, so each one has a single section path and an exact page range. Paragraphs are packed up to 400 tokens; longer ones are split at sentence boundaries. The last ~60 tokens of a chunk are repeated at the start of the next. The text that gets embedded is prefixed with the document title and section path, so "what does the warranty say" matches the Warranty section even when the body never uses that word.

### 3.3 Hybrid search ([pdfagent/store.py](pdfagent/store.py))

Vector search (OpenAI embeddings in ChromaDB) finds passages by meaning. BM25 keyword search finds exact terms that embeddings handle poorly: clause numbers, part codes like `PF-1`, names and figures. The two rankings are merged with reciprocal-rank fusion.

### 3.4 The agent ([pdfagent/agent.py](pdfagent/agent.py), [pdfagent/tools.py](pdfagent/tools.py))

| Tool | Purpose |
|---|---|
| `list_documents` | What is in the library (ids, titles, pages). |
| `search_documents(query, document_ids?, top_k?)` | Hybrid search; returns passages with a `source_id`. |
| `get_document_outline(document_id)` | Section tree with ids and page ranges: "what are the sections of...". |
| `read_section(document_id, section_id)` | Full text of a section and its subsections: "summarise the limitations". |
| `read_pages(document_id, start, end)` | Text of up to 5 pages: "what's on page 3". |

The loop works like this:

- **Model calls:** at most `agent_max_steps` per question. The model may call several tools in parallel. On the last step tools are disabled and the model must answer from what it has.
- **Errors:** a tool error, such as an unknown section or bad JSON arguments, goes back to the model as a message so it can correct itself. It never crashes the loop.
- **Result size:** tool results are trimmed to `max_tool_result_tokens`.
- **Document references:** the model can name a document by id, filename or title.

**Citations.** Every passage a tool returns gets an id (`S1`, `S2`, ...), and the model is told to cite those ids. After the answer:

- each `[S#]` is looked up in the registry of passages actually shown to the model;
- unknown ids are dropped and logged;
- the rest are renumbered `[1]`, `[2]`, ... in order of appearance, each with document, section, page range and the exact text.

So a citation can never point at text the model didn't see.

**Prompt injection.** Document text is untrusted. The system prompt tells the model to treat it as data, and tools can only read the library; none of them write or reach the network.

## 4. Configuration

All settings are in [config.yaml](config.yaml) and commented there. Any setting can be overridden with an environment variable `PDFQA_<NAME>`, for example `PDFQA_CHAT_MODEL=gpt-4.1` or `PDFQA_DATA_DIR=/data`.

| Setting | Default | Notes |
|---|---|---|
| `chat_model` | `gpt-4.1-mini` | Must support tool calling. For reasoning models set `temperature: null`. |
| `embedding_model` | `text-embedding-3-small` | Changing it requires `scripts/ingest.py --reset ...`. |
| `chunk_tokens` / `chunk_overlap_tokens` | 400 / 60 | Changing them requires re-ingesting. |
| `search_top_k` / `search_candidates` | 6 / 30 | Passages per search, and candidates per retriever before fusion. |
| `agent_max_steps` | 8 | Upper bound on model calls (cost and latency) per question. |
| `max_tool_result_tokens` | 4000 | Caps the context each tool call adds. |
| `max_upload_mb` / `max_pages` | 50 / 1000 | Also update `maxUploadSize` in `.streamlit/config.toml`. |

**Cost.** Ingesting the three samples costs well under 1 US cent. A typical question with `gpt-4.1-mini` takes 2-3 model calls and 4-8k tokens, about 0.2-0.5 US cents. The "Agent steps" panel shows the exact token count for each answer.

## 5. Command-line tools

```powershell
.\.venv\Scripts\python.exe scripts\make_samples.py                  # regenerate samples\*.pdf
.\.venv\Scripts\python.exe scripts\ingest.py samples                 # ingest a folder, files or globs
.\.venv\Scripts\python.exe scripts\ingest.py --list
.\.venv\Scripts\python.exe scripts\ingest.py --delete <doc_id>
.\.venv\Scripts\python.exe scripts\ingest.py --reset samples         # wipe the library, then ingest
.\.venv\Scripts\python.exe scripts\ask.py --trace "What is the liability cap?"
.\.venv\Scripts\python.exe scripts\ask.py --doc product_manual.pdf "What does a flashing red light mean?"
```

## 6. Testing and evaluation

### Offline tests (no API key, no cost)

```powershell
.\.venv\Scripts\python.exe -m pytest
```

These 47 tests use a deterministic bag-of-words embedder and a scripted fake model. They cover:

- **Parsing:** each of the three document types, including the exact heading list; header and footer removal; tables; the page fallback; and rejection of scanned, encrypted and invalid PDFs.
- **Chunking:** section boundaries, page ranges, overlap and oversized paragraphs.
- **Store:** hybrid search, scoping, persistence across restarts, deletion, and rollback after a failed or interrupted ingestion.
- **Tools and agent loop:** parallel tool calls, the step limit, streaming reset, history, error reporting, and citation validation and renumbering.
- **Page highlighting and UI:** the Streamlit app itself, run headlessly.

### Evaluation on the three document types (needs API key)

[eval/golden.jsonl](eval/golden.jsonl) has 20 questions: 5 about the research paper, 6 about the manual, 6 about the contract, 1 that needs two documents, and 2 the documents can't answer.

```powershell
.\.venv\Scripts\python.exe scripts\ingest.py samples
.\.venv\Scripts\python.exe scripts\evaluate.py --retrieval-only   # search quality only (embeddings cost)
.\.venv\Scripts\python.exe scripts\evaluate.py                    # full agent, about 1-2 US cents
```

| Metric | Meaning |
|---|---|
| retrieval hit | The expected page is among the top-k search results. |
| answer correct | Every key fact (for example "60 days", "England and Wales") appears in the answer. |
| citation ok | A citation points to the expected document and page; for the two-document question, both documents are cited. |
| refusal ok | For unanswerable questions, the answer says the documents don't contain it instead of guessing. |

The three sample types each test a different path: the **paper** has no bookmarks, so its headings come from font sizes and numbering, and it has a results table. The **manual** has bookmarks, a running header, page footers and several tables. The **contract** has bold numbered clauses in the body font size. To test your own documents, add lines to the golden file in the same format.

## 7. Deployment

```bash
docker build -t pdf-qa-agent .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... -v pdfqa-data:/data pdf-qa-agent
```

The image runs as a non-root user, keeps the library in the `/data` volume, and has a health check on `/_stcore/health`.

Before exposing it beyond your own machine:

- **Authentication.** The app has none. Put it behind an authenticating reverse proxy, or use Streamlit's built-in OIDC login (`st.login`).
- **Shared library.** All users see the same documents. For per-user libraries, give each user their own `data_dir` or ChromaDB collection.
- **Scale.** Data is local: ChromaDB on disk, plus an in-memory BM25 index rebuilt when the library changes. That is right for a single instance with up to a few thousand documents. Beyond that, move to a server vector database (Chroma server, pgvector, Qdrant) with built-in keyword search.

## 8. Project structure

```
pdf_qa_agent/
├── app.py                    Streamlit UI (Ask / Documents views)
├── config.yaml               all settings (env overrides: PDFQA_<NAME>)
├── pdfagent/
│   ├── config.py             Settings loading, API key, logging
│   ├── models.py             Document / Section / Block / Chunk data classes
│   ├── parsing.py            PDF → clean text → sections (bookmarks ▸ fonts ▸ pages)
│   ├── chunking.py           section-aware, token-bounded chunks with overlap
│   ├── tokens.py             token counting (tiktoken, with offline fallback)
│   ├── llm.py                OpenAI embeddings + streaming chat with tool calls; friendly errors
│   ├── store.py              ChromaDB + JSON records + BM25; hybrid search; crash-safe writes
│   ├── ingestion.py          validate → parse → chunk → embed → store; dedupe by SHA-256
│   ├── tools.py              tool schemas + implementations + citation source registry
│   ├── agent.py              function-calling loop, citation validation, streaming events
│   ├── render.py             page → PNG with the cited passage highlighted
│   └── samples.py            generator for the 3 fictional sample PDFs
├── scripts/                  ingest.py · ask.py · evaluate.py · make_samples.py
├── eval/golden.jsonl         evaluation questions
├── samples/                  generated sample PDFs
├── tests/                    offline test suite
├── Dockerfile
└── requirements.txt
```

## 9. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `No OpenAI API key found` | `.env` is missing or still holds the placeholder. Copy `.env.example` to `.env`, add the key, and restart. |
| `OpenAI rejected the API key (401)` | The key is wrong or was deleted. Create a new one. |
| `rate limit or quota reached (429)` | No credit left, or too many requests. Add credit, or wait a minute. |
| `streamlit.exe` blocked by an Application Control policy | Start with `.\.venv\Scripts\python.exe -m streamlit run app.py`. |
| `DLL load failed ... Application Control policy` | Smart App Control blocked a newly released compiled package. Install an earlier release of the package it names (`pip install "name==x.y.*"`); `requirements.txt` already pins pydantic and pandas. |
| "No extractable text" when uploading | The PDF is a scanned image. Run OCR first, for example `ocrmypdf in.pdf out.pdf`, then upload the result. |
| Sections look wrong | Check the **Documents** view. Unusual layouts can confuse heading detection; answers still work because search doesn't depend on sections, but section-level tools are less precise. |
| Changed `embedding_model` or chunk sizes | Run `scripts\ingest.py --reset <your files>`. |

## 10. Limitations and security

- **Scanned PDFs** need OCR first. A page without text is reported as a warning at ingestion.
- **Images and charts** are ignored. Tables are kept as text rows, but complex merged-cell tables may read imperfectly.
- **Multi-column layouts** follow the PDF's own text order, which is usually right for papers but not guaranteed.
- **Secrets:** the API key is read from `.env` or the environment and never logged. Question text isn't logged at INFO level.
- **Uploads:** checked for the PDF signature, size and page count. File names from the browser are reduced to their base name, and stored files are named by content hash.
- **Highlighting** happens in memory. Original PDFs are never modified.
