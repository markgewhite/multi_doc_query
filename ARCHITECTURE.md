# Architecture

Multi Doc Query is a three-stage RAG pipeline — ingestion, retrieval, generation — composed in a Chainlit app handler. Each stage is a subpackage under `src/` with simple, testable interfaces.

## Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              INGESTION                                   │
│                                                                          │
│  Documents folder ──▶ Scanner ──▶ Loader ──▶ Chunker ──▶ VectorStore     │
│  (.pdf .docx .txt .md) (discover   (parse     (split      (embed + store │
│                         + MD5       per-format) + metadata) in ChromaDB) │
│                         hash)                                            │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                              RETRIEVAL                                   │
│                                                                          │
│  User query ──▶ Condenser ──▶ HybridRetriever ──────────────▶ Reranker   │
│                 (rewrite      ┌──▶ BM25Index (top 20)        (cross-     │
│                  follow-ups)  │                               encoder    │
│                               ├──▶ VectorStore (top 20)      scores      │
│                               │                               pairs,     │
│                               └──▶ RRF fusion (top 30) ─────▶ top 10)    │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                              GENERATION                                  │
│                                                                          │
│  Top 10 chunks ──▶ Answerer ──▶ Streaming response with inline citations │
│                    (prompt construction    [filename.pdf, p. 12]         │
│                     + ollama.chat)                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

## Modules

### Ingestion (`src/ingestion/`)

| Module | Interface | Purpose |
|--------|-----------|---------|
| `scanner.py` | `scan_folder(path, recursive) → list[Path]` | Discovers supported files; `compute_file_hash(path) → str` computes MD5 for change detection |
| `loader.py` | `load_file(path, root) → list[Document]` | Parses documents via LangChain loaders (PyPDFLoader, Docx2txtLoader, TextLoader). One Document per PDF page, one per other format |
| `chunker.py` | `chunk_documents(docs, chunk_size, chunk_overlap) → list[Chunk]` | Splits text with RecursiveCharacterTextSplitter; Markdown files use MarkdownHeaderTextSplitter to preserve section headers. Attaches `doc_hash` metadata |
| `ingest.py` | `ingest_folder(folder, store, ...) → IngestResult` | Orchestrates scan → hash check → load → chunk → store. Supports `force` mode (skip hash check), per-file `on_progress` callback, and try/except per file for failure resilience |

### Retrieval (`src/retrieval/`)

| Module | Interface | Purpose |
|--------|-----------|---------|
| `vector_store.py` | `VectorStore.add_chunks()`, `.search(query, k)`, `.has_document(hash)` | ChromaDB wrapper with injected embedding function. Cosine similarity search |
| `embeddings.py` | `make_ollama_embed_fn(model) → EmbedFn` | Creates an embedding function that calls `ollama.embed()` directly (not via LangChain) |
| `bm25_index.py` | `BM25Index.build(texts, metadatas)`, `.search(query, k)` | BM25 keyword ranking via rank-bm25. Rebuilt from ChromaDB texts on startup |
| `fusion.py` | `reciprocal_rank_fusion(list1, list2, k, top_n) → list[SearchResult]` | Custom RRF implementation (~15 lines). Merges BM25 and semantic result lists |
| `hybrid.py` | `HybridRetriever.retrieve(query) → list[SearchResult]` | Orchestrates BM25 + semantic + RRF. Returns 30 fused candidates |
| `reranker.py` | `Reranker.rerank(query, results, top_k) → list[SearchResult]` | Cross-encoder scoring via sentence-transformers. Scores all query-chunk pairs and returns top_k sorted by relevance |

### Generation (`src/generation/`)

| Module | Interface | Purpose |
|--------|-----------|---------|
| `answerer.py` | `answer(question, results, model) → AsyncIterator[str]` | Builds prompt with source context, streams tokens via `ollama.chat()`. Also provides `build_source_elements()` for Chainlit display |
| `condenser.py` | `Condenser.condense(question, chat_history) → str` | Rewrites follow-up questions into standalone queries via LLM call. Skips condensation when no chat history (first question) |

### Supporting modules

| Module | Interface | Purpose |
|--------|-----------|---------|
| `config.py` | `load_config(path) → AppConfig` | Pydantic-validated config from `config.yaml`. All models, parameters, and paths configurable |
| `health_check.py` | `check_ollama()`, `check_models(required)` | Async HTTP checks for Ollama connectivity and model availability |
| `models.py` | `Document`, `Chunk`, `SearchResult` | Shared dataclasses used across pipeline stages |

## App entry point (`app.py`)

The Chainlit app wires modules together via three hooks:

- **`on_chat_start`** — loads config, runs health checks, sets up ChromaDB + vector store, auto-ingests documents, builds BM25 index and hybrid retriever, loads reranker and condenser, renders settings panel and re-ingest button
- **`on_message`** — condenses follow-ups → hybrid retrieval → reranking → streaming answer with citations and source elements. Wrapped in collapsible Chainlit steps
- **`on_settings_update`** — re-scans and re-ingests when folder path or recursive toggle changes

Additional handlers: `on_reingest` (action callback for the re-ingest button).

## LangChain boundary

LangChain is used selectively in exactly two places:

1. **Document loading** (`loader.py`) — PyPDFLoader, Docx2txtLoader, TextLoader
2. **Text splitting** (`chunker.py`) — RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

All other components — embeddings, vector store operations, BM25, fusion, reranking, prompts, LLM calls, session memory — use direct library calls. This is a deliberate design choice to limit framework coupling.

## Data flow

1. **Startup**: Scanner discovers files → computes MD5 hashes → compares against stored `file_hash` in ChromaDB → loads only new/changed files → chunks with metadata → embeds via Ollama → stores in ChromaDB. BM25 index rebuilt from all stored texts.

2. **Query**: User message → condenser rewrites if chat history exists → BM25 returns top 20 → semantic search returns top 20 → RRF fuses to top 30 → cross-encoder reranks to top 10 → answerer builds prompt with source context → Ollama streams response → citations and source elements displayed.

3. **Follow-up**: Same as query, but condenser first rewrites "tell me more" into "What additional details are available about [topic from chat history]?" using the LLM. The standalone question then follows the normal retrieval path. Chat history is tracked per session but not passed to the answer generation prompt.

## Storage

- **ChromaDB** — persisted at `~/.multi_doc_query/chroma_db/` (configurable). Single collection with cosine similarity. Chunk metadata includes: `filename`, `relative_path`, `doc_type`, `page_number`, `section_header`, `chunk_index`, `doc_hash`, `file_hash`.
- **BM25 index** — in-memory, rebuilt from ChromaDB on each startup (~0.02s for 1000 chunks).
- **Reranker model** — cached locally by HuggingFace Hub after first download (~2.3 GB).

## Testing

99 tests covering all deterministic modules. Non-deterministic modules (Condenser, Answerer) are tested via mocked LLM calls for prompt construction and control flow, not output quality.

Tests use:
- In-memory ChromaDB (no filesystem side effects)
- Mocked Ollama/CrossEncoder calls (no model dependencies)
- Real file fixtures for document loading and chunking
- pytest with pytest-asyncio (`asyncio_mode = "auto"`)

CI runs on every push/PR to main via GitHub Actions.
