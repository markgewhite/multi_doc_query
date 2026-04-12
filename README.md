# Multi Doc Query

A locally-hosted multi-document RAG chatbot that answers natural language questions across a collection of private documents. Runs entirely offline — no data leaves your machine.

## Features

- **Multi-format ingestion** — PDF, DOCX, TXT, and Markdown files with recursive folder scanning
- **Hybrid retrieval** — BM25 keyword search + semantic vector search fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** — BAAI/bge-reranker-v2-m3 scores query-chunk pairs for precision
- **Streaming answers** — token-by-token response via local LLM (Ollama)
- **Inline citations** — answers cite source documents with filenames and page numbers
- **Expandable sources** — click to view the retrieved chunks that informed each answer
- **Follow-up questions** — condense-then-retrieve rewrites vague follow-ups into standalone queries
- **Incremental ingestion** — MD5 hash change detection skips unchanged files on re-scan
- **Conflict detection** — highlights when sources from different documents disagree
- **Retrieval transparency** — collapsible step display shows searching, reranking, and candidate counts
- **Settings panel** — configure document folder and recursive scanning from the UI
- **Manual re-ingest** — action button to force a full refresh with progress and failure summary

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design, module interfaces, and pipeline diagram.

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — package manager
- **[Ollama](https://ollama.ai/)** — local LLM inference

Pull the required models:

```bash
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

The cross-encoder reranker (BAAI/bge-reranker-v2-m3) is downloaded automatically from HuggingFace on first run.

## Installation

```bash
git clone https://github.com/markgewhite/multi_doc_query.git
cd multi_doc_query
uv sync
```

## Configuration

Edit `config.yaml` in the project root:

```yaml
models:
  llm: "llama3.1:8b"
  embedding: "mxbai-embed-large"

chunking:
  chunk_size: 512
  chunk_overlap: 100

retrieval:
  bm25_top_k: 20
  semantic_top_k: 20
  rrf_k: 60
  rerank_top_k: 10
  reranker_model: "BAAI/bge-reranker-v2-m3"

paths:
  chroma_db: "~/.multi_doc_query/chroma_db/"
  documents: "~/path/to/your/documents/"

scanning:
  recursive: true
```

All model choices, chunking parameters, and retrieval settings are configurable without code changes. The document folder path can also be set from the UI settings panel.

## Usage

Start Ollama, then launch the app:

```bash
ollama serve
uv run chainlit run app.py
```

Open http://localhost:8000 in your browser.

1. Set your documents folder (via `config.yaml` or the settings panel gear icon)
2. Documents are automatically ingested on startup
3. Ask questions — answers stream in with inline citations
4. Click source elements to view the retrieved chunks
5. Ask follow-up questions naturally ("tell me more about that")
6. Use the Re-ingest button to force a full refresh after adding new documents

## Running Tests

```bash
uv run python -m pytest -v
```

99 tests cover all deterministic modules. CI runs automatically on push/PR via GitHub Actions.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | [Chainlit](https://chainlit.io/) |
| LLM | [Ollama](https://ollama.ai/) (llama3.1:8b) |
| Embeddings | Ollama (mxbai-embed-large) |
| Reranker | [sentence-transformers](https://www.sbert.net/) CrossEncoder (BAAI/bge-reranker-v2-m3) |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| Keyword search | [rank-bm25](https://github.com/dorianbrown/rank_bm25) |
| Document loading | [LangChain](https://www.langchain.com/) (loaders and text splitters only) |
| Config | [Pydantic](https://docs.pydantic.dev/) + YAML |
| Package manager | [uv](https://docs.astral.sh/uv/) |

## Licence

Private project — not currently licensed for redistribution.
