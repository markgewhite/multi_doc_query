# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi Doc Query is a local multi-document RAG chatbot using Chainlit as the UI and Ollama for LLM/embedding inference. It supports PDF and DOCX ingestion, hybrid retrieval (BM25 + semantic via ChromaDB), reciprocal rank fusion, and cross-encoder reranking.

## Commands

```bash
# Install dependencies (uses uv)
uv sync --dev

# Run all tests
uv run python -m pytest -v

# Run a single test file
uv run python -m pytest tests/test_config.py -v

# Run the Chainlit app
uv run chainlit run app.py

# Prerequisites: Ollama must be running (`ollama serve`) with required models pulled
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

## Architecture

The app entry point is `app.py` (Chainlit hooks). Source code lives in `src/` with three pipeline stages as subpackages:

- **`src/ingestion/`** - Document loading (PDF, DOCX), chunking via LangChain text splitters
- **`src/retrieval/`** - Hybrid search: BM25 (rank-bm25) + semantic (ChromaDB + sentence-transformers), fused with RRF, then reranked (cross-encoder)
- **`src/generation/`** - LLM response generation via Ollama

Supporting modules:
- **`src/config.py`** - Pydantic-validated config loaded from `config.yaml` (models, chunking params, retrieval params, paths)
- **`src/health_check.py`** - Async Ollama connectivity and model availability checks (httpx)

## Key Details

- **Python 3.11+**, managed with **uv** (not pip)
- Config in `config.yaml` at project root; validated by Pydantic models in `src/config.py`
- Tests use **pytest** with **pytest-asyncio** (`asyncio_mode = "auto"` in pyproject.toml)
- CI runs via GitHub Actions (`.github/workflows/ci.yml`) on push/PR to main
- All inference is local via Ollama -- no cloud API keys needed
- ChromaDB data stored at `~/.multi_doc_query/chroma_db/` by default
