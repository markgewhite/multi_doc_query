from unittest.mock import MagicMock

from src.config import RetrievalConfig
from src.models import SearchResult
from src.retrieval.hybrid import HybridRetriever


def _make_result(text, distance=0.0):
    return SearchResult(
        text=text,
        metadata={"filename": "test.pdf", "doc_type": "pdf", "page_number": 1, "chunk_index": 0, "doc_hash": "h1"},
        distance=distance,
    )


def test_retrieve_combines_bm25_and_semantic():
    """HybridRetriever calls both BM25 and semantic search, fuses results."""
    bm25 = MagicMock()
    bm25.search.return_value = [_make_result("BM25 only"), _make_result("Shared")]

    store = MagicMock()
    store.search.return_value = [_make_result("Semantic only"), _make_result("Shared")]

    config = RetrievalConfig(bm25_top_k=20, semantic_top_k=20, rrf_k=60)
    retriever = HybridRetriever(vector_store=store, bm25_index=bm25, config=config)

    results = retriever.retrieve("test query")

    bm25.search.assert_called_once_with("test query", k=20)
    store.search.assert_called_once_with("test query", k=20)
    texts = [r.text for r in results]
    assert "Shared" in texts
    assert "BM25 only" in texts
    assert "Semantic only" in texts
    # Shared should rank highest (appears in both lists)
    assert results[0].text == "Shared"


def test_retrieve_returns_limited_results():
    """HybridRetriever returns at most the configured number of results."""
    bm25 = MagicMock()
    bm25.search.return_value = [_make_result(f"BM25_{i}") for i in range(20)]

    store = MagicMock()
    store.search.return_value = [_make_result(f"Sem_{i}") for i in range(20)]

    config = RetrievalConfig(bm25_top_k=20, semantic_top_k=20, rrf_k=60)
    retriever = HybridRetriever(vector_store=store, bm25_index=bm25, config=config)

    results = retriever.retrieve("query")
    assert len(results) <= 30
