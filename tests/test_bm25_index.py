from src.retrieval.bm25_index import BM25Index


def _build_index():
    """Build a BM25 index from a known corpus."""
    texts = [
        "The capital of France is Paris",
        "Berlin is the capital of Germany",
        "Python is a programming language",
        "Machine learning uses neural networks",
        "Paris has the Eiffel Tower",
    ]
    metadatas = [
        {"filename": "geo.pdf", "doc_type": "pdf", "page_number": 1, "chunk_index": i, "doc_hash": "h1"}
        for i in range(len(texts))
    ]
    index = BM25Index()
    index.build(texts, metadatas)
    return index


def test_build_and_search_returns_results():
    """Searching a built index returns non-empty results."""
    index = _build_index()
    results = index.search("capital of France")
    assert len(results) > 0


def test_search_returns_k_results():
    """search() respects the k parameter."""
    index = _build_index()
    results = index.search("capital", k=2)
    assert len(results) == 2


def test_search_relevance():
    """Query about France returns the France chunk first."""
    index = _build_index()
    results = index.search("capital of France", k=3)
    assert "France" in results[0].text or "Paris" in results[0].text


def test_search_empty_index():
    """Searching an unbuilt index returns empty list."""
    index = BM25Index()
    results = index.search("anything")
    assert results == []


def test_search_results_have_metadata():
    """Search results include correct metadata from the corpus."""
    index = _build_index()
    results = index.search("Python programming", k=1)
    assert results[0].metadata["filename"] == "geo.pdf"
    assert "chunk_index" in results[0].metadata
