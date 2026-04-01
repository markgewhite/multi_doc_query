import uuid

from src.models import Chunk
from src.retrieval.vector_store import VectorStore


def _make_store(embed_fn):
    """Create a VectorStore with a unique collection name for test isolation."""
    return VectorStore(embed_fn=embed_fn, collection_name=f"test_{uuid.uuid4().hex}")


def test_add_chunks_stores_in_collection(fake_embed_fn, sample_chunks):
    """After add_chunks(), collection count matches the number of chunks."""
    store = _make_store(fake_embed_fn)
    store.add_chunks(sample_chunks)
    assert len(store.get_all_texts()) == 2


def test_get_all_texts_returns_stored_texts(fake_embed_fn, sample_chunks):
    """get_all_texts() returns all chunk texts that were added."""
    store = _make_store(fake_embed_fn)
    store.add_chunks(sample_chunks)
    texts = store.get_all_texts()
    assert set(texts) == {c.text for c in sample_chunks}


def test_has_document_true_after_add(fake_embed_fn, sample_chunks):
    """has_document returns True for a doc_hash that was added."""
    store = _make_store(fake_embed_fn)
    store.add_chunks(sample_chunks)
    assert store.has_document("abc123") is True


def test_has_document_false_when_absent(fake_embed_fn):
    """has_document returns False for an unknown doc_hash."""
    store = _make_store(fake_embed_fn)
    assert store.has_document("nonexistent") is False


def test_search_returns_k_results(fake_embed_fn):
    """search(query, k=2) returns exactly 2 results when store has >= 2 chunks."""
    chunks = [
        Chunk(text=f"Chunk number {i}", metadata={"filename": "t.pdf", "doc_type": "pdf", "page_number": 1, "chunk_index": i, "doc_hash": "h1"})
        for i in range(5)
    ]
    store = _make_store(fake_embed_fn)
    store.add_chunks(chunks)
    results = store.search("query", k=2)
    assert len(results) == 2


def test_search_returns_fewer_when_store_small(fake_embed_fn):
    """search(query, k=5) returns only 2 results when store has 2 chunks."""
    chunks = [
        Chunk(text=f"Chunk {i}", metadata={"filename": "t.pdf", "doc_type": "pdf", "page_number": 1, "chunk_index": i, "doc_hash": "h1"})
        for i in range(2)
    ]
    store = _make_store(fake_embed_fn)
    store.add_chunks(chunks)
    results = store.search("query", k=5)
    assert len(results) == 2


def test_search_results_have_text_and_metadata(fake_embed_fn, sample_chunks):
    """Each search result has text and metadata fields."""
    store = _make_store(fake_embed_fn)
    store.add_chunks(sample_chunks)
    results = store.search("AI", k=1)
    assert results[0].text
    assert "filename" in results[0].metadata
    assert "page_number" in results[0].metadata
