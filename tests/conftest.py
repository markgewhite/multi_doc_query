import hashlib

import pytest

from src.models import Chunk, Document


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic embedder: hash each text to a fixed-length vector."""
    result = []
    for t in texts:
        h = hashlib.md5(t.encode()).hexdigest()
        vec = [float(int(c, 16)) / 15.0 for c in h]
        result.append((vec * 12)[:384])
    return result


@pytest.fixture
def fake_embed_fn():
    return _fake_embed


@pytest.fixture
def sample_document():
    return Document(
        text="This is a test document about artificial intelligence and machine learning.",
        metadata={"filename": "test.pdf", "doc_type": "pdf", "page_number": 1},
    )


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            text="Chunk about AI and neural networks",
            metadata={
                "filename": "test.pdf",
                "doc_type": "pdf",
                "page_number": 1,
                "chunk_index": 0,
                "doc_hash": "abc123",
            },
        ),
        Chunk(
            text="Chunk about cooking recipes and ingredients",
            metadata={
                "filename": "test.pdf",
                "doc_type": "pdf",
                "page_number": 2,
                "chunk_index": 1,
                "doc_hash": "abc123",
            },
        ),
    ]
