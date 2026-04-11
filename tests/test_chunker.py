from src.ingestion.chunker import chunk_documents
from src.models import Document


def test_chunk_short_document_single_chunk():
    """A document shorter than chunk_size returns exactly one chunk."""
    doc = Document(
        text="Short text.",
        metadata={"filename": "test.pdf", "doc_type": "pdf", "page_number": 1},
    )
    chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_chunk_long_document_multiple_chunks():
    """A document longer than chunk_size returns multiple chunks."""
    doc = Document(
        text="word " * 200,  # 1000 chars, well over chunk_size=100
        metadata={"filename": "test.pdf", "doc_type": "pdf", "page_number": 1},
    )
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunk_size_within_limit():
    """Each chunk's text length does not exceed chunk_size."""
    doc = Document(
        text="word " * 200,
        metadata={"filename": "test.pdf", "doc_type": "pdf", "page_number": 1},
    )
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
    for chunk in chunks:
        assert len(chunk.text) <= 100


def test_chunk_preserves_source_metadata():
    """Each chunk retains filename, doc_type, page_number from its source."""
    doc = Document(
        text="Some content here.",
        metadata={"filename": "report.pdf", "doc_type": "pdf", "page_number": 3},
    )
    chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=100)
    assert chunks[0].metadata["filename"] == "report.pdf"
    assert chunks[0].metadata["doc_type"] == "pdf"
    assert chunks[0].metadata["page_number"] == 3


def test_chunk_has_chunk_index():
    """Chunks have sequential 0-based chunk_index."""
    doc = Document(
        text="word " * 200,
        metadata={"filename": "test.pdf", "doc_type": "pdf", "page_number": 1},
    )
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_chunk_has_doc_hash():
    """Each chunk has a doc_hash (MD5 of the source document text)."""
    doc = Document(
        text="Unique document content.",
        metadata={"filename": "test.pdf", "doc_type": "pdf", "page_number": 1},
    )
    chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=100)
    assert "doc_hash" in chunks[0].metadata
    assert isinstance(chunks[0].metadata["doc_hash"], str)
    assert len(chunks[0].metadata["doc_hash"]) == 32  # MD5 hex length


def test_chunk_markdown_has_section_header():
    """Markdown documents produce chunks with section_header metadata."""
    md_text = "# Introduction\n\nThis is the intro.\n\n## Methods\n\nThese are the methods."
    doc = Document(
        text=md_text,
        metadata={"filename": "README.md", "doc_type": "md", "page_number": 1},
    )
    chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=0)
    headers = [c.metadata.get("section_header") for c in chunks]
    assert any(h and "Introduction" in h for h in headers)
    assert any(h and "Methods" in h for h in headers)


def test_chunk_markdown_nested_headers():
    """Nested Markdown headers produce 'H1 > H2' format in section_header."""
    md_text = "# Guide\n\n## Setup\n\n### Prerequisites\n\nYou need Python."
    doc = Document(
        text=md_text,
        metadata={"filename": "guide.md", "doc_type": "md", "page_number": 1},
    )
    chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=0)
    headers = [c.metadata.get("section_header", "") for c in chunks]
    assert any("Guide" in h and "Setup" in h and "Prerequisites" in h for h in headers)


def test_chunk_non_markdown_no_section_header():
    """Non-Markdown documents don't get section_header metadata."""
    doc = Document(
        text="Some plain text content.",
        metadata={"filename": "notes.txt", "doc_type": "txt", "page_number": 1},
    )
    chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=0)
    assert "section_header" not in chunks[0].metadata
