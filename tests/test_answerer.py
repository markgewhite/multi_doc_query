from src.generation.answerer import build_prompt
from src.retrieval.vector_store import SearchResult


def _make_results():
    return [
        SearchResult(
            text="Paris is the capital of France.",
            metadata={"filename": "geo.pdf", "doc_type": "pdf", "page_number": 5},
            distance=0.1,
        ),
        SearchResult(
            text="Berlin is the capital of Germany.",
            metadata={"filename": "europe.pdf", "doc_type": "pdf", "page_number": 12},
            distance=0.2,
        ),
    ]


def test_build_prompt_includes_context():
    """The prompt includes the text from search results."""
    messages = build_prompt("What is the capital of France?", _make_results())
    content = " ".join(m["content"] for m in messages)
    assert "Paris is the capital of France." in content
    assert "Berlin is the capital of Germany." in content


def test_build_prompt_includes_question():
    """The prompt includes the user's question."""
    messages = build_prompt("What is the capital of France?", _make_results())
    content = " ".join(m["content"] for m in messages)
    assert "What is the capital of France?" in content


def test_build_prompt_source_format():
    """Context is formatted with source citations: --- Source: filename | Page N ---"""
    messages = build_prompt("question", _make_results())
    content = " ".join(m["content"] for m in messages)
    assert "--- Source: geo.pdf | Page 5 ---" in content
    assert "--- Source: europe.pdf | Page 12 ---" in content
