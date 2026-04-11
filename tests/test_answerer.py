from src.generation.answerer import build_prompt
from src.retrieval.vector_store import SearchResult


def _make_results():
    return [
        SearchResult(
            text="Paris is the capital of France.",
            metadata={"filename": "geo.pdf", "relative_path": "countries/geo.pdf", "doc_type": "pdf", "page_number": 5},
            distance=0.1,
        ),
        SearchResult(
            text="Berlin is the capital of Germany.",
            metadata={"filename": "europe.pdf", "relative_path": "countries/europe.pdf", "doc_type": "pdf", "page_number": 12},
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
    """Context is formatted with source citations using relative path."""
    messages = build_prompt("question", _make_results())
    content = " ".join(m["content"] for m in messages)
    assert "--- Source: countries/geo.pdf | Page 5 ---" in content
    assert "--- Source: countries/europe.pdf | Page 12 ---" in content


def test_build_prompt_uses_relative_path():
    """build_prompt prefers relative_path over filename in source headers."""
    results = [
        SearchResult(
            text="Some text.",
            metadata={"filename": "report.pdf", "relative_path": "reports/annual.pdf", "doc_type": "pdf", "page_number": 3},
            distance=0.1,
        ),
    ]
    messages = build_prompt("question", results)
    content = " ".join(m["content"] for m in messages)
    assert "--- Source: reports/annual.pdf | Page 3 ---" in content
    assert "--- Source: report.pdf" not in content


def test_build_prompt_falls_back_to_filename():
    """Without relative_path, build_prompt uses filename."""
    results = [
        SearchResult(
            text="Some text.",
            metadata={"filename": "legacy.pdf", "doc_type": "pdf", "page_number": 1},
            distance=0.1,
        ),
    ]
    messages = build_prompt("question", results)
    content = " ".join(m["content"] for m in messages)
    assert "--- Source: legacy.pdf | Page 1 ---" in content
