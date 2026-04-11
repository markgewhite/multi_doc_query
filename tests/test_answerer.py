from src.generation.answerer import build_prompt, build_source_elements
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


def test_build_source_elements_returns_list():
    """build_source_elements returns a list of dicts with name, content, display."""
    elements = build_source_elements(_make_results())
    assert len(elements) == 2
    for el in elements:
        assert "name" in el
        assert "content" in el
        assert "display" in el


def test_build_source_elements_name_format():
    """Element name follows 'Source: path | Page N' format."""
    results = [
        SearchResult(
            text="Some text.",
            metadata={"filename": "report.pdf", "relative_path": "reports/annual.pdf", "doc_type": "pdf", "page_number": 5},
            distance=0.1,
        ),
    ]
    elements = build_source_elements(results)
    assert elements[0]["name"] == "Source: reports/annual.pdf | Page 5"


def test_build_source_elements_preserves_order():
    """Elements maintain the same order as input results (relevance-ranked)."""
    elements = build_source_elements(_make_results())
    assert "Paris" in elements[0]["content"]
    assert "Berlin" in elements[1]["content"]


def test_build_source_elements_falls_back_to_filename():
    """Without relative_path, element name uses filename."""
    results = [
        SearchResult(
            text="Some text.",
            metadata={"filename": "old.pdf", "doc_type": "pdf", "page_number": 1},
            distance=0.1,
        ),
    ]
    elements = build_source_elements(results)
    assert elements[0]["name"] == "Source: old.pdf | Page 1"
