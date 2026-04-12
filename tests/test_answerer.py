from src.generation.answerer import (
    build_prompt,
    build_ref_map,
    build_reference_list,
    build_source_elements,
)
from src.models import SearchResult


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


def _make_results_same_doc():
    """Two chunks from the same document on different pages."""
    return [
        SearchResult(
            text="Chapter 1 content.",
            metadata={"filename": "report.pdf", "relative_path": "reports/annual.pdf", "doc_type": "pdf", "page_number": 5},
            distance=0.1,
        ),
        SearchResult(
            text="Chapter 3 content.",
            metadata={"filename": "report.pdf", "relative_path": "reports/annual.pdf", "doc_type": "pdf", "page_number": 22},
            distance=0.2,
        ),
        SearchResult(
            text="Berlin is the capital of Germany.",
            metadata={"filename": "europe.pdf", "relative_path": "countries/europe.pdf", "doc_type": "pdf", "page_number": 12},
            distance=0.3,
        ),
    ]


# --- build_prompt tests ---


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


def test_build_prompt_numeric_source_headers():
    """Context headers use numeric reference format [N]."""
    messages = build_prompt("question", _make_results())
    content = " ".join(m["content"] for m in messages)
    assert "--- [1, p. 5] ---" in content
    assert "--- [2, p. 12] ---" in content


def test_build_prompt_reference_list_appended():
    """A numbered reference list is appended after the context."""
    messages = build_prompt("question", _make_results())
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "[1] countries/geo.pdf" in user_content
    assert "[2] countries/europe.pdf" in user_content


def test_build_prompt_same_doc_shares_number():
    """Two chunks from the same document share the same reference number."""
    messages = build_prompt("question", _make_results_same_doc())
    content = " ".join(m["content"] for m in messages)
    # Both chunks from reports/annual.pdf should be [1]
    assert "--- [1, p. 5] ---" in content
    assert "--- [1, p. 22] ---" in content
    # europe.pdf should be [2]
    assert "--- [2, p. 12] ---" in content


def test_build_prompt_reference_list_deduplicates():
    """Reference list has one entry per unique document, not per chunk."""
    messages = build_prompt("question", _make_results_same_doc())
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    # Should have exactly one entry for reports/annual.pdf
    assert user_content.count("[1] reports/annual.pdf") == 1
    assert user_content.count("[2] countries/europe.pdf") == 1


def test_build_prompt_uses_relative_path_in_references():
    """Reference list uses relative_path, not filename."""
    results = [
        SearchResult(
            text="Some text.",
            metadata={"filename": "report.pdf", "relative_path": "reports/annual.pdf", "doc_type": "pdf", "page_number": 3},
            distance=0.1,
        ),
    ]
    messages = build_prompt("question", results)
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "[1] reports/annual.pdf" in user_content


def test_build_prompt_falls_back_to_filename():
    """Without relative_path, reference list uses filename."""
    results = [
        SearchResult(
            text="Some text.",
            metadata={"filename": "legacy.pdf", "doc_type": "pdf", "page_number": 1},
            distance=0.1,
        ),
    ]
    messages = build_prompt("question", results)
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "[1] legacy.pdf" in user_content


def test_build_prompt_section_header_in_source():
    """Markdown sources show section header instead of page number in context header."""
    results = [
        SearchResult(
            text="Install instructions.",
            metadata={
                "filename": "README.md",
                "relative_path": "README.md",
                "doc_type": "md",
                "page_number": 1,
                "section_header": "Getting Started > Installation",
            },
            distance=0.1,
        ),
    ]
    messages = build_prompt("How to install?", results)
    content = " ".join(m["content"] for m in messages)
    assert "--- [1, Section: Getting Started > Installation] ---" in content
    # Reference list should still show the document path
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "[1] README.md" in user_content


def test_build_prompt_system_prompt_numeric_format():
    """System prompt instructs the LLM to use [N, p. X] citation format."""
    messages = build_prompt("question", _make_results())
    system = next(m["content"] for m in messages if m["role"] == "system")
    assert "[1, p. 12]" in system or "[N, p." in system


# --- build_source_elements tests ---


def test_build_source_elements_returns_list():
    """build_source_elements returns a list of dicts with name, content, display."""
    elements = build_source_elements(_make_results())
    assert len(elements) == 2
    for el in elements:
        assert "name" in el
        assert "content" in el
        assert "display" in el


def test_build_source_elements_numeric_names():
    """Element names include numeric reference number."""
    elements = build_source_elements(_make_results())
    assert elements[0]["name"] == "[1] countries/geo.pdf, p. 5"
    assert elements[1]["name"] == "[2] countries/europe.pdf, p. 12"


def test_build_source_elements_same_doc_shares_number():
    """Elements from the same document share the same reference number."""
    elements = build_source_elements(_make_results_same_doc())
    assert elements[0]["name"] == "[1] reports/annual.pdf, p. 5"
    assert elements[1]["name"] == "[1] reports/annual.pdf, p. 22"
    assert elements[2]["name"] == "[2] countries/europe.pdf, p. 12"


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
    assert elements[0]["name"] == "[1] old.pdf, p. 1"


def test_build_source_elements_section_header():
    """Markdown source elements show section header."""
    results = [
        SearchResult(
            text="Install instructions.",
            metadata={
                "filename": "README.md",
                "relative_path": "README.md",
                "doc_type": "md",
                "page_number": 1,
                "section_header": "Getting Started > Installation",
            },
            distance=0.1,
        ),
    ]
    elements = build_source_elements(results)
    assert elements[0]["name"] == "[1] README.md, Section: Getting Started > Installation"


# --- build_ref_map tests ---


def test_build_ref_map_assigns_numbers():
    """build_ref_map assigns 1-based numbers to unique documents."""
    ref_map = build_ref_map(_make_results())
    assert ref_map == {"countries/geo.pdf": 1, "countries/europe.pdf": 2}


def test_build_ref_map_deduplicates():
    """Same document gets the same number regardless of how many chunks."""
    ref_map = build_ref_map(_make_results_same_doc())
    assert ref_map == {"reports/annual.pdf": 1, "countries/europe.pdf": 2}


def test_build_ref_map_extends_existing():
    """Passing an existing ref_map preserves those numbers and continues."""
    existing = {"countries/geo.pdf": 1}
    ref_map = build_ref_map(_make_results(), existing)
    # geo.pdf keeps number 1, europe.pdf gets 2
    assert ref_map["countries/geo.pdf"] == 1
    assert ref_map["countries/europe.pdf"] == 2


def test_build_ref_map_extends_with_new_docs():
    """New docs in second query get numbers after the existing max."""
    existing = {"old_doc.pdf": 1, "another.pdf": 2}
    results = [
        SearchResult(
            text="New content.",
            metadata={"filename": "new.pdf", "relative_path": "new.pdf", "doc_type": "pdf", "page_number": 1},
            distance=0.1,
        ),
    ]
    ref_map = build_ref_map(results, existing)
    assert ref_map["old_doc.pdf"] == 1
    assert ref_map["another.pdf"] == 2
    assert ref_map["new.pdf"] == 3


def test_build_ref_map_no_mutation():
    """build_ref_map does not mutate the existing dict passed in."""
    existing = {"countries/geo.pdf": 1}
    original = existing.copy()
    build_ref_map(_make_results(), existing)
    assert existing == original


# --- build_reference_list tests ---


def test_build_reference_list_format():
    """build_reference_list returns a formatted reference list string."""
    ref_map = {"countries/geo.pdf": 1, "countries/europe.pdf": 2}
    text = build_reference_list(ref_map)
    assert "[1] countries/geo.pdf" in text
    assert "[2] countries/europe.pdf" in text


def test_build_reference_list_sorted_by_number():
    """References are listed in numeric order."""
    ref_map = {"b.pdf": 2, "a.pdf": 1, "c.pdf": 3}
    text = build_reference_list(ref_map)
    lines = text.strip().split("\n")
    assert lines[0] == "[1] a.pdf"
    assert lines[1] == "[2] b.pdf"
    assert lines[2] == "[3] c.pdf"


def test_build_reference_list_only_for_given_numbers():
    """When filtering to specific numbers, only those appear."""
    ref_map = {"a.pdf": 1, "b.pdf": 2, "c.pdf": 3}
    text = build_reference_list(ref_map, only={1, 3})
    assert "[1] a.pdf" in text
    assert "[3] c.pdf" in text
    assert "[2]" not in text


# --- build_prompt with explicit ref_map ---


def test_build_prompt_with_ref_map_uses_existing_numbers():
    """build_prompt respects a pre-existing ref_map for numbering."""
    ref_map = {"countries/geo.pdf": 5, "countries/europe.pdf": 6}
    messages = build_prompt("question", _make_results(), ref_map=ref_map)
    content = " ".join(m["content"] for m in messages)
    assert "--- [5, p. 5] ---" in content
    assert "--- [6, p. 12] ---" in content


# --- build_source_elements with explicit ref_map ---


def test_build_source_elements_with_ref_map():
    """build_source_elements respects a pre-existing ref_map."""
    ref_map = {"countries/geo.pdf": 5, "countries/europe.pdf": 6}
    elements = build_source_elements(_make_results(), ref_map=ref_map)
    assert elements[0]["name"] == "[5] countries/geo.pdf, p. 5"
    assert elements[1]["name"] == "[6] countries/europe.pdf, p. 12"
