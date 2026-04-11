from src.models import SearchResult
from src.retrieval.fusion import reciprocal_rank_fusion


def _make_result(text, distance=0.0):
    return SearchResult(
        text=text,
        metadata={"filename": "test.pdf", "doc_type": "pdf", "page_number": 1, "chunk_index": 0, "doc_hash": "h1"},
        distance=distance,
    )


def test_rrf_merges_two_lists():
    """RRF merges two ranked lists into a single list."""
    list_a = [_make_result("A"), _make_result("B")]
    list_b = [_make_result("C"), _make_result("D")]
    merged = reciprocal_rank_fusion(list_a, list_b, k=60)
    texts = [r.text for r in merged]
    assert "A" in texts
    assert "C" in texts
    assert len(merged) == 4


def test_rrf_deduplicates():
    """Same chunk in both lists appears once with combined score."""
    list_a = [_make_result("A"), _make_result("B")]
    list_b = [_make_result("B"), _make_result("C")]
    merged = reciprocal_rank_fusion(list_a, list_b, k=60)
    texts = [r.text for r in merged]
    assert texts.count("B") == 1
    # B should rank highest since it appears in both lists
    assert merged[0].text == "B"


def test_rrf_respects_top_n():
    """RRF returns at most top_n results."""
    list_a = [_make_result(f"A{i}") for i in range(10)]
    list_b = [_make_result(f"B{i}") for i in range(10)]
    merged = reciprocal_rank_fusion(list_a, list_b, k=60, top_n=5)
    assert len(merged) == 5


def test_rrf_single_list():
    """RRF works with a single list (passthrough)."""
    results = [_make_result("A"), _make_result("B")]
    merged = reciprocal_rank_fusion(results, k=60)
    assert len(merged) == 2
    assert merged[0].text == "A"


def test_rrf_empty_lists():
    """RRF with empty lists returns empty."""
    merged = reciprocal_rank_fusion([], [], k=60)
    assert merged == []
