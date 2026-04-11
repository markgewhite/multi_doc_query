from src.models import SearchResult


def reciprocal_rank_fusion(
    *ranked_lists: list[SearchResult],
    k: int = 60,
    top_n: int = 30,
) -> list[SearchResult]:
    """Merge ranked lists using Reciprocal Rank Fusion.

    Score for each result = sum(1 / (k + rank)) across all lists it appears in.
    Deduplicates by chunk text. Returns top_n results sorted by RRF score.
    """
    scores: dict[str, float] = {}
    results_by_text: dict[str, SearchResult] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list):
            scores[result.text] = scores.get(result.text, 0.0) + 1.0 / (k + rank + 1)
            if result.text not in results_by_text:
                results_by_text[result.text] = result

    sorted_texts = sorted(scores, key=lambda t: scores[t], reverse=True)[:top_n]

    return [
        SearchResult(
            text=results_by_text[text].text,
            metadata=results_by_text[text].metadata,
            distance=scores[text],
        )
        for text in sorted_texts
    ]
