"""Cross-encoder reranker for search result refinement.

Not via LangChain: uses sentence-transformers CrossEncoder directly
for full control over scoring and result construction.
"""

from sentence_transformers import CrossEncoder

from src.models import SearchResult


class Reranker:
    """Scores query-chunk pairs with a cross-encoder and returns top results."""

    def __init__(self, model_name: str) -> None:
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Score all results against the query and return top_k by relevance."""
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            SearchResult(text=r.text, metadata=r.metadata, distance=float(score))
            for score, r in scored[:top_k]
        ]
