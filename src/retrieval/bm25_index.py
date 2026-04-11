from rank_bm25 import BM25Okapi

from src.models import SearchResult

# --- Not via LangChain: BM25 keyword search uses rank_bm25 directly. ---


class BM25Index:
    """BM25 keyword search index built from chunk texts."""

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def build(self, texts: list[str], metadatas: list[dict]) -> None:
        """Build the BM25 index from chunk texts and their metadata."""
        self._texts = texts
        self._metadatas = metadatas
        tokenised = [text.lower().split() for text in texts]
        self._index = BM25Okapi(tokenised)

    def search(self, query: str, k: int = 20) -> list[SearchResult]:
        """Return top-k results ranked by BM25 score."""
        if self._index is None or not self._texts:
            return []

        tokenised_query = query.lower().split()
        scores = self._index.get_scores(tokenised_query)

        # Get top-k indices sorted by score descending
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            SearchResult(
                text=self._texts[i],
                metadata=self._metadatas[i],
                distance=float(scores[i]),
            )
            for i in ranked
            if scores[i] > 0
        ]
