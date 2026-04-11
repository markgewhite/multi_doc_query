from src.config import RetrievalConfig
from src.models import SearchResult
from src.retrieval.bm25_index import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.vector_store import VectorStore


class HybridRetriever:
    """Orchestrates BM25 + semantic search with RRF fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        config: RetrievalConfig,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._config = config

    def retrieve(self, query: str) -> list[SearchResult]:
        """Run hybrid search: BM25 + semantic → RRF fusion."""
        bm25_results = self._bm25_index.search(query, k=self._config.bm25_top_k)
        semantic_results = self._vector_store.search(query, k=self._config.semantic_top_k)

        return reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            k=self._config.rrf_k,
            top_n=30,
        )
