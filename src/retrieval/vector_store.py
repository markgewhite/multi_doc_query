from collections.abc import Callable
from dataclasses import dataclass

import chromadb

from src.models import Chunk, SearchResult

# Type alias for the embedding function dependency.
# Not via LangChain — embeddings use direct ollama.embed() in production.
EmbedFn = Callable[[list[str]], list[list[float]]]


class VectorStore:
    """ChromaDB-backed vector store with injected embedding function."""

    def __init__(
        self,
        embed_fn: EmbedFn,
        client: chromadb.ClientAPI | None = None,
        collection_name: str = "documents",
    ):
        if client is None:
            client = chromadb.Client()
        self._client = client
        self._embed_fn = embed_fn
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and store chunks in ChromaDB."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._embed_fn(texts)
        metadatas = [c.metadata for c in chunks]
        ids = [f"chunk_{i}" for i in range(
            self._collection.count(),
            self._collection.count() + len(chunks),
        )]

        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the top-k most similar chunks for a query."""
        query_embedding = self._embed_fn([query])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
        )

        search_results = []
        for i in range(len(results["documents"][0])):
            search_results.append(
                SearchResult(
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    distance=results["distances"][0][i],
                )
            )
        return search_results

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self._collection.count()

    def get_all_texts_and_metadatas(self) -> tuple[list[str], list[dict]]:
        """Return all stored chunk texts and their metadata."""
        result = self._collection.get()
        return result["documents"], result["metadatas"]

    def get_all_texts(self) -> list[str]:
        """Return all stored chunk texts."""
        texts, _ = self.get_all_texts_and_metadatas()
        return texts

    def has_document(self, doc_hash: str) -> bool:
        """Check if any chunk with the given doc_hash exists."""
        result = self._collection.get(
            where={"doc_hash": doc_hash},
        )
        return len(result["ids"]) > 0
