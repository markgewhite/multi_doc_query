"""Incremental document ingestion with change detection."""

import logging
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_file
from src.ingestion.scanner import compute_file_hash, scan_folder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_folder(
    folder: Path,
    store: VectorStore,
    *,
    recursive: bool = True,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> tuple[int, int]:
    """Scan a folder and ingest new or changed documents.

    Returns (ingested_count, skipped_count) — files ingested vs unchanged.
    """
    files = scan_folder(folder, recursive=recursive)

    if not files:
        return 0, 0

    existing_hashes = _get_stored_file_hashes(store)

    ingested = 0
    skipped = 0

    for file_path in files:
        file_hash = compute_file_hash(file_path)
        if file_hash in existing_hashes:
            skipped += 1
            continue

        docs = load_file(file_path, folder)
        if not docs:
            continue

        # Attach file_hash to document metadata before chunking
        for doc in docs:
            doc.metadata["file_hash"] = file_hash

        chunks = chunk_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Propagate file_hash to chunk metadata
        for chunk in chunks:
            chunk.metadata["file_hash"] = file_hash

        store.add_chunks(chunks)
        ingested += 1
        logger.info("Ingested %s (%d chunks)", file_path.name, len(chunks))

    return ingested, skipped


def _get_stored_file_hashes(store: VectorStore) -> set[str]:
    """Extract unique file_hash values from all stored chunks."""
    if store.count() == 0:
        return set()

    _, metadatas = store.get_all_texts_and_metadatas()
    return {
        m["file_hash"]
        for m in metadatas
        if "file_hash" in m
    }
