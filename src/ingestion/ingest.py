"""Incremental document ingestion with change detection and failure handling."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_file
from src.ingestion.scanner import compute_file_hash, scan_folder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]
"""Callback(filename, status) called per file. Status: 'processing', 'ingested', 'skipped', 'failed'."""


@dataclass
class IngestResult:
    """Summary of an ingestion run."""

    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    """List of (filename, error_message) for failed files."""


def ingest_folder(
    folder: Path,
    store: VectorStore,
    *,
    recursive: bool = True,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> IngestResult:
    """Scan a folder and ingest new or changed documents.

    Args:
        folder: Path to the documents folder.
        store: Vector store to add chunks to.
        recursive: Whether to scan subdirectories.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between chunks.
        force: If True, re-ingest all files regardless of hash.
        on_progress: Optional callback for per-file progress.

    Returns:
        IngestResult with counts of ingested, skipped, and failed files.
    """
    files = scan_folder(folder, recursive=recursive)
    result = IngestResult()

    if not files:
        return result

    existing_hashes = set() if force else _get_stored_file_hashes(store)

    for file_path in files:
        filename = file_path.name

        if on_progress:
            on_progress(filename, "processing")

        file_hash = compute_file_hash(file_path)
        if not force and file_hash in existing_hashes:
            result.skipped += 1
            if on_progress:
                on_progress(filename, "skipped")
            continue

        try:
            docs = load_file(file_path, folder)
            if not docs:
                result.skipped += 1
                if on_progress:
                    on_progress(filename, "skipped")
                continue

            for doc in docs:
                doc.metadata["file_hash"] = file_hash

            chunks = chunk_documents(
                docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk in chunks:
                chunk.metadata["file_hash"] = file_hash

            store.add_chunks(chunks)
            result.ingested += 1
            if on_progress:
                on_progress(filename, "ingested")
            logger.info("Ingested %s (%d chunks)", filename, len(chunks))

        except Exception as e:
            result.failed += 1
            error_msg = f"{type(e).__name__}: {e}"
            result.failures.append((filename, error_msg))
            if on_progress:
                on_progress(filename, "failed")
            logger.warning("Failed to ingest %s: %s", filename, error_msg)

    return result


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
