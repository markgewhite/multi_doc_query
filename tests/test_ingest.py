"""Tests for incremental ingestion."""

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.ingest import ingest_folder


@pytest.fixture
def doc_folder(tmp_path):
    """Create a folder with a test text file."""
    (tmp_path / "notes.txt").write_text("Some notes about testing.")
    return tmp_path


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.count.return_value = 0
    store.get_all_texts_and_metadatas.return_value = ([], [])
    return store


class TestIngestFolder:
    def test_ingests_new_file(self, doc_folder, mock_store):
        ingested, skipped = ingest_folder(
            doc_folder, mock_store, chunk_size=512, chunk_overlap=100
        )
        assert ingested == 1
        assert skipped == 0
        mock_store.add_chunks.assert_called_once()

    def test_skips_unchanged_file(self, doc_folder, mock_store):
        # Simulate file already in store by returning its hash
        import hashlib
        file_hash = hashlib.md5(
            (doc_folder / "notes.txt").read_bytes()
        ).hexdigest()
        mock_store.count.return_value = 5
        mock_store.get_all_texts_and_metadatas.return_value = (
            ["text"],
            [{"file_hash": file_hash}],
        )

        ingested, skipped = ingest_folder(doc_folder, mock_store)
        assert ingested == 0
        assert skipped == 1
        mock_store.add_chunks.assert_not_called()

    def test_chunks_have_file_hash(self, doc_folder, mock_store):
        ingest_folder(doc_folder, mock_store)
        chunks = mock_store.add_chunks.call_args[0][0]
        for chunk in chunks:
            assert "file_hash" in chunk.metadata

    def test_empty_folder(self, tmp_path, mock_store):
        ingested, skipped = ingest_folder(tmp_path, mock_store)
        assert ingested == 0
        assert skipped == 0

    def test_relative_path_in_metadata(self, doc_folder, mock_store):
        ingest_folder(doc_folder, mock_store)
        chunks = mock_store.add_chunks.call_args[0][0]
        for chunk in chunks:
            rel_path = chunk.metadata.get("relative_path", "")
            assert not rel_path.startswith("/")
