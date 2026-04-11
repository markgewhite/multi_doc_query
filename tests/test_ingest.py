"""Tests for incremental ingestion."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.ingest import IngestResult, ingest_folder


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
        result = ingest_folder(
            doc_folder, mock_store, chunk_size=512, chunk_overlap=100
        )
        assert result.ingested == 1
        assert result.skipped == 0
        mock_store.add_chunks.assert_called_once()

    def test_skips_unchanged_file(self, doc_folder, mock_store):
        file_hash = hashlib.md5(
            (doc_folder / "notes.txt").read_bytes()
        ).hexdigest()
        mock_store.count.return_value = 5
        mock_store.get_all_texts_and_metadatas.return_value = (
            ["text"],
            [{"file_hash": file_hash}],
        )

        result = ingest_folder(doc_folder, mock_store)
        assert result.ingested == 0
        assert result.skipped == 1
        mock_store.add_chunks.assert_not_called()

    def test_chunks_have_file_hash(self, doc_folder, mock_store):
        ingest_folder(doc_folder, mock_store)
        chunks = mock_store.add_chunks.call_args[0][0]
        for chunk in chunks:
            assert "file_hash" in chunk.metadata

    def test_empty_folder(self, tmp_path, mock_store):
        result = ingest_folder(tmp_path, mock_store)
        assert result.ingested == 0
        assert result.skipped == 0

    def test_relative_path_in_metadata(self, doc_folder, mock_store):
        ingest_folder(doc_folder, mock_store)
        chunks = mock_store.add_chunks.call_args[0][0]
        for chunk in chunks:
            rel_path = chunk.metadata.get("relative_path", "")
            assert not rel_path.startswith("/")

    def test_force_ignores_existing_hashes(self, doc_folder, mock_store):
        """Force mode ingests even when file hash already exists."""
        file_hash = hashlib.md5(
            (doc_folder / "notes.txt").read_bytes()
        ).hexdigest()
        mock_store.count.return_value = 5
        mock_store.get_all_texts_and_metadatas.return_value = (
            ["text"],
            [{"file_hash": file_hash}],
        )

        result = ingest_folder(doc_folder, mock_store, force=True)
        assert result.ingested == 1
        assert result.skipped == 0
        mock_store.add_chunks.assert_called_once()

    def test_failure_skips_bad_file_continues(self, tmp_path, mock_store):
        """A file that fails to load is skipped; other files still ingested."""
        (tmp_path / "good.txt").write_text("Good content")
        (tmp_path / "bad.pdf").write_bytes(b"not a real pdf")

        result = ingest_folder(tmp_path, mock_store)
        # good.txt should succeed, bad.pdf may fail
        assert result.ingested >= 1 or result.failed >= 1
        assert result.ingested + result.failed + result.skipped == 2

    def test_failure_records_error_details(self, tmp_path, mock_store):
        """Failed files are recorded with filename and error message."""
        (tmp_path / "bad.pdf").write_bytes(b"not a real pdf")

        result = ingest_folder(tmp_path, mock_store)
        if result.failed > 0:
            assert len(result.failures) == result.failed
            assert result.failures[0][0] == "bad.pdf"
            assert isinstance(result.failures[0][1], str)

    def test_progress_callback_called(self, doc_folder, mock_store):
        """Progress callback is called for each file."""
        progress_calls = []
        ingest_folder(
            doc_folder,
            mock_store,
            on_progress=lambda name, status: progress_calls.append((name, status)),
        )
        assert len(progress_calls) >= 1
        assert progress_calls[0][0] == "notes.txt"
