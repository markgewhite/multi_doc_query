"""Tests for the folder scanner module."""

import hashlib
from pathlib import Path

import pytest

from src.ingestion.scanner import compute_file_hash, scan_folder


@pytest.fixture
def doc_folder(tmp_path):
    """Create a folder with test documents."""
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 test content")
    (tmp_path / "notes.txt").write_text("Some notes")
    (tmp_path / "readme.md").write_text("# Readme")
    (tmp_path / "letter.docx").write_bytes(b"PK docx content")
    (tmp_path / "image.png").write_bytes(b"PNG data")  # unsupported
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("Nested content")
    return tmp_path


class TestComputeFileHash:
    def test_hash_is_md5_of_contents(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        expected = hashlib.md5(b"hello world").hexdigest()
        assert compute_file_hash(f) == expected

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_file_hash(f1) != compute_file_hash(f2)


class TestScanFolder:
    def test_discovers_supported_files(self, doc_folder):
        files = scan_folder(doc_folder, recursive=True)
        extensions = {f.suffix for f in files}
        assert extensions == {".pdf", ".txt", ".md", ".docx"}

    def test_excludes_unsupported_files(self, doc_folder):
        files = scan_folder(doc_folder, recursive=True)
        names = {f.name for f in files}
        assert "image.png" not in names

    def test_recursive_finds_nested(self, doc_folder):
        files = scan_folder(doc_folder, recursive=True)
        names = {f.name for f in files}
        assert "nested.txt" in names

    def test_non_recursive_excludes_nested(self, doc_folder):
        files = scan_folder(doc_folder, recursive=False)
        names = {f.name for f in files}
        assert "nested.txt" not in names

    def test_returns_sorted_paths(self, doc_folder):
        files = scan_folder(doc_folder, recursive=True)
        assert files == sorted(files)

    def test_empty_folder(self, tmp_path):
        files = scan_folder(tmp_path, recursive=True)
        assert files == []
