from pathlib import Path

from src.ingestion.loader import load_folder

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_single_pdf_returns_documents():
    """Loading a folder with one PDF returns a non-empty list of Documents."""
    docs = load_folder(FIXTURES)
    assert len(docs) > 0


def test_document_has_text():
    """Each Document has non-empty text."""
    docs = load_folder(FIXTURES)
    for doc in docs:
        assert doc.text.strip()


def test_document_metadata_filename():
    """Each Document has the correct filename in metadata."""
    docs = load_folder(FIXTURES)
    filenames = {doc.metadata["filename"] for doc in docs}
    assert "sample.pdf" in filenames


def test_document_metadata_doc_type():
    """Each Document has doc_type='pdf' in metadata."""
    docs = load_folder(FIXTURES)
    for doc in docs:
        assert doc.metadata["doc_type"] == "pdf"


def test_document_metadata_page_number():
    """Each Document has a 1-indexed page_number in metadata."""
    docs = load_folder(FIXTURES)
    page_numbers = [doc.metadata["page_number"] for doc in docs]
    assert 1 in page_numbers
    assert all(isinstance(p, int) and p >= 1 for p in page_numbers)


def test_load_folder_empty_dir(tmp_path):
    """An empty folder returns an empty list."""
    docs = load_folder(tmp_path)
    assert docs == []


def test_load_folder_recursive(tmp_path):
    """With recursive=True, finds PDFs in subdirectories."""
    import shutil

    sub = tmp_path / "subdir"
    sub.mkdir()
    shutil.copy(FIXTURES / "sample.pdf", sub / "nested.pdf")

    docs = load_folder(tmp_path, recursive=True)
    filenames = {doc.metadata["filename"] for doc in docs}
    assert "nested.pdf" in filenames


def test_load_folder_non_recursive(tmp_path):
    """With recursive=False, ignores PDFs in subdirectories."""
    import shutil

    sub = tmp_path / "subdir"
    sub.mkdir()
    shutil.copy(FIXTURES / "sample.pdf", sub / "nested.pdf")

    docs = load_folder(tmp_path, recursive=False)
    assert docs == []
