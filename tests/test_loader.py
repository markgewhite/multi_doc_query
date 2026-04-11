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
    """Each Document has a valid doc_type in metadata."""
    docs = load_folder(FIXTURES)
    for doc in docs:
        assert doc.metadata["doc_type"] in ("pdf", "docx", "txt", "md")


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


def test_document_metadata_relative_path():
    """Documents at the root have relative_path equal to filename."""
    docs = load_folder(FIXTURES)
    pdf_docs = [d for d in docs if d.metadata["doc_type"] == "pdf"]
    for doc in pdf_docs:
        assert doc.metadata["relative_path"] == "sample.pdf"


def test_document_metadata_relative_path_nested(tmp_path):
    """Documents in subdirectories have relative_path including the subdir."""
    import shutil

    sub = tmp_path / "reports"
    sub.mkdir()
    shutil.copy(FIXTURES / "sample.pdf", sub / "annual.pdf")

    docs = load_folder(tmp_path)
    paths = {doc.metadata["relative_path"] for doc in docs}
    assert "reports/annual.pdf" in paths


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


def test_load_docx_returns_documents():
    """Loading a folder with a DOCX returns Documents with text."""
    docs = load_folder(FIXTURES)
    docx_docs = [d for d in docs if d.metadata["doc_type"] == "docx"]
    assert len(docx_docs) > 0
    assert docx_docs[0].text.strip()


def test_docx_metadata_doc_type():
    """DOCX documents have doc_type='docx'."""
    docs = load_folder(FIXTURES)
    docx_docs = [d for d in docs if d.metadata["filename"] == "sample.docx"]
    assert len(docx_docs) > 0
    assert docx_docs[0].metadata["doc_type"] == "docx"


def test_docx_metadata_relative_path():
    """DOCX documents have correct relative_path."""
    docs = load_folder(FIXTURES)
    docx_docs = [d for d in docs if d.metadata["filename"] == "sample.docx"]
    assert docx_docs[0].metadata["relative_path"] == "sample.docx"


def test_load_txt_returns_documents():
    """Loading a folder with a TXT file returns Documents with doc_type='txt'."""
    docs = load_folder(FIXTURES)
    txt_docs = [d for d in docs if d.metadata["filename"] == "sample.txt"]
    assert len(txt_docs) > 0
    assert txt_docs[0].metadata["doc_type"] == "txt"
    assert txt_docs[0].text.strip()


def test_load_md_returns_documents():
    """Loading a folder with an MD file returns Documents with doc_type='md'."""
    docs = load_folder(FIXTURES)
    md_docs = [d for d in docs if d.metadata["filename"] == "sample.md"]
    assert len(md_docs) > 0
    assert md_docs[0].metadata["doc_type"] == "md"
    assert md_docs[0].text.strip()


def test_load_mixed_formats(tmp_path):
    """A folder with PDF, DOCX, TXT, and MD loads all formats."""
    import shutil

    for name in ("sample.pdf", "sample.docx", "sample.txt", "sample.md"):
        shutil.copy(FIXTURES / name, tmp_path / name)

    docs = load_folder(tmp_path)
    doc_types = {d.metadata["doc_type"] for d in docs}
    assert doc_types == {"pdf", "docx", "txt", "md"}
