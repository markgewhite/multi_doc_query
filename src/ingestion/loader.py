from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

from src.models import Document

# --- LangChain boundary: PyPDFLoader, Docx2txtLoader, and TextLoader used
#     for document parsing. This is one of two LangChain touch-points
#     (the other is text splitters in chunker.py). All other pipeline
#     stages use direct library calls. ---

SUPPORTED_EXTENSIONS = ("*.pdf", "*.docx", "*.txt", "*.md")

EXTENSION_TO_DOC_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "md",
}


def load_folder(path: Path, *, recursive: bool = True) -> list[Document]:
    """Load all supported documents from a folder."""
    path = Path(path)

    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        if recursive:
            files.extend(path.rglob(ext))
        else:
            files.extend(path.glob(ext))
    files.sort()

    documents: list[Document] = []

    for file_path in files:
        doc_type = EXTENSION_TO_DOC_TYPE[file_path.suffix.lower()]

        if doc_type == "pdf":
            documents.extend(_load_pdf(file_path, path))
        else:
            documents.extend(_load_single_file(file_path, path, doc_type))

    return documents


def _load_pdf(file_path: Path, root: Path) -> list[Document]:
    """Load a PDF file, returning one Document per page."""
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    return [
        Document(
            text=page.page_content,
            metadata={
                "filename": file_path.name,
                "relative_path": str(file_path.relative_to(root)),
                "doc_type": "pdf",
                "page_number": page.metadata.get("page", 0) + 1,
            },
        )
        for page in pages
    ]


def _load_single_file(
    file_path: Path, root: Path, doc_type: str
) -> list[Document]:
    """Load a DOCX, TXT, or MD file, returning one Document."""
    if doc_type == "docx":
        loader = Docx2txtLoader(str(file_path))
    else:
        loader = TextLoader(str(file_path), autodetect_encoding=True)

    pages = loader.load()
    text = "\n".join(page.page_content for page in pages)

    return [
        Document(
            text=text,
            metadata={
                "filename": file_path.name,
                "relative_path": str(file_path.relative_to(root)),
                "doc_type": doc_type,
                "page_number": 1,
            },
        )
    ]
