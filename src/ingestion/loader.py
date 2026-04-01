from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from src.models import Document

# --- LangChain boundary: PyPDFLoader used for PDF parsing.
#     This is one of two LangChain touch-points (the other is
#     RecursiveCharacterTextSplitter in chunker.py). All other
#     pipeline stages use direct library calls. ---


def load_folder(path: Path, *, recursive: bool = True) -> list[Document]:
    """Load all PDF files from a folder into Document objects."""
    path = Path(path)

    if recursive:
        pdf_files = sorted(path.rglob("*.pdf"))
    else:
        pdf_files = sorted(path.glob("*.pdf"))

    documents: list[Document] = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            documents.append(
                Document(
                    text=page.page_content,
                    metadata={
                        "filename": pdf_path.name,
                        "doc_type": "pdf",
                        "page_number": page.metadata.get("page", 0) + 1,
                    },
                )
            )

    return documents
