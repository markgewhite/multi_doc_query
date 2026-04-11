import hashlib

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.models import Chunk, Document

# --- LangChain boundary: RecursiveCharacterTextSplitter and
#     MarkdownHeaderTextSplitter used for text splitting.
#     All other pipeline stages (embeddings, vector store, LLM calls)
#     use direct library calls, not LangChain. ---

MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Split documents into chunks with preserved and extended metadata."""
    chunks: list[Chunk] = []

    for doc in documents:
        doc_hash = hashlib.md5(doc.text.encode()).hexdigest()

        if doc.metadata.get("doc_type") == "md":
            doc_chunks = _chunk_markdown(doc, chunk_size, chunk_overlap)
        else:
            doc_chunks = _chunk_text(doc, chunk_size, chunk_overlap)

        for i, (text, extra_meta) in enumerate(doc_chunks):
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        **doc.metadata,
                        **extra_meta,
                        "chunk_index": i,
                        "doc_hash": doc_hash,
                    },
                )
            )

    return chunks


def _chunk_text(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> list[tuple[str, dict]]:
    """Split non-Markdown text using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    splits = splitter.split_text(doc.text)
    return [(text, {}) for text in splits]


def _chunk_markdown(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> list[tuple[str, dict]]:
    """Split Markdown by headers first, then by size."""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )
    header_splits = md_splitter.split_text(doc.text)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    result: list[tuple[str, dict]] = []
    for split in header_splits:
        section_header = _build_section_header(split.metadata)
        sub_splits = size_splitter.split_text(split.page_content)
        for text in sub_splits:
            extra = {"section_header": section_header} if section_header else {}
            result.append((text, extra))

    return result


def _build_section_header(metadata: dict[str, str]) -> str:
    """Build a section header string like 'Guide > Setup > Prerequisites'."""
    parts = []
    for key in ("h1", "h2", "h3", "h4"):
        if key in metadata:
            parts.append(metadata[key])
    return " > ".join(parts)
