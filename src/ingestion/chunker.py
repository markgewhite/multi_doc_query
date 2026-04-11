import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import Chunk, Document

# --- LangChain boundary: RecursiveCharacterTextSplitter used for text splitting.
#     All other pipeline stages (embeddings, vector store, LLM calls) use
#     direct library calls, not LangChain. ---


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Split documents into chunks with preserved and extended metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks: list[Chunk] = []

    for doc in documents:
        doc_hash = hashlib.md5(doc.text.encode()).hexdigest()
        splits = splitter.split_text(doc.text)

        for i, text in enumerate(splits):
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "doc_hash": doc_hash,
                    },
                )
            )

    return chunks
