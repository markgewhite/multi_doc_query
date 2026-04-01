from src.generation.answerer import build_prompt
from src.ingestion.chunker import chunk_documents
from src.models import Document
from src.retrieval.vector_store import VectorStore


def test_tracer_bullet_ingestion_to_prompt(fake_embed_fn):
    """Thinnest end-to-end path: document → chunk → store → search → prompt."""
    doc = Document(
        text="The capital of France is Paris.",
        metadata={"filename": "geo.pdf", "doc_type": "pdf", "page_number": 1},
    )

    chunks = chunk_documents([doc], chunk_size=9999, chunk_overlap=0)
    assert len(chunks) == 1

    store = VectorStore(embed_fn=fake_embed_fn)
    store.add_chunks(chunks)

    results = store.search("What is the capital of France?", k=1)
    assert len(results) == 1
    assert "Paris" in results[0].text

    messages = build_prompt("What is the capital of France?", results)
    prompt_text = " ".join(m["content"] for m in messages)
    assert "Paris" in prompt_text
    assert "capital of France" in prompt_text
