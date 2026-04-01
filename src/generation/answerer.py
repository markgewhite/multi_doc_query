from collections.abc import AsyncIterator

import ollama

from src.retrieval.vector_store import SearchResult

# --- Not via LangChain: prompt construction and LLM calls use direct
#     ollama.chat() for full control over streaming and prompt format. ---

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided "
    "document excerpts. Follow these rules:\n"
    "1. Base your answer only on the provided excerpts.\n"
    "2. Cite sources inline using [filename, p. N] format.\n"
    "3. If excerpts from different documents conflict, highlight the "
    "discrepancy and cite both sources.\n"
    "4. If no excerpts are relevant to the question, say so clearly."
)


def build_prompt(question: str, results: list[SearchResult]) -> list[dict]:
    """Build chat messages for Ollama from a question and search results."""
    context_parts = []
    for r in results:
        filename = r.metadata.get("filename", "unknown")
        page = r.metadata.get("page_number", "?")
        context_parts.append(
            f"--- Source: {filename} | Page {page} ---\n{r.text}"
        )

    context = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Document excerpts:\n\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]


async def answer(
    question: str,
    results: list[SearchResult],
    *,
    model: str = "llama3.1:8b",
) -> AsyncIterator[str]:
    """Stream answer tokens from Ollama."""
    messages = build_prompt(question, results)
    stream = ollama.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token
