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
        context_parts.append(
            f"--- {_source_name(r.metadata)} ---\n{r.text}"
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


def _source_name(metadata: dict[str, str | int]) -> str:
    """Build a display name for a source from its metadata."""
    path = metadata.get("relative_path", metadata.get("filename", "unknown"))
    page = metadata.get("page_number", "?")
    return f"Source: {path} | Page {page}"


def build_source_elements(results: list[SearchResult]) -> list[dict]:
    """Build source element data for Chainlit display.

    Returns a list of dicts with keys: name, content, display.
    Order matches the input (relevance-ranked by caller).
    """
    return [
        {
            "name": _source_name(r.metadata),
            "content": r.text,
            "display": "side",
        }
        for r in results
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
