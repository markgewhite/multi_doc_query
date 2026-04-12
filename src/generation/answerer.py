import re
from collections.abc import AsyncIterator

import ollama

from src.models import SearchResult

# --- Not via LangChain: prompt construction and LLM calls use direct
#     ollama.chat() for full control over streaming and prompt format. ---

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided "
    "document sources. Follow these rules:\n"
    "1. Base your answer only on the provided sources.\n"
    "2. Cite sources inline using numeric references, e.g. [1, p. 12] or "
    "[2, Section: Methods]. Use the reference numbers shown in the source "
    "headers and the Reference List. If citing the same document on different "
    "pages, reuse its number, e.g. [1, p. 5] and [1, p. 22]. Do NOT "
    "reproduce the Reference List — it will be appended automatically.\n"
    "3. If sources from different documents conflict, highlight the "
    "discrepancy and cite both sources.\n"
    "4. If no sources are relevant to the question, say so clearly."
)


def build_ref_map(
    results: list[SearchResult],
    existing: dict[str, int] | None = None,
) -> dict[str, int]:
    """Assign a stable numeric ID to each unique document path.

    Returns a new dict mapping document path to its reference number (1-based).
    If *existing* is provided, those mappings are preserved and new documents
    get numbers starting after the current maximum.
    """
    ref_map = dict(existing) if existing else {}
    next_num = max(ref_map.values(), default=0) + 1
    for r in results:
        path = _doc_path(r.metadata)
        if path not in ref_map:
            ref_map[path] = next_num
            next_num += 1
    return ref_map


def build_reference_list(
    ref_map: dict[str, int],
    only: set[int] | None = None,
) -> str:
    """Format a numbered reference list string.

    If *only* is given, include only the reference numbers in that set.
    """
    lines = []
    for path, num in sorted(ref_map.items(), key=lambda x: x[1]):
        if only is not None and num not in only:
            continue
        lines.append(f"[{num}] {path}")
    return "\n".join(lines)


_REFS_TAIL_RE = re.compile(
    r"\n*(?:---\n)?"          # optional horizontal rule
    r"\*{0,2}"                # optional bold markers
    r"[Rr]eferences?:?"      # "Reference:", "References:", bold variants
    r"\*{0,2}"                # closing bold markers
    r"\n.*",                  # everything after
    re.DOTALL,
)


def strip_llm_references(text: str) -> str:
    """Remove any trailing reference/references section the LLM generated."""
    return _REFS_TAIL_RE.sub("", text).rstrip()


def _doc_path(metadata: dict[str, str | int]) -> str:
    """Get the document path from metadata, preferring relative_path."""
    return metadata.get("relative_path", metadata.get("filename", "unknown"))


def _source_label(metadata: dict[str, str | int], ref_num: int) -> str:
    """Build the inline source label for a context header, e.g. [1, p. 5]."""
    section = metadata.get("section_header")
    if section:
        return f"[{ref_num}, Section: {section}]"
    page = metadata.get("page_number", "?")
    return f"[{ref_num}, p. {page}]"


def _element_name(metadata: dict[str, str | int], ref_num: int) -> str:
    """Build a display name for a source element, e.g. [1] path, p. 5."""
    path = _doc_path(metadata)
    section = metadata.get("section_header")
    if section:
        return f"[{ref_num}] {path}, Section: {section}"
    page = metadata.get("page_number", "?")
    return f"[{ref_num}] {path}, p. {page}"


def build_prompt(
    question: str,
    results: list[SearchResult],
    *,
    ref_map: dict[str, int] | None = None,
) -> list[dict]:
    """Build chat messages for Ollama from a question and search results.

    If *ref_map* is provided, it is used for numbering; otherwise a fresh
    map is built from *results*.
    """
    if ref_map is None:
        ref_map = build_ref_map(results)

    context_parts = []
    for r in results:
        ref_num = ref_map[_doc_path(r.metadata)]
        label = _source_label(r.metadata, ref_num)
        context_parts.append(f"--- {label} ---\n{r.text}")

    context = "\n\n".join(context_parts)

    # Include reference list in prompt so LLM knows the mapping
    ref_list = build_reference_list(ref_map)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Document excerpts:\n\n{context}\n\n"
                f"Reference List:\n{ref_list}\n\n"
                f"Question: {question}"
            ),
        },
    ]


def build_source_elements(
    results: list[SearchResult],
    *,
    ref_map: dict[str, int] | None = None,
) -> list[dict]:
    """Build source element data for Chainlit display.

    Returns a list of dicts with keys: name, content, display.
    Order matches the input (relevance-ranked by caller).
    """
    if ref_map is None:
        ref_map = build_ref_map(results)

    return [
        {
            "name": _element_name(r.metadata, ref_map[_doc_path(r.metadata)]),
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
    ref_map: dict[str, int] | None = None,
) -> AsyncIterator[str]:
    """Stream answer tokens from Ollama."""
    messages = build_prompt(question, results, ref_map=ref_map)
    stream = ollama.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token
