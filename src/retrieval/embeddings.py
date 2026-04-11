import logging
from collections.abc import Callable

import ollama

# --- Not via LangChain: embeddings use direct ollama.embed() calls
#     for full control and to avoid unnecessary framework coupling. ---

EmbedFn = Callable[[list[str]], list[list[float]]]

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
TRIM_FACTOR = 0.8  # Cut 20% on each retry


def make_ollama_embed_fn(model: str = "mxbai-embed-large") -> EmbedFn:
    """Create an embedding function that calls ollama.embed()."""

    def embed(texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(_embed_with_retry(model, text))
        return embeddings

    return embed


def _embed_with_retry(model: str, text: str) -> list[float]:
    """Embed a single text, progressively trimming on context length errors."""
    current = text
    for attempt in range(MAX_RETRIES):
        try:
            result = ollama.embed(model=model, input=[current])
            return result["embeddings"][0]
        except ollama.ResponseError as e:
            if "context length" not in str(e):
                raise
            new_len = int(len(current) * TRIM_FACTOR)
            logger.warning(
                "Embedding too long (%d chars), trimming to %d (attempt %d/%d)",
                len(current), new_len, attempt + 1, MAX_RETRIES,
            )
            current = current[:new_len]

    # Final attempt after max retries
    result = ollama.embed(model=model, input=[current])
    return result["embeddings"][0]
