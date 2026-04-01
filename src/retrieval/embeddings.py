from collections.abc import Callable

import ollama

# --- Not via LangChain: embeddings use direct ollama.embed() calls
#     for full control and to avoid unnecessary framework coupling. ---

EmbedFn = Callable[[list[str]], list[list[float]]]


def make_ollama_embed_fn(model: str = "mxbai-embed-large") -> EmbedFn:
    """Create an embedding function that calls ollama.embed()."""

    def embed(texts: list[str]) -> list[list[float]]:
        result = ollama.embed(model=model, input=texts)
        return result["embeddings"]

    return embed
