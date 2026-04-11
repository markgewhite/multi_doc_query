from collections.abc import Callable

import ollama

# --- Not via LangChain: embeddings use direct ollama.embed() calls
#     for full control and to avoid unnecessary framework coupling. ---

EmbedFn = Callable[[list[str]], list[list[float]]]


def make_ollama_embed_fn(model: str = "mxbai-embed-large") -> EmbedFn:
    """Create an embedding function that calls ollama.embed()."""

    def embed(texts: list[str]) -> list[list[float]]:
        # Embed one at a time to stay within the model's context window.
        # Truncate texts that exceed the model's token limit to avoid errors.
        max_chars = 2000  # ~512 tokens, conservative estimate
        embeddings = []
        for text in texts:
            truncated = text[:max_chars] if len(text) > max_chars else text
            result = ollama.embed(model=model, input=[truncated])
            embeddings.append(result["embeddings"][0])
        return embeddings

    return embed
