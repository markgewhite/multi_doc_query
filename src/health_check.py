from dataclasses import dataclass

import httpx


@dataclass
class HealthResult:
    ok: bool
    message: str = ""


async def check_ollama(base_url: str = "http://localhost:11434") -> HealthResult:
    """Check that Ollama is running and reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/")
        if response.status_code == 200:
            return HealthResult(ok=True)
        return HealthResult(
            ok=False,
            message=f"Ollama returned unexpected status {response.status_code}.",
        )
    except httpx.ConnectError:
        return HealthResult(
            ok=False,
            message="Ollama is not running. Please start it with `ollama serve`.",
        )
    except httpx.TimeoutException:
        return HealthResult(
            ok=False,
            message="Ollama is not responding (timed out). Please check it is running with `ollama serve`.",
        )


async def check_models(
    required: list[str], base_url: str = "http://localhost:11434"
) -> HealthResult:
    """Check that all required models are available in Ollama."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{base_url}/api/tags")

    available = set()
    for model in response.json().get("models", []):
        name = model["name"]
        available.add(name)
        # Also register without :latest so "mxbai-embed-large" matches
        # "mxbai-embed-large:latest"
        if name.endswith(":latest"):
            available.add(name.removesuffix(":latest"))

    missing = [name for name in required if name not in available]
    if missing:
        instructions = "\n".join(
            f"  - Run `ollama pull {name}` to download it" for name in missing
        )
        return HealthResult(
            ok=False,
            message=f"Missing required model(s):\n{instructions}",
        )

    return HealthResult(ok=True)
