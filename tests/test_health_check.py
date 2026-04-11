from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.health_check import check_models, check_ollama


def _mock_response(status_code=200, json_data=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    return response


@patch("src.health_check.httpx.AsyncClient")
async def test_check_ollama_healthy(mock_client_cls):
    mock_client = AsyncMock()
    mock_client.get.return_value = _mock_response(200)
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    result = await check_ollama()

    assert result.ok is True
    assert result.message == ""


@patch("src.health_check.httpx.AsyncClient")
async def test_check_ollama_down(mock_client_cls):
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    result = await check_ollama()

    assert result.ok is False
    assert "ollama serve" in result.message


@patch("src.health_check.httpx.AsyncClient")
async def test_check_ollama_timeout(mock_client_cls):
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.TimeoutException("Timed out")
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    result = await check_ollama()

    assert result.ok is False
    assert "timed out" in result.message.lower()


@patch("src.health_check.httpx.AsyncClient")
async def test_check_models_all_present(mock_client_cls):
    mock_client = AsyncMock()
    mock_client.get.return_value = _mock_response(
        200,
        {"models": [{"name": "llama3.1:8b"}, {"name": "mxbai-embed-large"}]},
    )
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    result = await check_models(["llama3.1:8b", "mxbai-embed-large"])

    assert result.ok is True


@patch("src.health_check.httpx.AsyncClient")
async def test_check_models_one_missing(mock_client_cls):
    mock_client = AsyncMock()
    mock_client.get.return_value = _mock_response(
        200,
        {"models": [{"name": "llama3.1:8b"}]},
    )
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    result = await check_models(["llama3.1:8b", "mxbai-embed-large"])

    assert result.ok is False
    assert "ollama pull mxbai-embed-large" in result.message


@patch("src.health_check.httpx.AsyncClient")
async def test_check_models_all_missing(mock_client_cls):
    mock_client = AsyncMock()
    mock_client.get.return_value = _mock_response(200, {"models": []})
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    result = await check_models(["llama3.1:8b", "mxbai-embed-large"])

    assert result.ok is False
    assert "ollama pull llama3.1:8b" in result.message
    assert "ollama pull mxbai-embed-large" in result.message


@patch("src.health_check.httpx.AsyncClient")
async def test_check_models_matches_without_latest_tag(mock_client_cls):
    """Requesting 'mxbai-embed-large' matches 'mxbai-embed-large:latest'."""
    mock_client = AsyncMock()
    mock_client.get.return_value = _mock_response(
        200,
        {"models": [{"name": "llama3.1:8b"}, {"name": "mxbai-embed-large:latest"}]},
    )
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    result = await check_models(["llama3.1:8b", "mxbai-embed-large"])

    assert result.ok is True
