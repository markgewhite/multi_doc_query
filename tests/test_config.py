from pathlib import Path

import pytest

from src.config import AppConfig, ConfigError, load_config


def test_load_valid_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
models:
  llm: "test-model:7b"
  embedding: "test-embed"
chunking:
  chunk_size: 256
  chunk_overlap: 50
retrieval:
  bm25_top_k: 10
  semantic_top_k: 15
  rrf_k: 30
  rerank_top_k: 5
  reranker_model: "test-reranker"
paths:
  chroma_db: "/tmp/test_chroma"
  documents: "/tmp/docs"
scanning:
  recursive: false
"""
    )
    config = load_config(config_file)

    assert config.models.llm == "test-model:7b"
    assert config.models.embedding == "test-embed"
    assert config.chunking.chunk_size == 256
    assert config.chunking.chunk_overlap == 50
    assert config.retrieval.bm25_top_k == 10
    assert config.retrieval.reranker_model == "test-reranker"
    assert config.paths.chroma_db == Path("/tmp/test_chroma")
    assert config.paths.documents == Path("/tmp/docs")
    assert config.scanning.recursive is False


def test_load_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("models:\n  llm: custom-llm\n")
    config = load_config(config_file)

    assert config.models.llm == "custom-llm"
    assert config.models.embedding == "mxbai-embed-large"
    assert config.chunking.chunk_size == 512
    assert config.retrieval.rrf_k == 60
    assert config.scanning.recursive is True


def test_load_empty_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    config = load_config(config_file)

    assert isinstance(config, AppConfig)
    assert config.models.llm == "llama3.1:8b"
    assert config.chunking.chunk_size == 512


def test_missing_file():
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config("nonexistent.yaml")


def test_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: [unclosed")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(config_file)


def test_invalid_field_type(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("chunking:\n  chunk_size: not_a_number\n")
    with pytest.raises(ConfigError, match="Config validation error"):
        load_config(config_file)


def test_path_expansion(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("paths:\n  chroma_db: ~/test_path\n")
    config = load_config(config_file)

    assert "~" not in str(config.paths.chroma_db)
    assert config.paths.chroma_db == Path.home() / "test_path"
