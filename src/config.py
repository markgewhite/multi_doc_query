from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError, field_validator


class ConfigError(Exception):
    """Raised for any configuration loading or validation error."""


class ModelsConfig(BaseModel):
    llm: str = "llama3.1:8b"
    embedding: str = "mxbai-embed-large"


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 100


class RetrievalConfig(BaseModel):
    bm25_top_k: int = 20
    semantic_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 10
    reranker_model: str = "bge-reranker-v2-m3"


class PathsConfig(BaseModel):
    chroma_db: Path = Path("~/.multi_doc_query/chroma_db/")
    documents: Path = Path("")

    @field_validator("chroma_db", "documents", mode="before")
    @classmethod
    def expand_home(cls, v: str | Path) -> Path:
        return Path(v).expanduser()


class ScanningConfig(BaseModel):
    recursive: bool = True


class AppConfig(BaseModel):
    models: ModelsConfig = ModelsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    paths: PathsConfig = PathsConfig()
    scanning: ScanningConfig = ScanningConfig()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file."""
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(
            f"Config file not found: {config_path}. "
            "Create a config.yaml in the project root."
        )

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {config_path}: {e}") from e

    if data is None:
        data = {}

    try:
        return AppConfig(**data)
    except ValidationError as e:
        raise ConfigError(f"Config validation error: {e}") from e
