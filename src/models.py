from dataclasses import dataclass


@dataclass
class Document:
    text: str
    metadata: dict[str, str | int]


@dataclass
class Chunk:
    text: str
    metadata: dict[str, str | int]


@dataclass
class SearchResult:
    text: str
    metadata: dict[str, str | int]
    distance: float
