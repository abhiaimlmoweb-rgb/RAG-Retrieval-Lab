"""
Shared retrieval types and interfaces.

All retrievers return the same RetrievalResult shape so the pipeline, UI,
evaluators, and generators can swap dense / BM25 / hybrid without changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from chunkers.base import Chunk


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked retrieval hit."""

    query: str
    chunk: Chunk
    similarity_score: float
    rank: int
    latency_ms: float
    source_document: str
    retrieval_method: str = "dense"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "rank": self.rank,
            "similarity_score": round(self.similarity_score, 6),
            "latency_ms": round(self.latency_ms, 3),
            "source_document": self.source_document,
            "retrieval_method": self.retrieval_method,
            "document_name": self.chunk.document_name,
            "chunk_id": self.chunk.chunk_id,
            "chunk_size": self.chunk.chunk_size,
            "chunking_strategy": self.chunk.chunking_strategy,
            "chunk_text": self.chunk.text,
        }


class BaseRetriever(ABC):
    """Common interface for dense, sparse, and hybrid retrievers."""

    @property
    @abstractmethod
    def is_indexed(self) -> bool:
        ...

    @property
    @abstractmethod
    def chunk_count(self) -> int:
        ...

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        ...
