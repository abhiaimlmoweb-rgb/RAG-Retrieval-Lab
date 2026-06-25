"""
Base chunker interface.

Chunking splits long documents into smaller units that fit embedding model
context limits and improve retrieval precision. Smaller chunks = finer-grained
matches; larger chunks = more surrounding context per hit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """
    A text segment ready for embedding and retrieval.

    Metadata travels with each chunk so results can be traced back to source
    documents — essential for debugging retrieval and building citations.
    """

    text: str
    document_name: str
    chunk_id: int
    chunk_size: int
    chunking_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "document_name": self.document_name,
            "chunk_id": self.chunk_id,
            "chunk_size": self.chunk_size,
            "chunking_strategy": self.chunking_strategy,
            "metadata": self.metadata,
        }


class BaseChunker(ABC):
    """Abstract chunker — all strategies share size/overlap configuration."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunking_strategy: str = "base",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy

    @abstractmethod
    def chunk_text(self, text: str, document_name: str) -> list[Chunk]:
        """Split a single document's text into chunks."""

    def chunk_documents(
        self, documents: list[tuple[str, str]]
    ) -> list[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of (document_name, text) tuples.
        """
        all_chunks: list[Chunk] = []
        for doc_name, text in documents:
            all_chunks.extend(self.chunk_text(text, doc_name))
        return all_chunks
