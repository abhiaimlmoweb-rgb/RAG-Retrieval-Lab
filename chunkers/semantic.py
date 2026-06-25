"""Semantic chunking by embedding similarity breakpoints."""

from __future__ import annotations

import re

import numpy as np

from chunkers.base import BaseChunker, Chunk
from embeddings.minilm import MiniLMEmbedder


class SemanticChunker(BaseChunker):
    """
    Split text into sentences, then merge until embedding similarity
    between consecutive sentences drops below a threshold.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        similarity_threshold: float = 0.72,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy="semantic",
        )
        self.similarity_threshold = similarity_threshold
        self._embedder = MiniLMEmbedder()

    def _sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [p.strip() for p in parts if p.strip()]

    def chunk_text(self, text: str, document_name: str) -> list[Chunk]:
        sentences = self._sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [
                Chunk(
                    text=sentences[0][: self.chunk_size],
                    document_name=document_name,
                    chunk_id=0,
                    chunk_size=len(sentences[0]),
                    chunking_strategy=self.chunking_strategy,
                )
            ]

        embeddings = self._embedder.embed_documents(sentences)
        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            candidate = " ".join(groups[-1] + [sentences[i]])
            if sim >= self.similarity_threshold and len(candidate) <= self.chunk_size:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        chunks: list[Chunk] = []
        for idx, group in enumerate(groups):
            segment = " ".join(group).strip()
            if not segment:
                continue
            if len(segment) > self.chunk_size:
                segment = segment[: self.chunk_size]
            chunks.append(
                Chunk(
                    text=segment,
                    document_name=document_name,
                    chunk_id=idx,
                    chunk_size=len(segment),
                    chunking_strategy=self.chunking_strategy,
                )
            )
        return chunks
