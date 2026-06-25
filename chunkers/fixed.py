"""
Fixed-size chunking.

Splits text into windows of a fixed character length with optional overlap.
Overlap preserves context at chunk boundaries — without it, sentences split
across two chunks may fail to match queries that span the boundary.
"""

from __future__ import annotations

from chunkers.base import BaseChunker, Chunk


class FixedChunker(BaseChunker):
    """Character-based fixed window chunker with sliding overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy="fixed",
        )

    def chunk_text(self, text: str, document_name: str) -> list[Chunk]:
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_id = 0
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            segment = text[start:end].strip()
            if segment:
                chunks.append(
                    Chunk(
                        text=segment,
                        document_name=document_name,
                        chunk_id=chunk_id,
                        chunk_size=len(segment),
                        chunking_strategy=self.chunking_strategy,
                        metadata={"start_char": start, "end_char": end},
                    )
                )
                chunk_id += 1
            if end >= text_len:
                break
            start += step

        return chunks
