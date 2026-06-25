"""
Parent-child chunking for small-to-big retrieval.

Indexes small child chunks for precise retrieval, but stores parent windows
in metadata so generation can use broader context.
"""

from __future__ import annotations

from chunkers.base import BaseChunker, Chunk
from chunkers.recursive import RecursiveChunker


class ParentChildChunker(BaseChunker):
    """Create child chunks linked to larger parent text windows."""

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        parent_size: int = 1200,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, chunking_strategy="parent_child")
        self.parent_size = parent_size
        self._child_chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _parent_windows(self, text: str) -> list[str]:
        if len(text) <= self.parent_size:
            return [text]
        windows: list[str] = []
        step = max(1, self.parent_size - self.chunk_overlap)
        for start in range(0, len(text), step):
            windows.append(text[start : start + self.parent_size])
            if start + self.parent_size >= len(text):
                break
        return windows

    def chunk_text(self, text: str, document_name: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_id = 0
        for parent_id, parent_text in enumerate(self._parent_windows(text)):
            children = self._child_chunker.chunk_text(parent_text, document_name)
            for child in children:
                chunks.append(
                    Chunk(
                        text=child.text,
                        document_name=document_name,
                        chunk_id=chunk_id,
                        chunk_size=len(child.text),
                        chunking_strategy=self.chunking_strategy,
                        metadata={
                            "parent_text": parent_text,
                            "parent_id": parent_id,
                            "is_child": True,
                        },
                    )
                )
                chunk_id += 1
        return chunks
