"""
Recursive character chunking.

Attempts to split on natural separators (paragraphs, sentences, words) before
falling back to fixed windows. This keeps semantic units intact and typically
improves retrieval quality compared to blind character splits.
"""

from __future__ import annotations

from chunkers.base import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """
    Hierarchical splitter that prefers paragraph → sentence → word boundaries.

    Mirrors the approach used by LangChain's RecursiveCharacterTextSplitter,
    implemented here for learning without external chunking dependencies.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy="recursive",
        )
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk_text(self, text: str, document_name: str) -> list[Chunk]:
        if not text.strip():
            return []

        raw_segments = self._split_recursively(text, self.separators)
        merged = self._merge_segments(raw_segments)

        chunks: list[Chunk] = []
        for idx, segment in enumerate(merged):
            segment = segment.strip()
            if not segment:
                continue
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

    def _split_recursively(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first applicable separator."""
        if len(text) <= self.chunk_size:
            return [text] if text else []

        separator = separators[0] if separators else ""
        next_separators = separators[1:] if len(separators) > 1 else [""]

        if separator == "":
            # Last resort: hard character split
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        parts = text.split(separator)
        segments: list[str] = []
        current = ""

        for i, part in enumerate(parts):
            piece = part if i == len(parts) - 1 else part + separator
            candidate = current + piece

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    segments.append(current)
                if len(piece) > self.chunk_size:
                    segments.extend(
                        self._split_recursively(piece, next_separators)
                    )
                    current = ""
                else:
                    current = piece

        if current.strip():
            segments.append(current)

        return segments

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """
        Merge small segments and apply overlap between consecutive chunks.

        Overlap helps queries that sit on a boundary still retrieve useful
        context from the adjacent chunk.
        """
        if not segments:
            return []

        merged: list[str] = []
        current = ""

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            candidate = f"{current} {segment}".strip() if current else segment
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = segment

        if current:
            merged.append(current)

        if self.chunk_overlap <= 0 or len(merged) <= 1:
            return merged

        overlapped: list[str] = [merged[0]]
        for i in range(1, len(merged)):
            prev_tail = merged[i - 1][-self.chunk_overlap :]
            overlapped.append(f"{prev_tail} {merged[i]}".strip())

        return overlapped
