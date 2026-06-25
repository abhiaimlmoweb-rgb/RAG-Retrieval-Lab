"""
Document-structure-aware chunking.

Splits on markdown headings, numbered sections, and markdown tables before
applying size limits — keeps logical document units intact for retrieval.
"""

from __future__ import annotations

import re

from chunkers.base import BaseChunker, Chunk
from chunkers.recursive import RecursiveChunker

MARKDOWN_HEADER = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
NUMBERED_HEADER = re.compile(r"^(\d+(?:\.\d+)*\.)\s+(.+)$", re.MULTILINE)
TABLE_BLOCK = re.compile(r"(?:^\|.+\|\s*\n)+", re.MULTILINE)


def _split_tables(text: str) -> list[tuple[str, str]]:
    """Return alternating (kind, content) where kind is 'text' or 'table'."""
    parts: list[tuple[str, str]] = []
    last = 0
    for match in TABLE_BLOCK.finditer(text):
        if match.start() > last:
            parts.append(("text", text[last : match.start()]))
        parts.append(("table", match.group(0).strip()))
        last = match.end()
    if last < len(text):
        parts.append(("text", text[last:]))
    return parts or [("text", text)]


def _find_section_breaks(text: str) -> list[tuple[int, int, str, str]]:
    """
    Find section boundaries.

    Returns list of (start, level, marker, title) sorted by position.
    level: 1-6 for markdown, 7 for numbered sections.
    """
    breaks: list[tuple[int, int, str, str]] = []
    for match in MARKDOWN_HEADER.finditer(text):
        breaks.append((match.start(), len(match.group(1)), match.group(1), match.group(2).strip()))
    for match in NUMBERED_HEADER.finditer(text):
        breaks.append((match.start(), 7, match.group(1), match.group(2).strip()))
    breaks.sort(key=lambda x: x[0])
    return breaks


def _section_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


class DocumentBasedChunker(BaseChunker):
    """Chunk by document structure (headings, tables), then by size within sections."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        super().__init__(chunk_size, chunk_overlap, chunking_strategy="document_based")
        self._sub_chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _sections_from_text(self, text: str) -> list[tuple[str, str, str]]:
        """Return (section_path, section_title, body) tuples."""
        breaks = _find_section_breaks(text)
        if not breaks:
            return [("Document", "Document", text.strip())]

        sections: list[tuple[str, str, str]] = []
        stack: list[tuple[int, str]] = []

        for i, (start, level, _marker, title) in enumerate(breaks):
            end = breaks[i + 1][0] if i + 1 < len(breaks) else len(text)
            header_end = text.find("\n", start)
            body_start = header_end + 1 if header_end != -1 else start + len(title) + 1
            body = text[body_start:end].strip()

            while stack and stack[-1][0] > level:
                stack.pop()
            stack.append((level, title))
            path = _section_path(stack)
            sections.append((path, title, body))

        return sections

    def _chunk_section(
        self,
        document_name: str,
        section_path: str,
        section_title: str,
        body: str,
        chunk_id_start: int,
        *,
        chunk_type: str = "section",
    ) -> tuple[list[Chunk], int]:
        if not body.strip():
            return [], chunk_id_start

        prefix = f"[{section_path}]\n" if section_path != "Document" else ""
        full_text = f"{prefix}{body.strip()}"

        if len(full_text) <= self.chunk_size:
            return [
                Chunk(
                    text=full_text,
                    document_name=document_name,
                    chunk_id=chunk_id_start,
                    chunk_size=len(full_text),
                    chunking_strategy=self.chunking_strategy,
                    metadata={
                        "section_title": section_title,
                        "section_path": section_path,
                        "chunk_type": chunk_type,
                    },
                )
            ], chunk_id_start + 1

        sub_chunks = self._sub_chunker.chunk_text(full_text, document_name)
        out: list[Chunk] = []
        cid = chunk_id_start
        for sub in sub_chunks:
            out.append(
                Chunk(
                    text=sub.text,
                    document_name=document_name,
                    chunk_id=cid,
                    chunk_size=sub.chunk_size,
                    chunking_strategy=self.chunking_strategy,
                    metadata={
                        "section_title": section_title,
                        "section_path": section_path,
                        "chunk_type": chunk_type,
                        "sub_chunk": True,
                    },
                )
            )
            cid += 1
        return out, cid

    def chunk_text(self, text: str, document_name: str) -> list[Chunk]:
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        chunk_id = 0

        for kind, block in _split_tables(text):
            if kind == "table":
                table_chunks, chunk_id = self._chunk_section(
                    document_name,
                    "Tables",
                    "Table",
                    block,
                    0,
                    chunk_type="table",
                )
                chunks.extend(table_chunks)
                continue

            for section_path, section_title, body in self._sections_from_text(block):
                section_chunks, chunk_id = self._chunk_section(
                    document_name,
                    section_path,
                    section_title,
                    body,
                    chunk_id,
                )
                chunks.extend(section_chunks)

        return chunks
