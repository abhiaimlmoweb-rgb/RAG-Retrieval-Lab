"""
Parent-document retriever wrapper.

Retrieves using child chunks but expands results to parent text for generation.
"""

from __future__ import annotations

import time

from chunkers.base import Chunk
from retrievers.base import BaseRetriever, RetrievalResult


class ParentDocumentRetriever(BaseRetriever):
    """Wrap a retriever and swap child chunks for parent context windows."""

    def __init__(self, inner: BaseRetriever) -> None:
        self.inner = inner
        self._chunks: list[Chunk] = []

    @property
    def is_indexed(self) -> bool:
        return self.inner.is_indexed

    @property
    def chunk_count(self) -> int:
        return self.inner.chunk_count

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self.inner.index(chunks)

    def _expand_chunk(self, chunk: Chunk) -> Chunk:
        parent_text = chunk.metadata.get("parent_text")
        if not parent_text:
            return chunk
        return Chunk(
            text=parent_text,
            document_name=chunk.document_name,
            chunk_id=chunk.chunk_id,
            chunk_size=len(parent_text),
            chunking_strategy=chunk.chunking_strategy,
            metadata={**chunk.metadata, "expanded_from_child": True},
        )

    def _result_key(self, chunk: Chunk) -> tuple[str, int]:
        """Dedupe key: parent window when available, else unique child chunk."""
        if chunk.metadata.get("parent_text"):
            parent_id = int(chunk.metadata.get("parent_id", chunk.chunk_id))
            return chunk.document_name, parent_id
        return chunk.document_name, chunk.chunk_id

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        raw = self.inner.retrieve(query, top_k=top_k * 2)
        seen: set[tuple[str, int]] = set()
        results: list[RetrievalResult] = []
        start = time.perf_counter()

        for hit in raw:
            key = self._result_key(hit.chunk)
            if key in seen:
                continue
            seen.add(key)
            expanded = self._expand_chunk(hit.chunk)
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=expanded,
                    similarity_score=hit.similarity_score,
                    rank=len(results) + 1,
                    latency_ms=hit.latency_ms,
                    source_document=hit.source_document,
                    retrieval_method=f"parent_doc:{hit.retrieval_method}",
                )
            )
            if len(results) >= top_k:
                break

        elapsed = (time.perf_counter() - start) * 1000
        return [
            RetrievalResult(
                query=r.query,
                chunk=r.chunk,
                similarity_score=r.similarity_score,
                rank=r.rank,
                latency_ms=elapsed,
                source_document=r.source_document,
                retrieval_method=r.retrieval_method,
            )
            for r in results
        ]
