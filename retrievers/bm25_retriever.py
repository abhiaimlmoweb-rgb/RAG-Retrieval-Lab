"""
BM25 sparse lexical retriever.

BM25 scores term overlap between query and documents — complementary to dense
embeddings. Lexical search excels at exact keywords, IDs, and rare terms;
dense search excels at paraphrases. Hybrid pipelines combine both.
"""

from __future__ import annotations

import re
import time

from rank_bm25 import BM25Okapi

from chunkers.base import Chunk
from retrievers.base import BaseRetriever, RetrievalResult


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever(BaseRetriever):
    """In-memory BM25 index over chunk texts."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []

    @property
    def is_indexed(self) -> bool:
        return self._bm25 is not None and len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        if not chunks:
            self._bm25 = None
            self._corpus_tokens = []
            return

        self._corpus_tokens = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.is_indexed:
            raise RuntimeError("BM25 retriever is not indexed. Call index() first.")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        start = time.perf_counter()
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        k = min(top_k, len(self._chunks))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        elapsed_ms = (time.perf_counter() - start) * 1000

        results: list[RetrievalResult] = []
        max_score = max((scores[i] for i in top_indices), default=1.0) or 1.0

        for rank, idx in enumerate(top_indices, start=1):
            chunk = self._chunks[idx]
            # Normalize BM25 scores to ~0–1 for display alongside cosine scores
            normalized = float(scores[idx]) / max_score if max_score > 0 else 0.0
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    similarity_score=normalized,
                    rank=rank,
                    latency_ms=elapsed_ms,
                    source_document=chunk.document_name,
                    retrieval_method="bm25",
                )
            )
        return results
