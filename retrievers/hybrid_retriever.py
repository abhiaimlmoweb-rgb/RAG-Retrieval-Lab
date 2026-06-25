"""
Hybrid retriever: dense + BM25 with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import time

from chunkers.base import Chunk
from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.bm25_retriever import BM25Retriever

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    *,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, chunk_idx in enumerate(ranking):
            scores[chunk_idx] = scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever(BaseRetriever):
    """Combines any dense retriever with BM25 via RRF."""

    def __init__(self, dense: BaseRetriever) -> None:
        self.dense = dense
        self.sparse = BM25Retriever()
        self._chunks: list[Chunk] = []

    @property
    def is_indexed(self) -> bool:
        return self.dense.is_indexed and self.sparse.is_indexed

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self.dense.index(chunks)
        self.sparse.index(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.is_indexed:
            raise RuntimeError("Hybrid retriever is not indexed. Call index() first.")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        start = time.perf_counter()
        pool_k = min(max(top_k * 3, top_k), self.chunk_count)
        dense_results = self.dense.retrieve(query, top_k=pool_k)
        sparse_results = self.sparse.retrieve(query, top_k=pool_k)

        dense_ranking = [self._chunk_index(r.chunk) for r in dense_results]
        sparse_ranking = [self._chunk_index(r.chunk) for r in sparse_results]
        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
        elapsed_ms = (time.perf_counter() - start) * 1000

        results: list[RetrievalResult] = []
        for rank, (idx, rrf_score) in enumerate(fused[:top_k], start=1):
            chunk = self._chunks[idx]
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    similarity_score=rrf_score,
                    rank=rank,
                    latency_ms=elapsed_ms,
                    source_document=chunk.document_name,
                    retrieval_method="hybrid_rrf",
                )
            )
        return results

    def _chunk_index(self, chunk: Chunk) -> int:
        for i, c in enumerate(self._chunks):
            if c.document_name == chunk.document_name and c.chunk_id == chunk.chunk_id:
                return i
        raise ValueError("Chunk not found in index")
