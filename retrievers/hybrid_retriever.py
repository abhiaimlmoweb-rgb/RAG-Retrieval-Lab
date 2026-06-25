"""
Hybrid retriever: dense cosine + BM25 with Reciprocal Rank Fusion (RRF).

RRF merges ranked lists without calibrating incompatible score scales — a
robust default in production search (Elasticsearch, Vespa, many RAG stacks).
"""

from __future__ import annotations

import time

from chunkers.base import Chunk
from embeddings.base import BaseEmbedder
from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.bm25_retriever import BM25Retriever
from retrievers.cosine_retriever import CosineRetriever

# RRF constant from the original paper; 60 is a common default
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    *,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Fuse multiple ranked chunk-index lists into one RRF score ordering.

    Returns:
        List of (chunk_index, rrf_score) sorted descending by score.
    """
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, chunk_idx in enumerate(ranking):
            scores[chunk_idx] = scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever(BaseRetriever):
    """Combines dense and BM25 retrieval via RRF."""

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.dense = CosineRetriever(embedder)
        self.sparse = BM25Retriever()

    @property
    def is_indexed(self) -> bool:
        return self.dense.is_indexed and self.sparse.is_indexed

    @property
    def chunk_count(self) -> int:
        return self.dense.chunk_count

    def index(self, chunks: list[Chunk]) -> None:
        self.dense.index(chunks)
        self.sparse.index(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.is_indexed:
            raise RuntimeError("Hybrid retriever is not indexed. Call index() first.")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        start = time.perf_counter()

        # Fetch a wider candidate pool from each retriever before fusion
        pool_k = min(max(top_k * 3, top_k), self.chunk_count)
        dense_results = self.dense.retrieve(query, top_k=pool_k)
        sparse_results = self.sparse.retrieve(query, top_k=pool_k)

        dense_ranking = [self._chunk_index(r.chunk) for r in dense_results]
        sparse_ranking = [self._chunk_index(r.chunk) for r in sparse_results]

        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
        elapsed_ms = (time.perf_counter() - start) * 1000

        results: list[RetrievalResult] = []
        for rank, (idx, rrf_score) in enumerate(fused[:top_k], start=1):
            chunk = self.dense.chunks[idx]
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
        for i, c in enumerate(self.dense.chunks):
            if c.document_name == chunk.document_name and c.chunk_id == chunk.chunk_id:
                return i
        raise ValueError("Chunk not found in index")
