"""
Weighted hybrid retriever: α·dense + (1-α)·BM25 score fusion.

Unlike RRF (rank-based), this blends normalized similarity scores directly —
useful for learning how the fusion weight α affects retrieval quality.
"""

from __future__ import annotations

import time

import numpy as np

from chunkers.base import Chunk
from embeddings.base import BaseEmbedder
from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.bm25_retriever import BM25Retriever, _tokenize
from retrievers.cosine_retriever import CosineRetriever


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


class WeightedHybridRetriever(BaseRetriever):
    """Fuse dense cosine and BM25 with a tunable α weight."""

    def __init__(self, embedder: BaseEmbedder, *, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.dense = CosineRetriever(embedder)
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
            raise RuntimeError("Weighted hybrid retriever not indexed")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        start = time.perf_counter()

        query_vec = self.dense.embedder.embed_query(query)
        dense_scores = self.dense._embeddings @ query_vec  # type: ignore[operator]

        query_tokens = _tokenize(query)
        bm25_scores = np.array(self.sparse._bm25.get_scores(query_tokens), dtype=np.float64)  # type: ignore[union-attr]

        fused = self.alpha * _min_max_normalize(dense_scores) + (1.0 - self.alpha) * _min_max_normalize(
            bm25_scores
        )
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(fused)[::-1][:k]
        elapsed_ms = (time.perf_counter() - start) * 1000

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self._chunks[int(idx)]
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    similarity_score=float(fused[idx]),
                    rank=rank,
                    latency_ms=elapsed_ms,
                    source_document=chunk.document_name,
                    retrieval_method=f"weighted_hybrid_a{self.alpha:.2f}",
                )
            )
        return results
