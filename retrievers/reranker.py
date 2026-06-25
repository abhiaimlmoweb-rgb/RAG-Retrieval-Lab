"""
Cross-encoder reranker.

Bi-encoders (embedding models) embed query and document separately — fast but
approximate. Cross-encoders score (query, document) pairs jointly — slower
but more accurate. A common production pattern: retrieve top-20, rerank to top-5.
"""

from __future__ import annotations

import time

from sentence_transformers import CrossEncoder

from config.settings import DEFAULT_RERANKER_MODEL
from retrievers.base import RetrievalResult


class CrossEncoderReranker:
    """Rerank retrieval hits with a cross-encoder model."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Rerank results by cross-encoder relevance scores.

        Returns a new list with updated ranks and scores.
        """
        if not results:
            return []

        k = top_k if top_k is not None else len(results)
        start = time.perf_counter()

        pairs = [(query, r.chunk.text) for r in results]
        scores = self.model.predict(pairs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        scored = sorted(
            zip(results, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:k]

        reranked: list[RetrievalResult] = []
        for rank, (result, score) in enumerate(scored, start=1):
            reranked.append(
                RetrievalResult(
                    query=result.query,
                    chunk=result.chunk,
                    similarity_score=float(score),
                    rank=rank,
                    latency_ms=elapsed_ms,
                    source_document=result.source_document,
                    retrieval_method=f"{result.retrieval_method}+rerank",
                )
            )
        return reranked
