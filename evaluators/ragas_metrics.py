"""
RAGAS-inspired generation metrics (embedding-based, no external RAGAS package).

Faithfulness, answer relevance, and context precision for learning how well
a RAG answer aligns with retrieved context and the user query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from embeddings.base import BaseEmbedder
from embeddings.minilm import MiniLMEmbedder
from generators.base import GenerationResult
from retrievers.base import RetrievalResult


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p.split()) >= 3]


@dataclass(frozen=True)
class RagasReport:
    """Lightweight RAG quality scores in [0, 1]."""

    faithfulness: float
    answer_relevance: float
    context_precision: float

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
        }


class RagasEvaluator:
    """Embedding similarity proxy for common RAG evaluation metrics."""

    def __init__(self, threshold: float = 0.35, embedder: BaseEmbedder | None = None) -> None:
        self.threshold = threshold
        self._embedder = embedder or MiniLMEmbedder()

    def _cos(self, a: str, b: str) -> float:
        va = self._embedder.embed_query(a)
        vb = self._embedder.embed_query(b)
        return float(np.dot(va, vb))

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Fraction of answer sentences supported by any context chunk."""
        sents = _sentences(answer)
        if not sents or not contexts:
            return 0.0
        supported = 0
        for sent in sents:
            if max(self._cos(sent, ctx) for ctx in contexts) >= self.threshold:
                supported += 1
        return supported / len(sents)

    def answer_relevance(self, query: str, answer: str) -> float:
        """Cosine similarity between query and answer (normalized to [0,1])."""
        sim = self._cos(query, answer)
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def context_precision(self, query: str, results: list[RetrievalResult]) -> float:
        """Fraction of top retrieved chunks relevant to the query."""
        if not results:
            return 0.0
        hits = sum(1 for r in results if self._cos(query, r.chunk.text) >= self.threshold)
        return hits / len(results)

    def evaluate(
        self,
        query: str,
        results: list[RetrievalResult],
        generation: GenerationResult,
    ) -> RagasReport:
        contexts = [r.chunk.text for r in results]
        return RagasReport(
            faithfulness=self.faithfulness(generation.answer, contexts),
            answer_relevance=self.answer_relevance(query, generation.answer),
            context_precision=self.context_precision(query, results),
        )
