"""
Retrieval evaluation metrics.

Implements standard IR metrics for comparing chunking strategies, embedding
models, and retrieval modes when ground-truth relevant chunks are known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from retrievers.base import RetrievalResult


@dataclass(frozen=True)
class RelevantChunk:
    """Ground-truth relevant chunk identifier."""

    document_name: str
    chunk_id: int

    def matches(self, result: RetrievalResult) -> bool:
        return (
            result.chunk.document_name == self.document_name
            and result.chunk.chunk_id == self.chunk_id
        )


def recall_at_k(results: list[RetrievalResult], relevant: list[RelevantChunk], k: int) -> float:
    """Fraction of relevant chunks found in top-K (binary per query)."""
    if not relevant:
        return 0.0
    top = results[:k]
    hits = sum(
        1 for rel in relevant if any(rel.matches(r) for r in top)
    )
    return hits / len(relevant)


def mrr(results: list[RetrievalResult], relevant: list[RelevantChunk]) -> float:
    """Mean reciprocal rank of the first relevant hit."""
    for rank, result in enumerate(results, start=1):
        if any(rel.matches(result) for rel in relevant):
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances: list[int], k: int) -> float:
    """Discounted cumulative gain."""
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg_at_k(results: list[RetrievalResult], relevant: list[RelevantChunk], k: int) -> float:
    """Normalized DCG — 1.0 when all relevant chunks rank at the top."""
    if not relevant:
        return 0.0

    relevances = [
        1 if any(rel.matches(r) for rel in relevant) else 0
        for r in results[:k]
    ]
    ideal = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal_dcg


@dataclass
class QueryEvalResult:
    """Metrics for a single labeled query."""

    query: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    top_k: int
    num_relevant: int

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            f"recall@{self.top_k}": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            f"ndcg@{self.top_k}": round(self.ndcg_at_k, 4),
            "num_relevant": self.num_relevant,
        }


def evaluate_query(
    query: str,
    results: list[RetrievalResult],
    relevant: list[RelevantChunk],
    *,
    k: int = 5,
) -> QueryEvalResult:
    """Compute all metrics for one query."""
    return QueryEvalResult(
        query=query,
        recall_at_k=recall_at_k(results, relevant, k),
        mrr=mrr(results, relevant),
        ndcg_at_k=ndcg_at_k(results, relevant, k),
        top_k=k,
        num_relevant=len(relevant),
    )
