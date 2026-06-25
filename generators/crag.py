"""
Corrective RAG (CRAG) — grade retrieval and retry or abstain.

If initial retrieval scores are weak, re-query with multi-query expansion.
If still weak, return an abstention message instead of hallucinating.
"""

from __future__ import annotations

from dataclasses import dataclass

from generators.base import GenerationResult
from retrievers.base import RetrievalResult
from retrievers.query_expansion import expand_query_multi


@dataclass(frozen=True)
class RetrievalGrade:
    """Retrieval quality assessment."""

    score: float
    passed: bool
    reason: str

    def to_dict(self) -> dict:
        return {"score": round(self.score, 4), "passed": self.passed, "reason": self.reason}


def grade_retrieval(results: list[RetrievalResult], *, threshold: float = 0.35) -> RetrievalGrade:
    """Grade retrieval by average top-hit similarity."""
    if not results:
        return RetrievalGrade(0.0, False, "no_results")
    avg = sum(r.similarity_score for r in results) / len(results)
    passed = avg >= threshold
    return RetrievalGrade(avg, passed, "ok" if passed else "low_confidence")


def merge_retrieval_results(
    query: str,
    batches: list[list[RetrievalResult]],
    *,
    top_k: int,
) -> list[RetrievalResult]:
    """Merge multiple retrieval passes, dedupe by chunk, keep best scores."""
    merged: dict[tuple[str, int], RetrievalResult] = {}
    for batch in batches:
        for r in batch:
            key = (r.chunk.document_name, r.chunk.chunk_id)
            if key not in merged or r.similarity_score > merged[key].similarity_score:
                merged[key] = r
    ranked = sorted(merged.values(), key=lambda x: x.similarity_score, reverse=True)[:top_k]
    return [
        RetrievalResult(
            query=query,
            chunk=r.chunk,
            similarity_score=r.similarity_score,
            rank=i,
            latency_ms=r.latency_ms,
            source_document=r.source_document,
            retrieval_method=r.retrieval_method,
        )
        for i, r in enumerate(ranked, start=1)
    ]


def abstention_answer(query: str, grade: RetrievalGrade) -> GenerationResult:
    """Return a safe response when retrieval is not trustworthy."""
    return GenerationResult(
        query=query,
        answer=(
            "I don't have enough reliable context in the indexed documents to answer this "
            f"confidently (retrieval grade: {grade.score:.2f}). "
            "Try rephrasing the question or adding more source documents."
        ),
        model="crag-abstain",
        latency_ms=0.0,
        context_chunks=0,
        prompt_tokens_estimate=0,
    )
