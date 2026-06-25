"""Tests for CRAG and RAGAS helpers."""

from chunkers.base import Chunk
from generators.base import GenerationResult
from generators.crag import abstention_answer, grade_retrieval, merge_retrieval_results
from retrievers.base import RetrievalResult


def _result(query: str, chunk_id: int, score: float) -> RetrievalResult:
    chunk = Chunk(f"text {chunk_id}", "doc.md", chunk_id, 8, "fixed")
    return RetrievalResult(query, chunk, score, 1, 1.0, "doc.md", "dense")


def test_grade_retrieval_pass_fail():
    assert grade_retrieval([_result("q", 0, 0.8)], threshold=0.35).passed
    assert not grade_retrieval([_result("q", 0, 0.1)], threshold=0.35).passed


def test_merge_retrieval_results_dedupes():
    a = [_result("q", 0, 0.5), _result("q", 1, 0.9)]
    b = [_result("q", 1, 0.7)]
    merged = merge_retrieval_results("q", [a, b], top_k=2)
    assert len(merged) == 2
    assert merged[0].chunk.chunk_id == 1
    assert merged[0].similarity_score == 0.9


def test_abstention_answer():
    grade = grade_retrieval([], threshold=0.35)
    ans = abstention_answer("what?", grade)
    assert "don't have enough" in ans.answer.lower()
