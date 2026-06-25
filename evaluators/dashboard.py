"""
Evaluation dashboard.

Surfaces retrieval outcomes in a structured table for manual inspection —
the fastest way to spot bad chunking, wrong embeddings, or missing content
before investing in automated metrics (nDCG, MRR, recall@K).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from retrievers.base import RetrievalResult


@dataclass
class EvaluationDashboard:
    """Format retrieval results for display and export."""

    def to_dataframe(self, results: list[RetrievalResult]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame(
                columns=[
                    "Query",
                    "Rank",
                    "Similarity Score",
                    "Source Document",
                    "Chunk ID",
                    "Chunk Size",
                    "Retrieval Latency (ms)",
                    "Retrieval Method",
                    "Retrieved Chunk",
                ]
            )

        rows = [
            {
                "Query": r.query,
                "Rank": r.rank,
                "Similarity Score": round(r.similarity_score, 4),
                "Source Document": r.source_document,
                "Chunk ID": r.chunk.chunk_id,
                "Chunk Size": r.chunk.chunk_size,
                "Retrieval Latency (ms)": round(r.latency_ms, 2),
                "Retrieval Method": r.retrieval_method,
                "Retrieved Chunk": r.chunk.text,
            }
            for r in results
        ]
        return pd.DataFrame(rows)

    def summary(self, results: list[RetrievalResult]) -> dict:
        """High-level stats for a single query run."""
        if not results:
            return {
                "result_count": 0,
                "top_score": None,
                "avg_score": None,
                "latency_ms": None,
            }

        scores = [r.similarity_score for r in results]
        return {
            "result_count": len(results),
            "top_score": round(max(scores), 4),
            "avg_score": round(sum(scores) / len(scores), 4),
            "latency_ms": round(results[0].latency_ms, 2),
        }
