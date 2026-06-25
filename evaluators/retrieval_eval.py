"""
Batch retrieval evaluation harness.

Runs labeled queries from a JSON dataset and aggregates nDCG, MRR, and recall@K.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from evaluators.metrics import QueryEvalResult, RelevantChunk, evaluate_query
from pipeline import RAGPipeline


@dataclass
class EvalDataset:
    """Labeled query set for retrieval evaluation."""

    items: list[dict]

    @classmethod
    def load(cls, path: Path | str) -> EvalDataset:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Eval dataset must be a JSON array of query objects")
        return cls(items=data)

    def to_relevant(self, item: dict) -> list[RelevantChunk]:
        return [
            RelevantChunk(
                document_name=rc["document_name"],
                chunk_id=int(rc["chunk_id"]),
            )
            for rc in item.get("relevant_chunks", [])
        ]


class RetrievalEvaluator:
    """Run batch eval over a pipeline and labeled dataset."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def run(self, dataset: EvalDataset, *, k: int | None = None) -> list[QueryEvalResult]:
        top_k = k or self.pipeline.config.top_k
        results: list[QueryEvalResult] = []

        for item in dataset.items:
            query = item["query"]
            relevant = dataset.to_relevant(item)
            retrieved = self.pipeline.retrieve(query, top_k=top_k)
            results.append(evaluate_query(query, retrieved, relevant, k=top_k))

        return results

    @staticmethod
    def summarize(per_query: list[QueryEvalResult]) -> dict:
        if not per_query:
            return {}
        k = per_query[0].top_k
        return {
            f"mean_recall@{k}": round(sum(r.recall_at_k for r in per_query) / len(per_query), 4),
            "mean_mrr": round(sum(r.mrr for r in per_query) / len(per_query), 4),
            f"mean_ndcg@{k}": round(sum(r.ndcg_at_k for r in per_query) / len(per_query), 4),
            "num_queries": len(per_query),
        }

    @staticmethod
    def to_dataframe(per_query: list[QueryEvalResult]) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in per_query])
