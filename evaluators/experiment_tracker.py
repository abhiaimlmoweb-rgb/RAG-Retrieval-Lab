"""
Experiment tracking.

Persists each retrieval / RAG run as JSON so you can compare chunking strategies,
embedding models, retrieval modes, and generation quality over time.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from generators.base import GenerationResult
from retrievers.base import RetrievalResult


class ExperimentTracker:
    """Save and list retrieval experiment artifacts."""

    def __init__(self, experiments_dir: Path | str) -> None:
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        chunking_strategy: str,
        embedding_model: str,
        retrieval_mode: str,
        use_reranker: bool,
        top_k: int,
        query: str,
        results: list[RetrievalResult],
        generation: GenerationResult | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """
        Persist one experiment run to JSON.

        Returns:
            Path to the saved file.
        """
        payload: dict[str, Any] = {
            "experiment_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunking_strategy": chunking_strategy,
            "embedding_model": embedding_model,
            "retrieval_mode": retrieval_mode,
            "use_reranker": use_reranker,
            "top_k": top_k,
            "query": query,
            "retrieved_chunks": [r.to_dict() for r in results],
            "scores": [round(r.similarity_score, 6) for r in results],
            "latency_ms": results[0].latency_ms if results else 0.0,
        }
        if generation is not None:
            payload["generation"] = generation.to_dict()
        if extra:
            payload["metadata"] = extra

        filename = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.json"
        path = self.experiments_dir / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def list_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        """Load recent experiment summaries (newest first)."""
        files = sorted(
            self.experiments_dir.glob("exp_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]

        summaries: list[dict[str, Any]] = []
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "file": path.name,
                    "timestamp": data.get("timestamp"),
                    "retrieval_mode": data.get("retrieval_mode", "dense"),
                    "chunking_strategy": data.get("chunking_strategy"),
                    "embedding_model": data.get("embedding_model"),
                    "top_k": data.get("top_k"),
                    "query": data.get("query"),
                    "num_results": len(data.get("retrieved_chunks", [])),
                    "top_score": data.get("scores", [None])[0],
                    "has_answer": "generation" in data,
                }
            )
        return summaries

    def load(self, filename: str) -> dict[str, Any]:
        """Load a single experiment by filename."""
        path = self.experiments_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Experiment not found: {filename}")
        return json.loads(path.read_text(encoding="utf-8"))
