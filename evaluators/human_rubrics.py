"""Human rubric scores for manual evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config.settings import EXPERIMENTS_DIR


class HumanRubricStore:
    """Persist manual 1–5 scores from the UI."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (EXPERIMENTS_DIR / "human_rubrics.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        query: str,
        answer: str,
        faithfulness: int,
        relevance: int,
        notes: str = "",
    ) -> dict:
        record = {
            "id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "answer": answer,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "notes": notes,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def list_recent(self, limit: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        return records[-limit:][::-1]
