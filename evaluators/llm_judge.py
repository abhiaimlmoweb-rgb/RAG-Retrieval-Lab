"""LLM-as-judge for answer quality evaluation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

from config.settings import GEMINI_API_KEY_ENV
from generators.base import GenerationResult
from retrievers.base import RetrievalResult


@dataclass(frozen=True)
class JudgeScore:
    faithfulness: float
    relevance: float
    overall: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "overall": self.overall,
            "rationale": self.rationale,
        }


class LLMJudge:
    """Score RAG answers 1–5 on faithfulness and relevance."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash") -> None:
        key = api_key or os.getenv(GEMINI_API_KEY_ENV)
        if not key:
            raise ValueError(f"LLM judge requires {GEMINI_API_KEY_ENV}")
        self._client = genai.Client(api_key=key)
        self._model = model

    def score(
        self,
        query: str,
        results: list[RetrievalResult],
        generation: GenerationResult,
    ) -> JudgeScore:
        context = "\n\n".join(r.chunk.text[:400] for r in results[:5])
        prompt = f"""You are an evaluator for RAG systems. Score the answer 1-5 for:
- faithfulness (grounded in context only)
- relevance (answers the question)

Return JSON only: {{"faithfulness": N, "relevance": N, "overall": N, "rationale": "..."}}

Context:
{context}

Question: {query}

Answer: {generation.answer}
"""
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = resp.text or "{}"
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group() if match else text)
        return JudgeScore(
            faithfulness=float(data.get("faithfulness", 0)),
            relevance=float(data.get("relevance", 0)),
            overall=float(data.get("overall", 0)),
            rationale=str(data.get("rationale", "")),
        )
