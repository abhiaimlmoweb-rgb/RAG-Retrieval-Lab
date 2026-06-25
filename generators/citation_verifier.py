"""Verify generated answers against retrieved context."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from embeddings.minilm import MiniLMEmbedder
from generators.base import GenerationResult
from retrievers.base import RetrievalResult


@dataclass(frozen=True)
class CitationReport:
    """Citation verification summary."""

    supported_ratio: float
    flagged_sentences: list[str]
    total_sentences: int

    def to_dict(self) -> dict:
        return {
            "supported_ratio": round(self.supported_ratio, 4),
            "flagged_sentences": self.flagged_sentences,
            "total_sentences": self.total_sentences,
        }


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p.split()) >= 4]


class CitationVerifier:
    """Check each answer sentence against retrieved chunk embeddings."""

    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold
        self._embedder = MiniLMEmbedder()

    def verify(
        self,
        generation: GenerationResult,
        results: list[RetrievalResult],
    ) -> CitationReport:
        sents = _sentences(generation.answer)
        if not sents or not results:
            return CitationReport(0.0, sents, len(sents))

        context_texts = [r.chunk.text for r in results]
        ctx_vecs = self._embedder.embed_documents(context_texts)
        sent_vecs = self._embedder.embed_documents(sents)

        supported = 0
        flagged: list[str] = []
        for i, svec in enumerate(sent_vecs):
            sims = ctx_vecs @ svec
            if float(np.max(sims)) >= self.threshold:
                supported += 1
            else:
                flagged.append(sents[i])

        ratio = supported / len(sents) if sents else 0.0
        return CitationReport(ratio, flagged, len(sents))
