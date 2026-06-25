"""
LLM generator interface.

Generation is the final stage of full RAG: retrieved chunks become context
for an LLM to produce a grounded answer. Keeping this behind an interface
lets you swap Gemini, OpenAI, Claude, or local models without touching retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from retrievers.base import RetrievalResult


@dataclass(frozen=True)
class GenerationResult:
    """LLM answer grounded in retrieved context."""

    query: str
    answer: str
    model: str
    latency_ms: float
    context_chunks: int
    prompt_tokens_estimate: int | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "context_chunks": self.context_chunks,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
        }


class BaseGenerator(ABC):
    """Common interface for answer generation backends."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def generate(self, query: str, results: list[RetrievalResult]) -> GenerationResult:
        ...
