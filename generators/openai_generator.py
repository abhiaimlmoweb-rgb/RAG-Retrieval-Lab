"""
OpenAI answer generator.

Uses the Chat Completions API for grounded RAG answers from retrieved chunks.
"""

from __future__ import annotations

import os
import time

from openai import OpenAI

from config.settings import DEFAULT_OPENAI_CHAT_MODEL, OPENAI_API_KEY_ENV
from generators.base import BaseGenerator, GenerationResult
from generators.prompts import RAG_SYSTEM_PROMPT, build_user_prompt
from retrievers.base import RetrievalResult


class OpenAIGenerator(BaseGenerator):
    """Generate grounded answers with OpenAI chat models."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_OPENAI_CHAT_MODEL,
        temperature: float = 0.2,
    ) -> None:
        key = api_key or os.getenv(OPENAI_API_KEY_ENV)
        if not key:
            raise ValueError(f"OpenAI API key required. Set {OPENAI_API_KEY_ENV} in .env.")
        self._model_name = model_name
        self.temperature = temperature
        self._client = OpenAI(api_key=key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, query: str, results: list[RetrievalResult]) -> GenerationResult:
        user_prompt = build_user_prompt(query, results)
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        answer = response.choices[0].message.content or "(Empty response.)"
        return GenerationResult(
            query=query,
            answer=answer.strip(),
            model=self._model_name,
            latency_ms=elapsed_ms,
            context_chunks=len(results),
            prompt_tokens_estimate=response.usage.total_tokens if response.usage else None,
        )
