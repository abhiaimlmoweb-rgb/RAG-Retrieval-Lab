"""Anthropic Claude answer generator."""

from __future__ import annotations

import os
import time

import anthropic

from config.settings import ANTHROPIC_API_KEY_ENV, DEFAULT_CLAUDE_MODEL
from generators.base import BaseGenerator, GenerationResult
from generators.prompts import RAG_SYSTEM_PROMPT, build_user_prompt
from retrievers.base import RetrievalResult


class ClaudeGenerator(BaseGenerator):
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_CLAUDE_MODEL,
        temperature: float = 0.2,
    ) -> None:
        key = api_key or os.getenv(ANTHROPIC_API_KEY_ENV)
        if not key:
            raise ValueError(f"Anthropic API key required. Set {ANTHROPIC_API_KEY_ENV}.")
        self._model_name = model_name
        self.temperature = temperature
        self._client = anthropic.Anthropic(api_key=key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, query: str, results: list[RetrievalResult]) -> GenerationResult:
        user_prompt = build_user_prompt(query, results)
        start = time.perf_counter()
        msg = self._client.messages.create(
            model=self._model_name,
            max_tokens=1024,
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
        )
        elapsed = (time.perf_counter() - start) * 1000
        answer = msg.content[0].text if msg.content else "(Empty response.)"
        return GenerationResult(
            query=query,
            answer=answer.strip(),
            model=self._model_name,
            latency_ms=elapsed,
            context_chunks=len(results),
        )
