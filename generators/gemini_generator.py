"""
Google Gemini answer generator.

Uses the Gemini API to synthesize answers from retrieved chunks.
Get a key: https://aistudio.google.com/apikey
"""

from __future__ import annotations

import os
import time

from google import genai
from google.genai import types

from config.settings import DEFAULT_GEMINI_MODEL, GEMINI_API_KEY_ENV
from generators.base import BaseGenerator, GenerationResult
from generators.prompts import RAG_SYSTEM_PROMPT, build_user_prompt
from retrievers.base import RetrievalResult


class GeminiGenerator(BaseGenerator):
    """Generate grounded answers with Google Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 0.2,
    ) -> None:
        self._api_key = api_key or os.getenv(GEMINI_API_KEY_ENV)
        if not self._api_key:
            raise ValueError(
                f"Gemini API key required. Set {GEMINI_API_KEY_ENV} in .env or pass api_key=."
            )
        self._model_name = model_name
        self.temperature = temperature
        self._client = genai.Client(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, query: str, results: list[RetrievalResult]) -> GenerationResult:
        user_prompt = build_user_prompt(query, results)
        start = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=RAG_SYSTEM_PROMPT,
                temperature=self.temperature,
            ),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        answer = response.text.strip() if response.text else "(Empty response from model.)"
        return GenerationResult(
            query=query,
            answer=answer,
            model=self._model_name,
            latency_ms=elapsed_ms,
            context_chunks=len(results),
            prompt_tokens_estimate=len(user_prompt.split()) + len(RAG_SYSTEM_PROMPT.split()),
        )
