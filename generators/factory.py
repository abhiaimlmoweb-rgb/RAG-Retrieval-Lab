"""
Generator factory.
"""

from __future__ import annotations

from generators.base import BaseGenerator
from generators.claude_generator import ClaudeGenerator
from generators.gemini_generator import GeminiGenerator
from generators.openai_generator import OpenAIGenerator


def get_generator(
    provider: str,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
) -> BaseGenerator:
    if provider == "gemini":
        return GeminiGenerator(api_key=api_key, model_name=model_name or "gemini-2.0-flash")
    if provider == "openai":
        return OpenAIGenerator(api_key=api_key, model_name=model_name or "gpt-4o-mini")
    if provider == "claude":
        return ClaudeGenerator(api_key=api_key, model_name=model_name or "claude-3-5-haiku-latest")
    raise ValueError(f"Unknown generator provider: {provider}. Choose: gemini, openai, claude")
