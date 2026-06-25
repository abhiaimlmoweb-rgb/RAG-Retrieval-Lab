"""Query expansion: multi-query and HyDE-style hypothetical document."""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from config.settings import GEMINI_API_KEY_ENV


def expand_query_multi(query: str, n: int = 3) -> list[str]:
    """Generate paraphrased query variants (rule-based fallback)."""
    variants = [query]
    templates = [
        f"Explain: {query}",
        f"What do documents say about {query.rstrip('?')}?",
        f"Key facts regarding {query.rstrip('?')}",
    ]
    for t in templates[: max(0, n - 1)]:
        if t not in variants:
            variants.append(t)
    return variants[:n]


def expand_query_hyde(query: str, api_key: str | None = None) -> str:
    """
    HyDE: generate a hypothetical answer passage, then embed/search with it.
    Uses Gemini when API key is available; otherwise returns the original query.
    """
    key = api_key or os.getenv(GEMINI_API_KEY_ENV)
    if not key:
        return query

    client = genai.Client(api_key=key)
    prompt = (
        "Write a short factual paragraph that would answer this question. "
        "Do not say you don't know — write as if you had the document.\n\n"
        f"Question: {query}"
    )
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3),
    )
    return resp.text.strip() if resp.text else query
