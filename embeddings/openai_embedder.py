"""
OpenAI embedding adapter.

Uses text-embedding-3-small for cloud-hosted dense vectors — useful when you
want strong retrieval without running local sentence-transformers models.
"""

from __future__ import annotations

import os

import numpy as np
from openai import OpenAI

from config.settings import DEFAULT_OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY_ENV
from embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """Embedding adapter for OpenAI embedding models."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.getenv(OPENAI_API_KEY_ENV)
        if not key:
            raise ValueError(
                f"OpenAI API key required. Set {OPENAI_API_KEY_ENV} in .env."
            )
        self._model_name = model_name
        self._client = OpenAI(api_key=key)
        self._dimension = 1536 if "3-small" in model_name else 3072

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([query])[0]
