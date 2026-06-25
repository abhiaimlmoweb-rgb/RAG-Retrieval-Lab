"""Cohere embedding adapter."""

from __future__ import annotations

import os

import cohere
import numpy as np

from config.settings import COHERE_API_KEY_ENV, DEFAULT_COHERE_EMBED_MODEL
from embeddings.base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = DEFAULT_COHERE_EMBED_MODEL, api_key: str | None = None) -> None:
        key = api_key or os.getenv(COHERE_API_KEY_ENV)
        if not key:
            raise ValueError(f"Cohere API key required. Set {COHERE_API_KEY_ENV}.")
        self._model_name = model_name
        self._client = cohere.ClientV2(api_key=key)
        self._dimension = 1024

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _normalize(self, vectors: list[list[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        resp = self._client.embed(
            texts=texts,
            model=self._model_name,
            input_type="search_document",
            embedding_types=["float"],
        )
        return self._normalize(resp.embeddings.float_)

    def embed_query(self, query: str) -> np.ndarray:
        resp = self._client.embed(
            texts=[query],
            model=self._model_name,
            input_type="search_query",
            embedding_types=["float"],
        )
        return self._normalize(resp.embeddings.float_)[0]
