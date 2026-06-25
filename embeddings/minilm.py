"""
all-MiniLM-L6-v2 embedder.

A lightweight, fast baseline model. Good for prototyping and latency-sensitive
workloads. Uses the same encoding path for queries and documents (symmetric).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings.base import BaseEmbedder


class MiniLMEmbedder(BaseEmbedder):
    """Embedding adapter for sentence-transformers/all-MiniLM-L6-v2."""

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str | None = None) -> None:
        self._model = SentenceTransformer(self.MODEL_ID, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self.MODEL_ID

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vector = self._model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)
