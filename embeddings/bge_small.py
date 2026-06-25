"""
BAAI/bge-small-en-v1.5 embedder.

BGE (BAAI General Embedding) models are strong open-source retrievers.
For retrieval, BGE expects a "Represent this sentence for searching relevant
passages: " prefix on queries — not on document chunks.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings.base import BaseEmbedder

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEEmbedder(BaseEmbedder):
    """Embedding adapter for BAAI/bge-small-en-v1.5."""

    MODEL_ID = "BAAI/bge-small-en-v1.5"

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
        prefixed = f"{BGE_QUERY_PREFIX}{query}"
        vector = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)
