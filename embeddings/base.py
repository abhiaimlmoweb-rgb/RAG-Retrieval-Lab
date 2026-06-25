"""
Embedding model interface.

Embeddings map text into dense vectors where semantic similarity corresponds
to geometric proximity (cosine similarity). Swappable embedders let you A/B
test models without changing retrieval logic — a core production pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Common interface for all embedding backends."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable / Hugging Face model identifier."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector size."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of document/chunk texts.

        Returns:
            Array of shape (n_texts, dimension).
        """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query.

        Some models (e.g. BGE) use different prefixes for queries vs documents.
        Keeping a separate method makes that distinction explicit.
        """

    def embed_single(self, text: str) -> np.ndarray:
        """Convenience wrapper for one-off document embedding."""
        return self.embed_documents([text])[0]
