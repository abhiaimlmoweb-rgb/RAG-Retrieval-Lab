"""
FAISS dense retriever.

Uses Facebook AI Similarity Search for approximate/exact nearest-neighbor lookup.
IndexFlatIP on L2-normalized vectors equals cosine similarity — same math as the
in-memory retriever, but scales better and mirrors production vector DB patterns.
"""

from __future__ import annotations

import time

import faiss
import numpy as np

from chunkers.base import Chunk
from config.settings import CACHE_DIR
from cache.embedding_cache import EmbeddingCache
from embeddings.base import BaseEmbedder
from retrievers.base import BaseRetriever, RetrievalResult


class FAISSRetriever(BaseRetriever):
    """Dense retrieval backed by a FAISS inner-product index."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        *,
        use_cache: bool = True,
        cache_dir=CACHE_DIR,
    ) -> None:
        self.embedder = embedder
        self.use_cache = use_cache
        self._cache = EmbeddingCache(cache_dir) if use_cache else None
        self._chunks: list[Chunk] = []
        self._index: faiss.IndexFlatIP | None = None

    @property
    def is_indexed(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._cache is not None:
            return self._cache.embed_with_cache(
                self.embedder.model_name,
                texts,
                self.embedder.embed_documents,
                dimension=self.embedder.dimension,
            )
        return self.embedder.embed_documents(texts)

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        if not chunks:
            self._index = None
            return

        texts = [c.text for c in chunks]
        embeddings = self._embed_documents(texts).astype(np.float32)

        index = faiss.IndexFlatIP(self.embedder.dimension)
        index.add(embeddings)
        self._index = index

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.is_indexed:
            raise RuntimeError("FAISS retriever is not indexed. Call index() first.")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        start = time.perf_counter()
        query_vec = self.embedder.embed_query(query).astype(np.float32).reshape(1, -1)

        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query_vec, k)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results: list[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0], strict=True), start=1):
            if idx < 0:
                continue
            chunk = self._chunks[int(idx)]
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    similarity_score=float(score),
                    rank=rank,
                    latency_ms=elapsed_ms,
                    source_document=chunk.document_name,
                    retrieval_method="faiss",
                )
            )
        return results
