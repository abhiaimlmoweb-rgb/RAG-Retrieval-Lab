"""
Cosine similarity retriever (in-memory).

Retrieval = embed the query, compare against all chunk vectors, return top-K.
Cosine similarity measures the angle between vectors; with L2-normalized
embeddings it equals the dot product and ranges from -1 to 1 (1 = identical).

Production systems replace this brute-force scan with approximate nearest
neighbor indexes (Qdrant, Pinecone, FAISS) — same math, faster at scale.
"""

from __future__ import annotations

import time

import numpy as np

from chunkers.base import Chunk
from embeddings.base import BaseEmbedder
from retrievers.base import BaseRetriever, RetrievalResult


class CosineRetriever(BaseRetriever):
    """
    Brute-force cosine similarity search over an in-memory embedding matrix.

    Vectors must be L2-normalized (our embedders do this) so cosine similarity
    reduces to a dot product: cos(q, d) = q · d.
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    @property
    def is_indexed(self) -> bool:
        return self._embeddings is not None and len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def index(self, chunks: list[Chunk]) -> None:
        """
        Embed all chunks and store vectors in memory.

        In production this step writes to a vector DB; here we keep a NumPy
        matrix for transparency and easy debugging.
        """
        self._chunks = list(chunks)
        if not chunks:
            self._embeddings = None
            return

        texts = [c.text for c in chunks]
        self._embeddings = self.embedder.embed_documents(texts)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Run top-K cosine similarity retrieval for a query.

        Returns results sorted by descending similarity score.
        """
        if not self.is_indexed:
            raise RuntimeError("Retriever is not indexed. Call index() first.")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        start = time.perf_counter()

        query_vec = self.embedder.embed_query(query)
        # Dot product on normalized vectors = cosine similarity
        scores = self._embeddings @ query_vec

        k = min(top_k, len(self._chunks))
        # argpartition is O(n); full sort only on top-k candidates
        if k == len(self._chunks):
            top_indices = np.argsort(scores)[::-1]
        else:
            partition = np.argpartition(scores, -k)[-k:]
            top_indices = partition[np.argsort(scores[partition])[::-1]]

        elapsed_ms = (time.perf_counter() - start) * 1000

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self._chunks[int(idx)]
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    similarity_score=float(scores[idx]),
                    rank=rank,
                    latency_ms=elapsed_ms,
                    source_document=chunk.document_name,
                    retrieval_method="dense",
                )
            )
        return results
