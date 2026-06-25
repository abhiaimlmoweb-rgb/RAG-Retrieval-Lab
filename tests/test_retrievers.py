"""Tests for hybrid retrieval."""

import numpy as np

from chunkers.base import Chunk
from embeddings.base import BaseEmbedder
from retrievers.cosine_retriever import CosineRetriever
from retrievers.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from retrievers.weighted_hybrid_retriever import WeightedHybridRetriever


class _FakeEmbedder(BaseEmbedder):
    """Deterministic tiny embedder for offline tests."""

    def __init__(self) -> None:
        self._dim = 4

    @property
    def model_name(self) -> str:
        return "fake-test"

    @property
    def dimension(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        base = np.array(
            [len(text) % 7, text.count(" "), text.count("B"), text.count("s")],
            dtype=np.float32,
        )
        norm = np.linalg.norm(base) or 1.0
        return base / norm

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def _chunks() -> list[Chunk]:
    return [
        Chunk("BM25 keyword search uses term frequency", "doc.md", 0, 40, "fixed"),
        Chunk("Neural embeddings capture semantic meaning", "doc.md", 1, 38, "fixed"),
        Chunk("Hybrid retrieval combines dense and sparse methods", "doc.md", 2, 45, "fixed"),
    ]


def test_reciprocal_rank_fusion_orders_shared_hits():
    fused = reciprocal_rank_fusion([[2, 0], [0, 1]])
    assert fused[0][0] == 0


def test_hybrid_retriever_returns_results():
    embedder = _FakeEmbedder()
    dense = CosineRetriever(embedder)
    hybrid = HybridRetriever(dense)
    chunks = _chunks()
    hybrid.index(chunks)
    results = hybrid.retrieve("keyword BM25", top_k=2)
    assert len(results) == 2
    assert results[0].retrieval_method == "hybrid_rrf"


def test_weighted_hybrid_alpha_extremes():
    embedder = _FakeEmbedder()
    chunks = _chunks()
    bm25_only = WeightedHybridRetriever(embedder, alpha=0.0)
    dense_only = WeightedHybridRetriever(embedder, alpha=1.0)
    bm25_only.index(chunks)
    dense_only.index(chunks)
    r_sparse = bm25_only.retrieve("keyword BM25", top_k=1)
    r_dense = dense_only.retrieve("semantic meaning", top_k=1)
    assert r_sparse[0].chunk.chunk_id == 0
    assert r_dense[0].rank == 1
