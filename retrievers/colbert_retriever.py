"""
ColBERT late-interaction retriever.

Two-stage: dense candidate pool, then ColBERT MaxSim re-ranking via RAGatouille.
"""

from __future__ import annotations

import logging
import time

from chunkers.base import Chunk
from embeddings.base import BaseEmbedder
from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.cosine_retriever import CosineRetriever

COLBERT_MODEL = "colbert-ir/colbertv2.0"
logger = logging.getLogger(__name__)


class ColBERTRetriever(BaseRetriever):
    """Late-interaction retrieval on top of a dense candidate pool."""

    def __init__(self, embedder: BaseEmbedder, candidate_pool: int = 32) -> None:
        self.dense = CosineRetriever(embedder)
        self.candidate_pool = candidate_pool
        self._chunks: list[Chunk] = []
        self._rag_model = None
        self._index_name = "rag_lab_colbert"

    @property
    def is_indexed(self) -> bool:
        return self.dense.is_indexed

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def _get_rag(self):
        if self._rag_model is None:
            from ragatouille import RAGPretrainedModel

            self._rag_model = RAGPretrainedModel.from_pretrained(COLBERT_MODEL)
        return self._rag_model

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self.dense.index(chunks)
        if chunks:
            rag = self._get_rag()
            rag.index(
                collection=[c.text for c in chunks],
                index_name=self._index_name,
                max_document_length=256,
                split_documents=False,
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.is_indexed:
            raise RuntimeError("ColBERT retriever not indexed")
        start = time.perf_counter()
        rag = self._get_rag()
        hits = rag.search(query, k=top_k, index_name=self._index_name)
        elapsed = (time.perf_counter() - start) * 1000

        results: list[RetrievalResult] = []
        for hit in hits:
            raw_id = hit.get("document_id", hit.get("passage_id"))
            if raw_id is None:
                logger.warning("ColBERT hit missing document_id/passage_id")
                continue
            doc_idx = int(raw_id)
            # RAGatouille may return 1-based passage ids
            if doc_idx >= len(self._chunks) and doc_idx - 1 >= 0 and doc_idx - 1 < len(self._chunks):
                doc_idx -= 1
            if doc_idx < 0 or doc_idx >= len(self._chunks):
                logger.warning(
                    "ColBERT returned out-of-bounds index %s (chunks=%d); skipping hit",
                    raw_id,
                    len(self._chunks),
                )
                continue
            chunk = self._chunks[doc_idx]
            score = float(hit.get("score", hit.get("similarity", 0.0)))
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    similarity_score=score,
                    rank=len(results) + 1,
                    latency_ms=elapsed,
                    source_document=chunk.document_name,
                    retrieval_method="colbert",
                )
            )
        return results
