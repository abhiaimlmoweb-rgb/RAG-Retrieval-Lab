"""Pinecone managed vector index retriever."""

from __future__ import annotations

import os
import time

from pinecone import Pinecone, ServerlessSpec

from chunkers.base import Chunk
from config.settings import PINECONE_API_KEY_ENV, PINECONE_INDEX_NAME
from embeddings.base import BaseEmbedder
from retrievers.base import BaseRetriever, RetrievalResult


class PineconeRetriever(BaseRetriever):
    """Dense retrieval via Pinecone."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        *,
        index_name: str = PINECONE_INDEX_NAME,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.getenv(PINECONE_API_KEY_ENV)
        if not key:
            raise ValueError(f"Pinecone API key required. Set {PINECONE_API_KEY_ENV}.")
        self.embedder = embedder
        self.index_name = index_name
        self.pc = Pinecone(api_key=key)
        self._chunks: list[Chunk] = []
        self._index = None

    @property
    def is_indexed(self) -> bool:
        return len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def _ensure_index(self) -> None:
        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedder.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = self.pc.Index(self.index_name)

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        if not chunks:
            return
        self._ensure_index()
        assert self._index is not None

        try:
            self._index.delete(delete_all=True)
        except Exception:
            pass

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_documents(texts)
        records = [
            {
                "id": f"chunk-{i}",
                "values": vectors[i].tolist(),
                "metadata": {
                    "idx": i,
                    "document_name": chunks[i].document_name,
                    "chunk_id": chunks[i].chunk_id,
                },
            }
            for i in range(len(chunks))
        ]
        self._index.upsert(vectors=records)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.is_indexed or self._index is None:
            raise RuntimeError("Pinecone retriever not indexed")
        start = time.perf_counter()
        qvec = self.embedder.embed_query(query).tolist()
        response = self._index.query(vector=qvec, top_k=top_k, include_metadata=True)
        elapsed = (time.perf_counter() - start) * 1000

        results: list[RetrievalResult] = []
        for rank, match in enumerate(response.matches, start=1):
            idx = int(match.metadata["idx"])
            chunk = self._chunks[idx]
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    similarity_score=float(match.score),
                    rank=rank,
                    latency_ms=elapsed,
                    source_document=chunk.document_name,
                    retrieval_method="pinecone",
                )
            )
        return results
