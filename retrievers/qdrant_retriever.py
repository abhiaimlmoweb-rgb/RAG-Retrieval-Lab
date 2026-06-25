"""Qdrant vector database retriever (local or remote)."""

from __future__ import annotations

import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from chunkers.base import Chunk
from config.settings import QDRANT_COLLECTION, QDRANT_PATH, QDRANT_URL
from embeddings.base import BaseEmbedder
from retrievers.base import BaseRetriever, RetrievalResult


class QdrantRetriever(BaseRetriever):
    """Dense retrieval backed by Qdrant."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        *,
        collection: str = QDRANT_COLLECTION,
        url: str | None = QDRANT_URL,
        path: str | None = QDRANT_PATH,
    ) -> None:
        self.embedder = embedder
        self.collection = collection
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(path=path or str(QDRANT_PATH))
        self._chunks: list[Chunk] = []

    @property
    def is_indexed(self) -> bool:
        return len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.embedder.dimension, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i].tolist(),
                payload={
                    "idx": i,
                    "document_name": chunks[i].document_name,
                    "chunk_id": chunks[i].chunk_id,
                    "text": chunks[i].text,
                },
            )
            for i in range(len(chunks))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.is_indexed:
            raise RuntimeError("Qdrant retriever not indexed")
        start = time.perf_counter()
        qvec = self.embedder.embed_query(query).tolist()
        hits = self.client.search(collection_name=self.collection, query_vector=qvec, limit=top_k)
        elapsed = (time.perf_counter() - start) * 1000

        results: list[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            idx = int(hit.payload["idx"])
            chunk = self._chunks[idx]
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    similarity_score=float(hit.score),
                    rank=rank,
                    latency_ms=elapsed,
                    source_document=chunk.document_name,
                    retrieval_method="qdrant",
                )
            )
        return results
