"""Tests for config hash and parent-document retriever."""

import hashlib
import json

from chunkers.base import Chunk
from pipeline import PipelineConfig
from retrievers.bm25_retriever import BM25Retriever
from retrievers.parent_document_retriever import ParentDocumentRetriever


def _hash_config(config: PipelineConfig, chunk_count: int) -> str:
    payload = {
        "chunking": config.chunking_strategy,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embed": config.embedding_model,
        "retrieval": config.retrieval_mode,
        "backend": config.index_backend,
        "chunks": chunk_count,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def test_config_hash_uses_chunk_size_not_top_k():
    base = dict(chunk_size=512, top_k=10, chunk_overlap=64)
    h1 = _hash_config(PipelineConfig(**base), 1)
    h2 = _hash_config(PipelineConfig(chunk_size=1024, top_k=10, chunk_overlap=64), 1)
    h3 = _hash_config(PipelineConfig(chunk_size=512, top_k=20, chunk_overlap=64), 1)
    assert h1 != h2
    assert h1 == h3


def test_parent_retriever_dedupes_by_chunk_id_without_parent_metadata():
    chunks = [
        Chunk("alpha", "doc.md", 0, 5, "recursive"),
        Chunk("beta", "doc.md", 1, 4, "recursive"),
    ]
    inner = BM25Retriever()
    parent = ParentDocumentRetriever(inner)
    parent.index(chunks)
    results = parent.retrieve("alpha beta", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id != results[1].chunk.chunk_id
