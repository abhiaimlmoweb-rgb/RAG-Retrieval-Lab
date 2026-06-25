"""
SPLADE-style sparse neural retriever (learning lab).

Uses a masked language model to produce sparse term weights for documents and
queries, then scores via dot product — a classic learned sparse retrieval method.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from chunkers.base import Chunk
from retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


def _sparse_vector(model, tokenizer, text: str, device: str) -> dict[int, float]:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        logits = model(**tokens).logits
        weights = torch.max(torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1), dim=1).values.squeeze()
    nz = weights.nonzero(as_tuple=False).flatten()
    vec: dict[int, float] = {}
    for idx in nz.tolist():
        vec[int(idx)] = float(weights[idx].item())
    return vec


def _dot(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class SpladeRetriever(BaseRetriever):
    """Learned sparse retrieval with SPLADE term weighting."""

    def __init__(self, model_name: str = SPLADE_MODEL) -> None:
        self.model_name = model_name
        self._chunks: list[Chunk] = []
        self._doc_vectors: list[dict[int, float]] = []
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._model is None:
            logger.info("Loading SPLADE model %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()

    @property
    def is_indexed(self) -> bool:
        return len(self._chunks) > 0 and len(self._doc_vectors) == len(self._chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def index(self, chunks: list[Chunk]) -> None:
        self._load()
        self._chunks = list(chunks)
        self._doc_vectors = []
        for chunk in chunks:
            self._doc_vectors.append(_sparse_vector(self._model, self._tokenizer, chunk.text, self._device))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.is_indexed:
            raise RuntimeError("SPLADE retriever not indexed")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        start = time.perf_counter()
        q_vec = _sparse_vector(self._model, self._tokenizer, query, self._device)
        scores = np.array([_dot(q_vec, d) for d in self._doc_vectors], dtype=np.float64)
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:k]
        elapsed_ms = (time.perf_counter() - start) * 1000

        max_score = float(scores[top_indices[0]]) if len(top_indices) else 1.0
        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self._chunks[int(idx)]
            norm = float(scores[idx]) / max_score if max_score > 0 else 0.0
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    similarity_score=norm,
                    rank=rank,
                    latency_ms=elapsed_ms,
                    source_document=chunk.document_name,
                    retrieval_method="splade",
                )
            )
        return results
