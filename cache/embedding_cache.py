"""
Disk cache for document embeddings.

Avoids re-embedding unchanged chunks on rebuild — useful when iterating on
chunk size or retrieval mode without re-downloading models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from config.settings import CACHE_DIR


class EmbeddingCache:
    """File-backed cache keyed by (model_id, text_hash)."""

    def __init__(self, cache_dir: Path | str = CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model_id: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        safe_model = model_id.replace("/", "_")
        return f"{safe_model}_{digest}"

    def _path(self, model_id: str, text: str) -> Path:
        return self.cache_dir / f"{self._key(model_id, text)}.json"

    def get_batch(self, model_id: str, texts: list[str]) -> tuple[list[int], list[str]]:
        """
        Return indices of cache misses and their texts.

        Hits are not returned here; caller merges via set_batch.
        """
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for i, text in enumerate(texts):
            if not self._path(model_id, text).exists():
                missing_indices.append(i)
                missing_texts.append(text)
        return missing_indices, missing_texts

    def load_vector(self, model_id: str, text: str) -> np.ndarray | None:
        path = self._path(model_id, text)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(data["vector"], dtype=np.float32)

    def save_vector(self, model_id: str, text: str, vector: np.ndarray) -> None:
        path = self._path(model_id, text)
        payload = {"model_id": model_id, "vector": vector.tolist()}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def embed_with_cache(
        self,
        model_id: str,
        texts: list[str],
        compute_fn,
        *,
        dimension: int,
    ) -> np.ndarray:
        """Return embedding matrix, using disk cache where possible."""
        if not texts:
            return np.empty((0, dimension), dtype=np.float32)

        vectors: list[np.ndarray] = []
        missing_indices, missing_texts = self.get_batch(model_id, texts)

        for text in texts:
            cached = self.load_vector(model_id, text)
            if cached is not None:
                vectors.append(cached)
            else:
                vectors.append(np.zeros(dimension, dtype=np.float32))  # placeholder

        if missing_texts:
            computed = compute_fn(missing_texts)
            for idx, vec in zip(missing_indices, computed, strict=True):
                vectors[idx] = np.asarray(vec, dtype=np.float32)
                self.save_vector(model_id, texts[idx], vectors[idx])

        return np.vstack(vectors)
