"""Optional Redis cache for retrieval results."""

from __future__ import annotations

import hashlib
import json
import os

from config.settings import REDIS_URL_ENV


class RedisQueryCache:
    """Cache retrieval JSON by (query, config hash). Graceful no-op if Redis unavailable."""

    def __init__(self, url: str | None = None, ttl_seconds: int = 3600) -> None:
        self.ttl = ttl_seconds
        self._client = None
        redis_url = url or os.getenv(REDIS_URL_ENV)
        if redis_url:
            try:
                import redis

                self._client = redis.from_url(redis_url, decode_responses=True)
                self._client.ping()
            except Exception:
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _key(self, query: str, config_hash: str) -> str:
        digest = hashlib.sha256(f"{config_hash}:{query}".encode()).hexdigest()[:24]
        return f"rag:retrieve:{digest}"

    def get(self, query: str, config_hash: str) -> list[dict] | None:
        if not self._client:
            return None
        raw = self._client.get(self._key(query, config_hash))
        return json.loads(raw) if raw else None

    def set(self, query: str, config_hash: str, payload: list[dict]) -> None:
        if not self._client:
            return
        self._client.setex(self._key(query, config_hash), self.ttl, json.dumps(payload))
