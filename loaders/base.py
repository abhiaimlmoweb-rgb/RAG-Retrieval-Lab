"""Shared document types."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class LoadedDocument:
    """Normalized document with optional versioning metadata."""

    document_name: str
    text: str
    page_count: int
    source_path: str
    content_hash: str = ""

    version: int = 1
