"""Stable document names for web URLs."""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse


def url_document_slug(url: str, *, prefix: str = "web", max_path_len: int = 80) -> str:
    """
    Build a unique, filesystem-safe document name from a URL.

    Includes a short hash of the full URL so truncated paths cannot collide.
    """
    parsed = urlparse(url)
    host = parsed.netloc.replace(":", "_")
    path_slug = parsed.path.replace("/", "_").strip("_")[:max_path_len] or "index"
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    base = f"{prefix}_{host}_{path_slug}_{url_hash}"
    ext = ".html" if not base.endswith(".html") else ""
    return f"{base}{ext}"[:140]
