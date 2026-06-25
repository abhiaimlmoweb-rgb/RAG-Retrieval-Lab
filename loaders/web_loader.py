"""
Web page loader.

Fetches a URL and extracts main text content — handy for one-off ingestion of
online documentation without saving files manually.
"""

from __future__ import annotations

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from loaders.base import LoadedDocument, content_hash
from loaders.html_loader import HTMLLoader
from loaders.web_naming import url_document_slug

DEFAULT_HEADERS = {
    "User-Agent": "RAG-Retrieval-Lab/1.0 (educational; +https://github.com)",
}


class WebLoader:
    """Fetch and parse a single web page into a LoadedDocument."""

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def load_url(self, url: str) -> LoadedDocument:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL must be http(s): {url}")

        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=self.timeout)
        response.raise_for_status()
        response.encoding = response.encoding or "utf-8"

        text = HTMLLoader.html_to_text(response.text)
        if not text.strip():
            raise ValueError(f"No extractable text from URL: {url}")

        return LoadedDocument(
            document_name=url_document_slug(url, prefix="web"),
            text=text,
            page_count=1,
            source_path=url,
            content_hash=content_hash(text),
        )
