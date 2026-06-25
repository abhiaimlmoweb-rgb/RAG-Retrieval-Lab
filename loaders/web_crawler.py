"""Bounded web crawler for ingesting documentation sites."""

from __future__ import annotations

from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from loaders.base import LoadedDocument
from loaders.html_loader import HTMLLoader
from loaders.web_loader import DEFAULT_HEADERS

CRAWLABLE = (".html", ".htm", "", "/")


class WebCrawler:
    """Breadth-first crawl within the same host, up to max_pages."""

    def __init__(self, max_pages: int = 10, timeout: int = 15) -> None:
        self.max_pages = max_pages
        self.timeout = timeout

    def crawl(self, start_url: str) -> list[LoadedDocument]:
        parsed = urlparse(start_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must be http(s)")

        base_host = parsed.netloc
        queue: deque[str] = deque([start_url])
        visited: set[str] = set()
        documents: list[LoadedDocument] = []

        while queue and len(documents) < self.max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException:
                continue

            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype and not url.endswith(CRAWLABLE):
                continue

            text = HTMLLoader.html_to_text(resp.text)
            if text.strip():
                path_slug = urlparse(url).path.replace("/", "_")[:80] or "index"
                documents.append(
                    LoadedDocument(
                        document_name=f"crawl_{base_host}_{path_slug}.html",
                        text=text,
                        page_count=1,
                        source_path=url,
                    )
                )

            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                lp = urlparse(link)
                if lp.netloc == base_host and link not in visited:
                    queue.append(link.split("#")[0])

        return documents
