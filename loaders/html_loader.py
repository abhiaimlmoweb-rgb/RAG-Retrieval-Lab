"""
HTML document loader.

Extracts visible text from saved HTML files using BeautifulSoup — useful for
web-scraped pages, exported Confluence/Notion HTML, and documentation mirrors.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from loaders.base import LoadedDocument, content_hash


class HTMLLoader:
    """Load .html / .htm files from a directory."""

    EXTENSIONS = (".html", ".htm")

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def html_to_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    def load_file(self, file_path: Path | str) -> LoadedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"HTML file not found: {path}")
        if path.suffix.lower() not in self.EXTENSIONS:
            raise ValueError(f"Expected HTML file, got: {path.suffix}")

        raw = path.read_text(encoding="utf-8", errors="replace")
        text = self.html_to_text(raw)
        return LoadedDocument(
            document_name=path.name,
            text=text,
            page_count=1,
            source_path=str(path.resolve()),
            content_hash=content_hash(text),
        )

    def load_all(self) -> list[LoadedDocument]:
        files: list[Path] = []
        for ext in self.EXTENSIONS:
            files.extend(self.data_dir.glob(f"*{ext}"))
        return [self.load_file(p) for p in sorted(set(files))]
