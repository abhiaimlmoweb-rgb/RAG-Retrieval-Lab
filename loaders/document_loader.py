"""
Unified document loader.

Aggregates PDF, text, Markdown, HTML, and DOCX loaders. Web URLs are fetched
separately via WebLoader.load_url().
"""

from __future__ import annotations

from pathlib import Path

from loaders.docx_loader import DOCXLoader
from loaders.html_loader import HTMLLoader
from loaders.base import LoadedDocument
from loaders.pdf_loader import PDFLoader
from loaders.text_loader import TextLoader


class DocumentLoader:
    """Load all supported document types from a directory."""

    SUPPORTED_EXTENSIONS = (
        ".pdf", ".txt", ".md", ".markdown", ".html", ".htm", ".docx",
    )

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self._pdf = PDFLoader(self.data_dir)
        self._text = TextLoader(self.data_dir)
        self._html = HTMLLoader(self.data_dir)
        self._docx = DOCXLoader(self.data_dir)

    def load_file(self, file_path: Path | str) -> LoadedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._pdf.load_file(path)
        if suffix in (".txt", ".md", ".markdown"):
            return self._text.load_file(path)
        if suffix in (".html", ".htm"):
            return self._html.load_file(path)
        if suffix == ".docx":
            return self._docx.load_file(path)
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
        )

    def load_all(self) -> list[LoadedDocument]:
        docs: list[LoadedDocument] = []
        docs.extend(self._pdf.load_all())
        docs.extend(self._text.load_all())
        docs.extend(self._html.load_all())
        docs.extend(self._docx.load_all())
        return sorted(docs, key=lambda d: d.document_name.lower())

    def load_from_paths(self, paths: list[Path | str]) -> list[LoadedDocument]:
        return [self.load_file(p) for p in paths]
