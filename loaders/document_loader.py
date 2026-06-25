"""
Unified document loader.

Aggregates PDF, plain-text, and Markdown loaders so the pipeline can ingest
mixed corpora from a single data/ folder.
"""

from __future__ import annotations

from pathlib import Path

from loaders.pdf_loader import LoadedDocument, PDFLoader
from loaders.text_loader import TextLoader


class DocumentLoader:
    """Load all supported document types from a directory."""

    SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".markdown")

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self._pdf = PDFLoader(self.data_dir)
        self._text = TextLoader(self.data_dir)

    def load_file(self, file_path: Path | str) -> LoadedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._pdf.load_file(path)
        if suffix in (".txt", ".md", ".markdown"):
            return self._text.load_file(path)
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
        )

    def load_all(self) -> list[LoadedDocument]:
        docs = self._pdf.load_all()
        docs.extend(self._text.load_all())
        return sorted(docs, key=lambda d: d.document_name.lower())

    def load_from_paths(self, paths: list[Path | str]) -> list[LoadedDocument]:
        return [self.load_file(p) for p in paths]
