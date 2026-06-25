"""
PDF document loader.

In production RAG, ingestion is the first stage of the pipeline: raw documents
are normalized into plain text before chunking. Quality here directly affects
retrieval recall — garbled or missing text cannot be recovered later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


@dataclass(frozen=True)
class LoadedDocument:
    """A single document extracted from a PDF."""

    document_name: str
    text: str
    page_count: int
    source_path: str


class PDFLoader:
    """
    Load and extract text from PDF files using PyMuPDF.

    Supports loading a single file or all PDFs in a directory. Each page is
    concatenated with newlines to preserve rough paragraph structure.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_file(self, file_path: Path | str) -> LoadedDocument:
        """Extract text from one PDF file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        doc = fitz.open(path)
        try:
            pages = [page.get_text("text") for page in doc]
            text = "\n".join(pages).strip()
            return LoadedDocument(
                document_name=path.name,
                text=text,
                page_count=len(doc),
                source_path=str(path.resolve()),
            )
        finally:
            doc.close()

    def load_all(self) -> list[LoadedDocument]:
        """Load every PDF in the configured data directory."""
        pdf_files = sorted(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            return []
        return [self.load_file(p) for p in pdf_files]

    def iter_documents(self) -> Iterator[LoadedDocument]:
        """Lazy iterator over PDFs in the data directory."""
        for pdf_path in sorted(self.data_dir.glob("*.pdf")):
            yield self.load_file(pdf_path)
