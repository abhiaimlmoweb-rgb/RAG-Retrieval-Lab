"""
PDF document loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from loaders.base import LoadedDocument, content_hash


class PDFLoader:
    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_file(self, file_path: Path | str) -> LoadedDocument:
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
                content_hash=content_hash(text),
            )
        finally:
            doc.close()

    def load_all(self) -> list[LoadedDocument]:
        pdf_files = sorted(self.data_dir.glob("*.pdf"))
        return [self.load_file(p) for p in pdf_files]

    def iter_documents(self) -> Iterator[LoadedDocument]:
        for pdf_path in sorted(self.data_dir.glob("*.pdf")):
            yield self.load_file(pdf_path)


# Backward compatibility
__all__ = ["LoadedDocument", "PDFLoader"]
