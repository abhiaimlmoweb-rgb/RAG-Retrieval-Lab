"""
Plain-text and Markdown document loader.

Many RAG corpora include READMEs, notes, and exports as .txt or .md — same
chunking and embedding pipeline as PDFs once normalized to plain text.
"""

from __future__ import annotations

from pathlib import Path

from loaders.base import LoadedDocument, content_hash

TEXT_EXTENSIONS = (".txt", ".md", ".markdown")


class TextLoader:
    """Load .txt and .md files from a directory."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_file(self, file_path: Path | str) -> LoadedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError(f"Expected a text file, got: {path.suffix}")

        text = path.read_text(encoding="utf-8", errors="replace").strip()
        line_count = text.count("\n") + 1 if text else 0
        return LoadedDocument(
            document_name=path.name,
            text=text,
            page_count=line_count,
            source_path=str(path.resolve()),
            content_hash=content_hash(text),
        )

    def load_all(self) -> list[LoadedDocument]:
        files: list[Path] = []
        for ext in TEXT_EXTENSIONS:
            files.extend(self.data_dir.glob(f"*{ext}"))
        return [self.load_file(p) for p in sorted(set(files))]
