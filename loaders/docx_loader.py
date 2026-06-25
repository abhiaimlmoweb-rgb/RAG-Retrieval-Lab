"""DOCX loader with paragraph and table extraction."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from loaders.base import LoadedDocument, content_hash


def _iter_block_items(parent):
    from docx.document import Document as DocxDocument

    if isinstance(parent, DocxDocument):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._element
    for child in parent_elm.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


class DOCXLoader:
    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _table_to_text(table: Table) -> str:
        rows: list[str] = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    def load_file(self, file_path: Path | str) -> LoadedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX not found: {path}")

        doc = Document(path)
        blocks: list[str] = []
        for block in _iter_block_items(doc):
            if isinstance(block, Paragraph):
                t = block.text.strip()
                if t:
                    blocks.append(t)
            elif isinstance(block, Table):
                t = self._table_to_text(block)
                if t:
                    blocks.append(t)

        text = "\n\n".join(blocks)
        return LoadedDocument(
            document_name=path.name,
            text=text,
            page_count=len(blocks),
            source_path=str(path.resolve()),
            content_hash=content_hash(text),
        )

    def load_all(self) -> list[LoadedDocument]:
        return [self.load_file(p) for p in sorted(self.data_dir.glob("*.docx"))]
