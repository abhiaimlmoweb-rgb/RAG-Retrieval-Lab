"""Tests for document deduplication and versioning."""

from loaders.base import LoadedDocument, content_hash
from loaders.dedup import assign_versions, deduplicate_documents


def _doc(name: str, text: str) -> LoadedDocument:
    return LoadedDocument(
        document_name=name,
        text=text,
        page_count=1,
        source_path=f"/tmp/{name}",
        content_hash=content_hash(text),
    )


def test_assign_versions_keeps_different_content_same_name():
    docs = assign_versions([_doc("same.html", "version one"), _doc("same.html", "version two")])
    assert len(docs) == 2
    versions = sorted(d.version for d in docs)
    assert versions == [1, 2]


def test_deduplicate_documents_by_hash():
    docs = deduplicate_documents([_doc("a.txt", "hello"), _doc("b.txt", "hello")])
    assert len(docs) == 1


def test_assign_versions_skips_identical_hash():
    docs = assign_versions([_doc("x.html", "same"), _doc("x.html", "same")])
    assert len(docs) == 1
