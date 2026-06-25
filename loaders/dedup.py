"""Document deduplication by content hash."""

from __future__ import annotations

from loaders.base import LoadedDocument


def deduplicate_documents(
    documents: list[LoadedDocument],
    *,
    keep: str = "latest",
) -> list[LoadedDocument]:
    """
    Remove exact duplicate content. When names differ but hash matches, keep one.

    Args:
        keep: ``latest`` keeps the last seen doc per hash; ``first`` keeps the first.
    """
    seen: dict[str, LoadedDocument] = {}
    for doc in documents:
        key = doc.content_hash
        if key not in seen:
            seen[key] = doc
        elif keep == "latest":
            seen[key] = doc
    return list(seen.values())


def assign_versions(documents: list[LoadedDocument]) -> list[LoadedDocument]:
    """Bump version when the same filename appears with different content."""
    by_name: dict[str, list[LoadedDocument]] = {}
    for doc in documents:
        by_name.setdefault(doc.document_name, []).append(doc)

    versioned: list[LoadedDocument] = []
    for name, group in by_name.items():
        hashes_seen: list[str] = []
        for doc in group:
            if doc.content_hash in hashes_seen:
                continue
            version = len(hashes_seen) + 1
            hashes_seen.append(doc.content_hash)
            versioned.append(
                LoadedDocument(
                    document_name=doc.document_name,
                    text=doc.text,
                    page_count=doc.page_count,
                    source_path=doc.source_path,
                    content_hash=doc.content_hash,
                    version=version,
                )
            )
    return versioned
