"""Tests for document-based and agent-based chunking."""

from chunkers.agent_based import AgentChunker, _parse_agent_json
from chunkers.document_based import DocumentBasedChunker

SAMPLE_MD = """# RAG Overview

Retrieval-Augmented Generation combines search with LLMs.

## Dense Retrieval

Maps text to embedding vectors for semantic search.

## BM25

Keyword-based lexical retrieval.

| Method | Type |
|--------|------|
| Dense | Semantic |
| BM25 | Lexical |
"""


def test_document_based_splits_on_headings():
    chunker = DocumentBasedChunker(chunk_size=512)
    chunks = chunker.chunk_text(SAMPLE_MD, "notes.md")
    assert len(chunks) >= 3
    titles = {c.metadata.get("section_title") for c in chunks}
    assert "Dense Retrieval" in titles or any("Dense" in c.text for c in chunks)
    assert any(c.metadata.get("chunk_type") == "table" for c in chunks)


def test_document_based_includes_section_path_metadata():
    chunker = DocumentBasedChunker(chunk_size=200)
    chunks = chunker.chunk_text(SAMPLE_MD, "notes.md")
    dense_chunks = [c for c in chunks if c.metadata.get("section_title") == "Dense Retrieval"]
    assert dense_chunks
    assert "RAG Overview" in dense_chunks[0].metadata.get("section_path", "")


def test_agent_chunker_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    chunker = AgentChunker(api_key=None)
    chunks = chunker.chunk_text("Hello world. " * 50, "test.txt")
    assert chunks
    assert all(c.chunking_strategy == "agent" for c in chunks)


def test_parse_agent_json_with_code_fence():
    raw = '```json\n[{"title": "Intro", "text": "Hello"}]\n```'
    items = _parse_agent_json(raw)
    assert items[0]["title"] == "Intro"
