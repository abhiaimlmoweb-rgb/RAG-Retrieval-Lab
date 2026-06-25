"""
Agent-based chunking using Gemini.

An LLM analyzes each document and proposes logical chunk boundaries with titles.
Falls back to recursive chunking when no API key is available.
"""

from __future__ import annotations

import json
import re

from google import genai
from google.genai import types

from chunkers.base import BaseChunker, Chunk
from chunkers.recursive import RecursiveChunker
from config.settings import DEFAULT_GEMINI_MODEL, resolve_api_key

AGENT_PROMPT = """You are a document chunking agent for a RAG retrieval system.

Split the document into logical chunks for search and retrieval. Each chunk should:
- Be self-contained and understandable on its own
- Respect natural boundaries (topics, sections, procedures)
- Stay under {max_chars} characters when possible
- Include a short descriptive title

Return ONLY valid JSON — an array of objects:
[{{"title": "Section title", "text": "chunk content..."}}]

Document name: {document_name}

--- DOCUMENT ---
{text}
--- END ---"""


def _parse_agent_json(raw: str) -> list[dict]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Agent response must be a JSON array")
    return data


class AgentChunker(BaseChunker):
    """Gemini-powered chunk boundary proposal with recursive fallback."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        *,
        api_key: str | None = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        max_doc_chars: int = 12_000,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, chunking_strategy="agent")
        self._api_key = api_key
        self._model_name = model_name
        self._max_doc_chars = max_doc_chars
        self._fallback = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _resolve_key(self) -> str | None:
        return resolve_api_key("gemini", self._api_key)

    def _fallback_chunks(self, text: str, document_name: str) -> list[Chunk]:
        return [
            Chunk(
                text=c.text,
                document_name=c.document_name,
                chunk_id=c.chunk_id,
                chunk_size=c.chunk_size,
                chunking_strategy=self.chunking_strategy,
                metadata={**c.metadata, "agent_fallback": True},
            )
            for c in self._fallback.chunk_text(text, document_name)
        ]

    def _agent_split(self, text: str, document_name: str) -> list[Chunk]:
        key = self._resolve_key()
        if not key:
            return self._fallback_chunks(text, document_name)

        client = genai.Client(api_key=key)
        prompt = AGENT_PROMPT.format(
            max_chars=self.chunk_size,
            document_name=document_name,
            text=text[: self._max_doc_chars],
        )
        response = client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        raw = response.text.strip() if response.text else "[]"
        items = _parse_agent_json(raw)

        chunks: list[Chunk] = []
        for idx, item in enumerate(items):
            title = str(item.get("title", f"Chunk {idx}")).strip()
            body = str(item.get("content", item.get("text", ""))).strip()
            if not body:
                continue
            chunk_text = f"[{title}]\n{body}" if title else body
            if len(chunk_text) > self.chunk_size * 3:
                sub = self._fallback.chunk_text(chunk_text, document_name)
                for sub_chunk in sub:
                    chunks.append(
                        Chunk(
                            text=sub_chunk.text,
                            document_name=document_name,
                            chunk_id=len(chunks),
                            chunk_size=len(sub_chunk.text),
                            chunking_strategy=self.chunking_strategy,
                            metadata={
                                "agent_title": title,
                                "agent_chunk": True,
                                "agent_split": True,
                            },
                        )
                    )
            else:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        document_name=document_name,
                        chunk_id=len(chunks),
                        chunk_size=len(chunk_text),
                        chunking_strategy=self.chunking_strategy,
                        metadata={"agent_title": title, "agent_chunk": True},
                    )
                )
        return chunks if chunks else self._fallback_chunks(text, document_name)

    def chunk_text(self, text: str, document_name: str) -> list[Chunk]:
        if not text.strip():
            return []

        if len(text) <= self._max_doc_chars:
            return self._agent_split(text, document_name)

        # Long documents: agent-chunk each window, then merge
        all_chunks: list[Chunk] = []
        step = self._max_doc_chars + self.chunk_overlap
        for start in range(0, len(text), step):
            window = text[start : start + self._max_doc_chars]
            window_chunks = self._agent_split(window, document_name)
            for wc in window_chunks:
                all_chunks.append(
                    Chunk(
                        text=wc.text,
                        document_name=document_name,
                        chunk_id=len(all_chunks),
                        chunk_size=wc.chunk_size,
                        chunking_strategy=self.chunking_strategy,
                        metadata={**wc.metadata, "window_start": start},
                    )
                )
            if start + self._max_doc_chars >= len(text):
                break
        return all_chunks
