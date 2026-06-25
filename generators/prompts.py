"""Shared RAG prompt templates."""

from __future__ import annotations

from retrievers.base import RetrievalResult

RAG_SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question using ONLY the provided context.

Rules:
- If the context does not contain enough information, say so clearly.
- Cite source document names in brackets when referencing specific facts.
- Be concise but thorough. Use bullet points when listing multiple items.
- Do not invent facts not supported by the context.
"""


def format_context(results: list[RetrievalResult]) -> str:
    if not results:
        return "(No context retrieved.)"
    blocks = [
        f"--- Source: {r.source_document} (chunk {r.chunk.chunk_id}) ---\n{r.chunk.text}"
        for r in results
    ]
    return "\n\n".join(blocks)


def build_user_prompt(query: str, results: list[RetrievalResult]) -> str:
    return f"""Context:
{format_context(results)}

Question: {query}

Answer:"""
