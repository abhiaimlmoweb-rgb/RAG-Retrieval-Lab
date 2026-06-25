# RAG Lab Notes

## What is RAG?

Retrieval-Augmented Generation (RAG) combines **retrieval** (finding relevant documents)
with **generation** (producing an answer with an LLM). The LLM only sees retrieved
chunks as context, which reduces hallucinations and keeps answers grounded in your data.

## Retrieval strategies in this lab

### Dense (embedding) retrieval
- Maps text to vectors with models like BGE or MiniLM
- Good at semantic similarity and paraphrases
- Weak on exact IDs, codes, and rare keywords

### BM25 (lexical) retrieval
- Classic keyword search based on term frequency
- Strong on exact matches and technical terms
- Misses paraphrases and synonyms

### Hybrid (RRF)
- Runs both dense and BM25, merges rankings with Reciprocal Rank Fusion
- Often the best default in production RAG systems

### Cross-encoder reranking
- After retrieving top-N candidates, a cross-encoder scores (query, chunk) pairs jointly
- More accurate than bi-encoder similarity, but slower — use on a small pool only

## Chunking tips

- **Recursive** chunking (default) tries paragraph and sentence boundaries first
- **512 characters** with **64 overlap** is a solid starting point
- Too small → fragmented context; too large → diluted relevance

## Gemini generation

Set `GEMINI_API_KEY` in `.env` to enable answer generation. The model receives
only retrieved chunks and is instructed to cite sources and admit when context is insufficient.
