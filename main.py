#!/usr/bin/env python3
"""CLI for RAG Retrieval Lab."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    DATA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_CHAT_MODEL,
    DEFAULT_TOP_K,
    EXPERIMENTS_DIR,
    GEMINI_API_KEY_ENV,
    OPENAI_API_KEY_ENV,
    ANTHROPIC_API_KEY_ENV,
)
from evaluators.dashboard import EvaluationDashboard  # noqa: E402
from pipeline import PipelineConfig, RAGPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG Retrieval Lab CLI")
    p.add_argument("--query", required=True)
    p.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "semantic", "parent_child", "document_based", "agent"],
        default="recursive",
    )
    p.add_argument("--model", choices=["bge-small", "minilm", "openai", "cohere"], default="bge-small")
    p.add_argument(
        "--retrieval",
        choices=["dense", "bm25", "hybrid", "weighted_hybrid", "splade", "colbert"],
        default="hybrid",
    )
    p.add_argument("--index-backend", choices=["memory", "faiss", "qdrant", "pinecone"], default="memory")
    p.add_argument("--query-expansion", choices=["none", "multi", "hyde"], default="none")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--generator", choices=["gemini", "openai", "claude"], default="gemini")
    p.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    p.add_argument("--openai-model", default=DEFAULT_OPENAI_CHAT_MODEL)
    p.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate:
        keys = {"gemini": GEMINI_API_KEY_ENV, "openai": OPENAI_API_KEY_ENV, "claude": ANTHROPIC_API_KEY_ENV}
        if not os.getenv(keys[args.generator]):
            print(f"Missing {keys[args.generator]}")
            sys.exit(1)

    cfg = PipelineConfig(
        chunking_strategy=args.strategy,
        embedding_model=args.model,
        retrieval_mode=args.retrieval,
        index_backend=args.index_backend,
        query_expansion=args.query_expansion,
        use_reranker=args.rerank,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        data_dir=args.data_dir,
        experiments_dir=EXPERIMENTS_DIR,
        generator_provider=args.generator,
        gemini_model=args.gemini_model,
        openai_model=args.openai_model,
        claude_model=args.claude_model,
    )
    pipe = RAGPipeline(cfg)
    if not pipe.load_documents():
        sys.exit(1)
    print(f"Indexed {pipe.build_index()} chunks")
    res, gen, path = pipe.run_query(args.query, save_experiment=not args.no_save, top_k=args.top_k, generate=args.generate, generator_provider=args.generator)
    print(EvaluationDashboard().to_dataframe(res).to_string(index=False))
    if gen:
        print("\n", gen.answer)
        if pipe.last_citation_report:
            print("\nCitation:", pipe.last_citation_report.to_dict())
    if path:
        print("Saved:", path)


if __name__ == "__main__":
    main()
