#!/usr/bin/env python3
"""
CLI entry point for RAG Retrieval Lab.

Examples:
    # Retrieval only (hybrid search)
    python main.py --query "What is chunking?" --retrieval hybrid

    # Full RAG with Gemini generation
    python main.py --query "What is RAG?" --retrieval hybrid --rerank --generate
"""

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
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TOP_K,
    EXPERIMENTS_DIR,
    GEMINI_API_KEY_ENV,
)
from evaluators.dashboard import EvaluationDashboard  # noqa: E402
from pipeline import PipelineConfig, RAGPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Retrieval Lab — retrieval and full RAG with Gemini"
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive"],
        default="recursive",
        help="Chunking strategy",
    )
    parser.add_argument(
        "--model",
        choices=["bge-small", "minilm"],
        default="bge-small",
        help="Embedding model key",
    )
    parser.add_argument(
        "--retrieval",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval backend",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank candidates with a cross-encoder",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a grounded answer with Gemini (requires API key)",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model name",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-save", action="store_true", help="Skip experiment JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.generate and not os.getenv(GEMINI_API_KEY_ENV):
        print(
            f"Error: --generate requires {GEMINI_API_KEY_ENV}. "
            f"Set it in .env or export it in your shell."
        )
        sys.exit(1)

    config = PipelineConfig(
        chunking_strategy=args.strategy,
        embedding_model=args.model,
        retrieval_mode=args.retrieval,
        use_reranker=args.rerank,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        data_dir=args.data_dir,
        experiments_dir=EXPERIMENTS_DIR,
        gemini_model=args.gemini_model,
    )

    pipeline = RAGPipeline(config)
    docs = pipeline.load_documents()
    if not docs:
        print(f"No documents found in {args.data_dir}. Add PDF, .txt, or .md files.")
        sys.exit(1)

    print(f"Loaded {len(docs)} document(s), building index...")
    n = pipeline.build_index()
    print(
        f"Indexed {n} chunks | retrieval={args.retrieval} | "
        f"embeddings={pipeline.embedding_model_id}"
    )

    results, generation, path = pipeline.run_query(
        args.query,
        save_experiment=not args.no_save,
        top_k=args.top_k,
        generate=args.generate,
    )

    dashboard = EvaluationDashboard()
    print("\n--- Retrieval Results ---")
    print(dashboard.to_dataframe(results).to_string(index=False))

    if generation is not None:
        print("\n--- Generated Answer (Gemini) ---")
        print(f"Model: {generation.model} | Latency: {generation.latency_ms:.0f} ms")
        print(generation.answer)

    if path:
        print(f"\nExperiment saved: {path}")


if __name__ == "__main__":
    main()
