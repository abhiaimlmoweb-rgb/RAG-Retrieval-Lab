#!/usr/bin/env python3
"""
Batch retrieval evaluation CLI.

Example:
    python eval.py --dataset data/eval_qa.json --retrieval hybrid --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR, DEFAULT_TOP_K, EXPERIMENTS_DIR  # noqa: E402
from evaluators.retrieval_eval import EvalDataset, RetrievalEvaluator  # noqa: E402
from pipeline import PipelineConfig, RAGPipeline  # noqa: E402
from utils.observability import get_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch retrieval evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_DIR / "eval_qa.json",
        help="JSON file with labeled queries",
    )
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"], default="recursive")
    parser.add_argument("--model", choices=["bge-small", "minilm", "openai", "cohere"], default="bge-small")
    parser.add_argument("--retrieval", choices=["dense", "bm25", "hybrid", "colbert"], default="hybrid")
    parser.add_argument("--index-backend", choices=["memory", "faiss", "qdrant", "pinecone"], default="memory")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("eval")

    config = PipelineConfig(
        chunking_strategy=args.strategy,
        embedding_model=args.model,
        retrieval_mode=args.retrieval,
        index_backend=args.index_backend,
        use_reranker=args.rerank,
        top_k=args.top_k,
        data_dir=args.data_dir,
        experiments_dir=EXPERIMENTS_DIR,
    )

    pipeline = RAGPipeline(config)
    docs = pipeline.load_documents()
    if not docs:
        print(f"No documents in {args.data_dir}")
        sys.exit(1)

    logger.info("Building index for %d documents", len(docs))
    n = pipeline.build_index()
    logger.info("Indexed %d chunks", n)

    dataset = EvalDataset.load(args.dataset)
    evaluator = RetrievalEvaluator(pipeline)
    per_query = evaluator.run(dataset, k=args.top_k)
    summary = evaluator.summarize(per_query)

    print("\n--- Per-query metrics ---")
    print(evaluator.to_dataframe(per_query).to_string(index=False))
    print("\n--- Summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
