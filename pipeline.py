"""
End-to-end retrieval and RAG pipeline.

Orchestrates ingestion → chunking → embedding → indexing → retrieval
→ optional reranking → optional LLM generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from chunkers.base import BaseChunker, Chunk
from chunkers.fixed import FixedChunker
from chunkers.recursive import RecursiveChunker
from config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RERANK_POOL,
    DEFAULT_TOP_K,
    EMBEDDING_MODELS,
)
from embeddings.base import BaseEmbedder
from embeddings.bge_small import BGEEmbedder
from embeddings.minilm import MiniLMEmbedder
from evaluators.dashboard import EvaluationDashboard
from evaluators.experiment_tracker import ExperimentTracker
from generators.base import BaseGenerator, GenerationResult
from generators.gemini_generator import GeminiGenerator
from loaders.document_loader import DocumentLoader
from loaders.pdf_loader import LoadedDocument
from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.bm25_retriever import BM25Retriever
from retrievers.cosine_retriever import CosineRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.reranker import CrossEncoderReranker


def get_chunker(
    strategy: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> BaseChunker:
    """Factory for chunking strategies."""
    if strategy == "fixed":
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def get_embedder(model_key: str) -> BaseEmbedder:
    """Factory for embedding models."""
    registry: dict[str, Callable[[], BaseEmbedder]] = {
        "bge-small": BGEEmbedder,
        "minilm": MiniLMEmbedder,
    }
    if model_key not in registry:
        raise ValueError(
            f"Unknown embedding model: {model_key}. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[model_key]()


def get_retriever(mode: str, embedder: BaseEmbedder | None = None) -> BaseRetriever:
    """Factory for retrieval backends."""
    if mode == "bm25":
        return BM25Retriever()
    if embedder is None:
        raise ValueError(f"{mode} retrieval requires an embedder")
    if mode == "dense":
        return CosineRetriever(embedder)
    if mode == "hybrid":
        return HybridRetriever(embedder)
    raise ValueError(f"Unknown retrieval mode: {mode}. Choose: dense, bm25, hybrid")


@dataclass
class PipelineConfig:
    """Runtime configuration for a retrieval / RAG experiment."""

    chunking_strategy: str = "recursive"
    embedding_model: str = "bge-small"
    retrieval_mode: str = "hybrid"
    use_reranker: bool = False
    rerank_pool: int = DEFAULT_RERANK_POOL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    data_dir: Path = field(default_factory=lambda: Path("data"))
    experiments_dir: Path = field(default_factory=lambda: Path("experiments"))
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str | None = None


@dataclass
class RAGPipeline:
    """
    Stateful pipeline: build index once, run many queries.

    Supports retrieval-only mode and full RAG with Gemini generation.
    """

    config: PipelineConfig
    loader: DocumentLoader = field(init=False)
    chunker: BaseChunker = field(init=False)
    embedder: BaseEmbedder | None = field(init=False, default=None)
    retriever: BaseRetriever = field(init=False)
    reranker: CrossEncoderReranker | None = field(init=False)
    dashboard: EvaluationDashboard = field(init=False)
    tracker: ExperimentTracker = field(init=False)

    documents: list[LoadedDocument] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.loader = DocumentLoader(self.config.data_dir)
        self.chunker = get_chunker(
            self.config.chunking_strategy,
            self.config.chunk_size,
            self.config.chunk_overlap,
        )
        self.embedder = (
            get_embedder(self.config.embedding_model)
            if self.config.retrieval_mode != "bm25"
            else None
        )
        self.retriever = get_retriever(self.config.retrieval_mode, self.embedder)
        self.reranker = CrossEncoderReranker() if self.config.use_reranker else None
        self.dashboard = EvaluationDashboard()
        self.tracker = ExperimentTracker(self.config.experiments_dir)

    @property
    def embedding_model_id(self) -> str:
        return EMBEDDING_MODELS.get(
            self.config.embedding_model, self.config.embedding_model
        )

    def load_documents(self) -> list[LoadedDocument]:
        """Load all supported files from the data directory."""
        self.documents = self.loader.load_all()
        return self.documents

    def load_documents_from_paths(self, paths: list[Path | str]) -> list[LoadedDocument]:
        """Load specific files (e.g. from Streamlit uploads)."""
        self.documents = self.loader.load_from_paths(paths)
        return self.documents

    def build_index(self) -> int:
        """
        Chunk documents, embed/index chunks.

        Returns:
            Number of indexed chunks.
        """
        doc_pairs = [(d.document_name, d.text) for d in self.documents]
        self.chunks = self.chunker.chunk_documents(doc_pairs)
        self.retriever.index(self.chunks)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Run retrieval, optionally with cross-encoder reranking."""
        k = top_k if top_k is not None else self.config.top_k

        if self.reranker is not None:
            pool_k = min(self.config.rerank_pool, len(self.chunks))
            candidates = self.retriever.retrieve(query, top_k=pool_k)
            return self.reranker.rerank(query, candidates, top_k=k)

        return self.retriever.retrieve(query, top_k=k)

    def generate_answer(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> GenerationResult:
        """Generate a grounded answer with Gemini."""
        generator: BaseGenerator = GeminiGenerator(
            api_key=api_key or self.config.gemini_api_key,
            model_name=model or self.config.gemini_model,
        )
        return generator.generate(query, results)

    def run_query(
        self,
        query: str,
        *,
        save_experiment: bool = True,
        top_k: int | None = None,
        generate: bool = False,
        gemini_api_key: str | None = None,
    ) -> tuple[list[RetrievalResult], GenerationResult | None, Path | None]:
        """
        Retrieve, optionally generate an answer, optionally persist experiment JSON.
        """
        results = self.retrieve(query, top_k=top_k)
        generation: GenerationResult | None = None

        if generate:
            generation = self.generate_answer(
                query,
                results,
                api_key=gemini_api_key,
            )

        saved_path: Path | None = None
        if save_experiment:
            saved_path = self.tracker.save(
                chunking_strategy=self.config.chunking_strategy,
                embedding_model=self.embedding_model_id,
                retrieval_mode=self.config.retrieval_mode,
                use_reranker=self.config.use_reranker,
                top_k=top_k or self.config.top_k,
                query=query,
                results=results,
                generation=generation,
                extra={
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "document_count": len(self.documents),
                    "chunk_count": len(self.chunks),
                    "gemini_model": self.config.gemini_model if generate else None,
                },
            )
        return results, generation, saved_path

    def sync_runtime_config(self) -> None:
        """Re-create retriever/reranker from config and re-index existing chunks."""
        if self.config.retrieval_mode != "bm25" and self.embedder is None:
            self.embedder = get_embedder(self.config.embedding_model)
        self.retriever = get_retriever(self.config.retrieval_mode, self.embedder)
        self.reranker = CrossEncoderReranker() if self.config.use_reranker else None
        if self.chunks:
            self.retriever.index(self.chunks)

    def rebuild(self, config: PipelineConfig | None = None) -> int:
        """Apply new config and rebuild the full index."""
        if config is not None:
            self.config = config
            self.__post_init__()
        if not self.documents:
            self.load_documents()
        return self.build_index()


# Backward-compatible alias
RetrievalPipeline = RAGPipeline
