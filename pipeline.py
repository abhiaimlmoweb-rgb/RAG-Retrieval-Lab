"""
End-to-end retrieval and RAG pipeline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from chunkers.base import BaseChunker, Chunk
from chunkers.fixed import FixedChunker
from chunkers.recursive import RecursiveChunker
from chunkers.agent_based import AgentChunker
from chunkers.document_based import DocumentBasedChunker
from chunkers.parent_child import ParentChildChunker
from chunkers.semantic import SemanticChunker
from config.settings import (
    COMPARE_RETRIEVAL_MODES,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CRAWL_MAX_PAGES,
    DEFAULT_CRAG_THRESHOLD,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_OPENAI_CHAT_MODEL,
    DEFAULT_RERANK_POOL,
    DEFAULT_TOP_K,
    EMBEDDING_MODELS,
    resolve_api_key,
)
from embeddings.base import BaseEmbedder
from embeddings.bge_small import BGEEmbedder
from embeddings.cohere_embedder import CohereEmbedder
from embeddings.minilm import MiniLMEmbedder
from embeddings.openai_embedder import OpenAIEmbedder
from evaluators.dashboard import EvaluationDashboard
from evaluators.ragas_metrics import RagasEvaluator, RagasReport
from evaluators.experiment_tracker import ExperimentTracker
from generators.base import GenerationResult
from generators.citation_verifier import CitationVerifier, CitationReport
from generators.crag import (
    RetrievalGrade,
    abstention_answer,
    grade_retrieval,
    merge_retrieval_results,
)
from loaders.base import LoadedDocument
from loaders.dedup import assign_versions, deduplicate_documents
from loaders.document_loader import DocumentLoader
from loaders.web_crawler import WebCrawler
from loaders.web_loader import WebLoader
from cache.redis_cache import RedisQueryCache
from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.bm25_retriever import BM25Retriever
from retrievers.cosine_retriever import CosineRetriever
from retrievers.faiss_retriever import FAISSRetriever
from generators.factory import get_generator
from retrievers.parent_document_retriever import ParentDocumentRetriever
from retrievers.splade_retriever import SpladeRetriever
from retrievers.weighted_hybrid_retriever import WeightedHybridRetriever
from retrievers.qdrant_retriever import QdrantRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.reranker import CrossEncoderReranker
from retrievers.query_expansion import expand_query_hyde, expand_query_multi
from utils.observability import get_logger
from utils.telemetry import trace_span


def get_chunker(
    strategy: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    *,
    gemini_api_key: str | None = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> BaseChunker:
    if strategy == "fixed":
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "semantic":
        return SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "parent_child":
        return ParentChildChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "document_based":
        return DocumentBasedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "agent":
        return AgentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            api_key=gemini_api_key,
            model_name=gemini_model,
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def get_embedder(model_key: str) -> BaseEmbedder:
    registry: dict[str, Callable[[], BaseEmbedder]] = {
        "bge-small": BGEEmbedder,
        "minilm": MiniLMEmbedder,
        "openai": OpenAIEmbedder,
        "cohere": CohereEmbedder,
    }
    if model_key not in registry:
        raise ValueError(f"Unknown embedding model: {model_key}")
    return registry[model_key]()


def get_dense_retriever(
    embedder: BaseEmbedder,
    *,
    index_backend: str = "memory",
    use_cache: bool = True,
) -> BaseRetriever:
    if index_backend == "faiss":
        return FAISSRetriever(embedder, use_cache=use_cache)
    if index_backend == "qdrant":
        return QdrantRetriever(embedder)
    if index_backend == "pinecone":
        from retrievers.pinecone_retriever import PineconeRetriever

        return PineconeRetriever(embedder)
    if index_backend == "memory":
        return CosineRetriever(embedder)
    raise ValueError(f"Unknown index backend: {index_backend}")


def get_retriever(
    mode: str,
    embedder: BaseEmbedder | None = None,
    *,
    index_backend: str = "memory",
    use_cache: bool = True,
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA,
    use_parent_document: bool = False,
) -> BaseRetriever:
    if mode == "splade":
        return SpladeRetriever()
    if mode == "colbert":
        if embedder is None:
            raise ValueError("ColBERT retrieval requires an embedder")
        from retrievers.colbert_retriever import ColBERTRetriever

        inner = ColBERTRetriever(embedder)
        return ParentDocumentRetriever(inner) if use_parent_document else inner
    if mode == "bm25":
        inner = BM25Retriever()
        return ParentDocumentRetriever(inner) if use_parent_document else inner
    if embedder is None:
        raise ValueError(f"{mode} retrieval requires an embedder")
    dense = get_dense_retriever(embedder, index_backend=index_backend, use_cache=use_cache)
    if mode == "dense":
        inner = dense
    elif mode == "hybrid":
        inner = HybridRetriever(dense)
    elif mode == "weighted_hybrid":
        inner = WeightedHybridRetriever(dense, alpha=hybrid_alpha)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    return ParentDocumentRetriever(inner) if use_parent_document else inner


@dataclass
class PipelineConfig:
    chunking_strategy: str = "recursive"
    embedding_model: str = "bge-small"
    retrieval_mode: str = "hybrid"
    index_backend: str = "memory"
    query_expansion: str = "none"
    use_embedding_cache: bool = True
    use_redis_cache: bool = True
    use_reranker: bool = False
    verify_citations: bool = True
    deduplicate_docs: bool = True
    use_parent_document: bool = False
    use_crag: bool = False
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA
    crag_threshold: float = DEFAULT_CRAG_THRESHOLD
    rerank_pool: int = DEFAULT_RERANK_POOL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    crawl_max_pages: int = DEFAULT_CRAWL_MAX_PAGES
    data_dir: Path = field(default_factory=lambda: Path("data"))
    experiments_dir: Path = field(default_factory=lambda: Path("experiments"))
    generator_provider: str = "gemini"
    gemini_model: str = DEFAULT_GEMINI_MODEL
    openai_model: str = DEFAULT_OPENAI_CHAT_MODEL
    claude_model: str = DEFAULT_CLAUDE_MODEL
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None


@dataclass
class RAGPipeline:
    config: PipelineConfig
    loader: DocumentLoader = field(init=False)
    web_loader: WebLoader = field(init=False)
    web_crawler: WebCrawler = field(init=False)
    chunker: BaseChunker = field(init=False)
    embedder: BaseEmbedder | None = field(init=False, default=None)
    retriever: BaseRetriever = field(init=False)
    reranker: CrossEncoderReranker | None = field(init=False)
    citation_verifier: CitationVerifier = field(init=False)
    query_cache: RedisQueryCache = field(init=False)
    dashboard: EvaluationDashboard = field(init=False)
    tracker: ExperimentTracker = field(init=False)
    logger: object = field(init=False)

    documents: list[LoadedDocument] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    last_citation_report: CitationReport | None = None
    last_ragas_report: RagasReport | None = None
    last_retrieval_grade: RetrievalGrade | None = None

    def __post_init__(self) -> None:
        self.logger = get_logger("pipeline")
        self.loader = DocumentLoader(self.config.data_dir)
        self.web_loader = WebLoader()
        self.web_crawler = WebCrawler(max_pages=self.config.crawl_max_pages)
        self.chunker = get_chunker(
            self.config.chunking_strategy,
            self.config.chunk_size,
            self.config.chunk_overlap,
            gemini_api_key=self.config.gemini_api_key,
            gemini_model=self.config.gemini_model,
        )
        needs_embedder = self.config.retrieval_mode not in ("bm25", "splade")
        self.embedder = get_embedder(self.config.embedding_model) if needs_embedder else None
        self.retriever = get_retriever(
            self.config.retrieval_mode,
            self.embedder,
            index_backend=self.config.index_backend,
            use_cache=self.config.use_embedding_cache,
            hybrid_alpha=self.config.hybrid_alpha,
            use_parent_document=self.config.use_parent_document,
        )
        self.reranker = CrossEncoderReranker() if self.config.use_reranker else None
        self.citation_verifier = CitationVerifier()
        self.ragas_evaluator = RagasEvaluator(embedder=self.embedder)
        self.query_cache = RedisQueryCache() if self.config.use_redis_cache else RedisQueryCache(url=None)
        self.dashboard = EvaluationDashboard()
        self.tracker = ExperimentTracker(self.config.experiments_dir)

    @property
    def embedding_model_id(self) -> str:
        return EMBEDDING_MODELS.get(self.config.embedding_model, self.config.embedding_model)

    def _config_hash(self) -> str:
        payload = {
            "chunking": self.config.chunking_strategy,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "embed": self.config.embedding_model,
            "retrieval": self.config.retrieval_mode,
            "backend": self.config.index_backend,
            "chunks": len(self.chunks),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _finalize_documents(self, docs: list[LoadedDocument]) -> list[LoadedDocument]:
        docs = assign_versions(docs)
        if self.config.deduplicate_docs:
            docs = deduplicate_documents(docs)
        return docs

    def load_documents(self) -> list[LoadedDocument]:
        self.documents = self._finalize_documents(self.loader.load_all())
        self.logger.info("Loaded %d documents", len(self.documents))
        return self.documents

    def load_documents_from_paths(self, paths: list[Path | str]) -> list[LoadedDocument]:
        self.documents = self._finalize_documents(self.loader.load_from_paths(paths))
        return self.documents

    def load_from_url(self, url: str) -> LoadedDocument:
        doc = self.web_loader.load_url(url)
        self.documents = self._finalize_documents(self.documents + [doc])
        return doc

    def crawl_site(self, start_url: str) -> list[LoadedDocument]:
        crawled = self.web_crawler.crawl(start_url)
        self.documents = self._finalize_documents(self.documents + crawled)
        self.logger.info("Crawled %d pages from %s", len(crawled), start_url)
        return crawled

    def build_index(self) -> int:
        with trace_span("build_index", mode=self.config.retrieval_mode):
            doc_pairs = [(d.document_name, d.text) for d in self.documents]
            self.chunks = self.chunker.chunk_documents(doc_pairs)
            self.retriever.index(self.chunks)
        self.logger.info("Indexed %d chunks", len(self.chunks))
        return len(self.chunks)

    def _expand_query(self, query: str) -> list[str]:
        mode = self.config.query_expansion
        if mode == "multi":
            return expand_query_multi(query)
        if mode == "hyde":
            return [expand_query_hyde(query, self.config.gemini_api_key)]
        return [query]

    def _retrieve_single(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self.reranker is not None:
            pool_k = min(self.config.rerank_pool, len(self.chunks))
            candidates = self.retriever.retrieve(query, top_k=pool_k)
            return self.reranker.rerank(query, candidates, top_k=top_k)
        return self.retriever.retrieve(query, top_k=top_k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        k = top_k if top_k is not None else self.config.top_k

        cached = self.query_cache.get(query, self._config_hash())
        if cached:
            self.logger.info("Redis cache hit for query")
            return self._results_from_cache(query, cached)

        with trace_span("retrieve", mode=self.config.retrieval_mode):
            queries = self._expand_query(query)
            if len(queries) == 1:
                results = self._retrieve_single(queries[0], k)
            else:
                merged: dict[tuple[str, int], RetrievalResult] = {}
                for q in queries:
                    for r in self._retrieve_single(q, k):
                        key = (r.chunk.document_name, r.chunk.chunk_id)
                        if key not in merged or r.similarity_score > merged[key].similarity_score:
                            merged[key] = r
                results = sorted(merged.values(), key=lambda x: x.similarity_score, reverse=True)[:k]
                for i, r in enumerate(results, start=1):
                    results[i - 1] = RetrievalResult(
                        query=query,
                        chunk=r.chunk,
                        similarity_score=r.similarity_score,
                        rank=i,
                        latency_ms=r.latency_ms,
                        source_document=r.source_document,
                        retrieval_method=r.retrieval_method,
                    )

        self.query_cache.set(query, self._config_hash(), [r.to_dict() for r in results])
        return results

    def retrieve_with_crag(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Corrective RAG: grade retrieval, retry with multi-query expansion, or abstain."""
        k = top_k if top_k is not None else self.config.top_k
        results = self.retrieve(query, top_k=k)
        grade = grade_retrieval(results, threshold=self.config.crag_threshold)
        self.last_retrieval_grade = grade
        if grade.passed or not self.config.use_crag:
            return results

        expanded = expand_query_multi(query)
        batches = [self._retrieve_single(q, k) for q in expanded if q.strip()]
        corrected = merge_retrieval_results(query, batches, top_k=k)
        retry_grade = grade_retrieval(corrected, threshold=self.config.crag_threshold)
        self.last_retrieval_grade = retry_grade
        return corrected if retry_grade.passed else results

    def compare_retrieval(
        self,
        query: str,
        *,
        modes: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Run the same query across multiple retrieval modes for side-by-side learning."""
        if not self.chunks:
            raise RuntimeError("Index not built. Call build_index() first.")
        k = top_k if top_k is not None else self.config.top_k
        modes = modes or list(COMPARE_RETRIEVAL_MODES)
        comparison: dict[str, list[RetrievalResult]] = {}

        for mode in modes:
            retriever = get_retriever(
                mode,
                self.embedder,
                index_backend=self.config.index_backend,
                use_cache=self.config.use_embedding_cache,
                hybrid_alpha=self.config.hybrid_alpha,
                use_parent_document=self.config.use_parent_document,
            )
            retriever.index(self.chunks)
            comparison[mode] = retriever.retrieve(query, top_k=k)
        return comparison

    def _results_from_cache(self, query: str, cached: list[dict]) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        chunk_map = {(c.document_name, c.chunk_id): c for c in self.chunks}
        for item in cached:
            chunk = chunk_map.get((item["document_name"], item["chunk_id"]))
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    similarity_score=item["similarity_score"],
                    rank=item["rank"],
                    latency_ms=item["latency_ms"],
                    source_document=item["source_document"],
                    retrieval_method=item.get("retrieval_method", "cached"),
                )
            )
        return results

    def generate_answer(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> GenerationResult:
        gen_provider = provider or self.config.generator_provider
        if model is None:
            model = {
                "gemini": self.config.gemini_model,
                "openai": self.config.openai_model,
                "claude": self.config.claude_model,
            }.get(gen_provider, self.config.gemini_model)

        key = resolve_api_key(
            gen_provider,
            api_key
            or {
                "gemini": self.config.gemini_api_key,
                "openai": self.config.openai_api_key,
                "claude": self.config.anthropic_api_key,
            }.get(gen_provider),
        )
        if not key:
            raise ValueError(
                f"API key required for {gen_provider}. "
                f"Enter it in the sidebar or set the provider key in .env at project root."
            )

        with trace_span("generate", provider=gen_provider):
            generator = get_generator(gen_provider, api_key=key, model_name=model)
            generation = generator.generate(query, results)

        if self.config.verify_citations:
            self.last_citation_report = self.citation_verifier.verify(generation, results)
        self.last_ragas_report = self.ragas_evaluator.evaluate(query, results, generation)
        return generation

    def run_query(
        self,
        query: str,
        *,
        save_experiment: bool = True,
        top_k: int | None = None,
        generate: bool = False,
        generator_provider: str | None = None,
        api_key: str | None = None,
        use_crag: bool | None = None,
    ) -> tuple[list[RetrievalResult], GenerationResult | None, Path | None]:
        crag_enabled = self.config.use_crag if use_crag is None else use_crag
        if crag_enabled:
            results = self.retrieve_with_crag(query, top_k=top_k)
        else:
            results = self.retrieve(query, top_k=top_k)
        generation: GenerationResult | None = None

        if generate:
            if crag_enabled and self.last_retrieval_grade and not self.last_retrieval_grade.passed:
                generation = abstention_answer(query, self.last_retrieval_grade)
            else:
                generation = self.generate_answer(
                    query, results, provider=generator_provider, api_key=api_key
                )

        extra = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "index_backend": self.config.index_backend,
            "query_expansion": self.config.query_expansion,
            "generator_provider": generator_provider or self.config.generator_provider,
            "use_crag": crag_enabled,
            "hybrid_alpha": self.config.hybrid_alpha,
        }
        if self.last_citation_report is not None:
            extra["citation_report"] = self.last_citation_report.to_dict()
        if self.last_ragas_report is not None:
            extra["ragas_report"] = self.last_ragas_report.to_dict()
        if self.last_retrieval_grade is not None:
            extra["retrieval_grade"] = self.last_retrieval_grade.to_dict()

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
                extra=extra,
            )
        return results, generation, saved_path

    def sync_runtime_config(self) -> None:
        self.chunker = get_chunker(
            self.config.chunking_strategy,
            self.config.chunk_size,
            self.config.chunk_overlap,
            gemini_api_key=self.config.gemini_api_key,
            gemini_model=self.config.gemini_model,
        )
        if self.config.retrieval_mode not in ("bm25", "splade") and self.embedder is None:
            self.embedder = get_embedder(self.config.embedding_model)
        self.ragas_evaluator = RagasEvaluator(embedder=self.embedder)
        self.retriever = get_retriever(
            self.config.retrieval_mode,
            self.embedder,
            index_backend=self.config.index_backend,
            use_cache=self.config.use_embedding_cache,
            hybrid_alpha=self.config.hybrid_alpha,
            use_parent_document=self.config.use_parent_document,
        )
        self.reranker = CrossEncoderReranker() if self.config.use_reranker else None
        if self.chunks:
            self.retriever.index(self.chunks)

    def rebuild(self, config: PipelineConfig | None = None) -> int:
        if config is not None:
            self.config = config
            self.__post_init__()
        if not self.documents:
            self.load_documents()
        return self.build_index()


RetrievalPipeline = RAGPipeline
