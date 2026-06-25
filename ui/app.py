"""
RAG Retrieval Lab — Streamlit UI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    CHUNKING_STRATEGIES,
    CLAUDE_MODELS,
    COMPARE_RETRIEVAL_MODES,
    DATA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CRAWL_MAX_PAGES,
    DEFAULT_CRAG_THRESHOLD,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_TOP_K,
    EMBEDDING_MODELS,
    EXPERIMENTS_DIR,
    GEMINI_API_KEY_ENV,
    GEMINI_MODELS,
    GENERATOR_PROVIDERS,
    INDEX_BACKENDS,
    OPENAI_API_KEY_ENV,
    OPENAI_CHAT_MODELS,
    QUERY_EXPANSION_MODES,
    RETRIEVAL_MODES,
    ANTHROPIC_API_KEY_ENV,
    resolve_api_key,
)
from evaluators.dashboard import EvaluationDashboard  # noqa: E402
from evaluators.experiment_tracker import ExperimentTracker  # noqa: E402
from evaluators.human_rubrics import HumanRubricStore  # noqa: E402
from evaluators.llm_judge import LLMJudge  # noqa: E402
from evaluators.retrieval_eval import EvalDataset, RetrievalEvaluator  # noqa: E402
from pipeline import PipelineConfig, RAGPipeline  # noqa: E402
from utils.auth import require_auth  # noqa: E402

st.set_page_config(page_title="RAG Retrieval Lab", page_icon="🔬", layout="wide")

if not require_auth():
    st.stop()

st.title("🔬 RAG Retrieval Lab")
st.caption("Full production-style RAG: Qdrant/Pinecone, ColBERT, semantic chunking, eval harness, Claude/Gemini/OpenAI.")

with st.sidebar:
    st.header("Pipeline")
    chunking_strategy = st.selectbox(
        "Chunking",
        list(CHUNKING_STRATEGIES.keys()),
        format_func=lambda k: CHUNKING_STRATEGIES[k],
    )
    embedding_key = st.selectbox("Embedding", list(EMBEDDING_MODELS.keys()), format_func=lambda k: EMBEDDING_MODELS[k])
    retrieval_mode = st.selectbox("Retrieval", list(RETRIEVAL_MODES.keys()), format_func=lambda k: RETRIEVAL_MODES[k], index=2)
    hybrid_alpha = st.slider("Hybrid α (dense weight)", 0.0, 1.0, DEFAULT_HYBRID_ALPHA, 0.05, disabled=retrieval_mode != "weighted_hybrid")
    use_parent_doc = st.checkbox("Parent-document retrieval", False)
    use_crag = st.checkbox("Corrective RAG (CRAG)", False)
    crag_threshold = st.slider("CRAG threshold", 0.1, 0.9, DEFAULT_CRAG_THRESHOLD, 0.05, disabled=not use_crag)
    index_backend = st.selectbox("Index Backend", list(INDEX_BACKENDS.keys()), format_func=lambda k: INDEX_BACKENDS[k])
    query_expansion = st.selectbox("Query Expansion", list(QUERY_EXPANSION_MODES.keys()), format_func=lambda k: QUERY_EXPANSION_MODES[k])
    use_reranker = st.checkbox("Reranking", False)
    use_cache = st.checkbox("Embedding cache", True)
    use_redis = st.checkbox("Redis query cache", False)
    verify_citations = st.checkbox("Citation verification", True)
    dedup = st.checkbox("Dedup documents", True)
    chunk_size = st.slider("Chunk size", 128, 2048, DEFAULT_CHUNK_SIZE, 64)
    chunk_overlap = st.slider("Overlap", 0, min(512, chunk_size - 1), DEFAULT_CHUNK_OVERLAP, 16)
    top_k = st.slider("Top K", 1, 20, DEFAULT_TOP_K)

    st.divider()
    st.subheader("Generation")
    generator_provider = st.selectbox("Provider", list(GENERATOR_PROVIDERS.keys()), format_func=lambda k: GENERATOR_PROVIDERS[k])
    gemini_key = st.text_input("Gemini key", os.getenv(GEMINI_API_KEY_ENV, ""), type="password")
    if chunking_strategy == "agent" and not (gemini_key or os.getenv(GEMINI_API_KEY_ENV)):
        st.caption("Agent chunking needs a Gemini API key (sidebar or `.env`). Falls back to recursive.")
    openai_key = st.text_input("OpenAI key", os.getenv(OPENAI_API_KEY_ENV, ""), type="password")
    anthropic_key = st.text_input("Anthropic key", os.getenv(ANTHROPIC_API_KEY_ENV, ""), type="password")

    if generator_provider == "gemini":
        gen_model = st.selectbox("Model", list(GEMINI_MODELS.keys()), format_func=lambda k: GEMINI_MODELS[k])
    elif generator_provider == "openai":
        gen_model = st.selectbox("Model", list(OPENAI_CHAT_MODELS.keys()), format_func=lambda k: OPENAI_CHAT_MODELS[k])
    else:
        gen_model = st.selectbox("Model", list(CLAUDE_MODELS.keys()), format_func=lambda k: CLAUDE_MODELS[k])

    save_experiments = st.checkbox("Save experiments", True)

tab_query, tab_compare, tab_eval, tab_human = st.tabs(["Query", "Compare Methods", "Batch Eval", "Human Rubrics"])

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.index_built = False
    st.session_state.last_answer = ""


def _config() -> PipelineConfig:
    return PipelineConfig(
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_key,
        retrieval_mode=retrieval_mode,
        index_backend=index_backend,
        query_expansion=query_expansion,
        use_embedding_cache=use_cache,
        use_redis_cache=use_redis,
        use_reranker=use_reranker,
        verify_citations=verify_citations,
        deduplicate_docs=dedup,
        use_parent_document=use_parent_doc,
        use_crag=use_crag,
        hybrid_alpha=hybrid_alpha,
        crag_threshold=crag_threshold,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        crawl_max_pages=DEFAULT_CRAWL_MAX_PAGES,
        data_dir=DATA_DIR,
        experiments_dir=EXPERIMENTS_DIR,
        generator_provider=generator_provider,
        gemini_model=gen_model if generator_provider == "gemini" else "gemini-2.0-flash",
        openai_model=gen_model if generator_provider == "openai" else "gpt-4o-mini",
        claude_model=gen_model if generator_provider == "claude" else "claude-3-5-haiku-latest",
        gemini_api_key=gemini_key or None,
        openai_api_key=openai_key or None,
        anthropic_api_key=anthropic_key or None,
    )


def _show_results(results, generation, exp_path, pipeline: RAGPipeline, query_text: str) -> None:
    dash = EvaluationDashboard()
    summary = dash.summary(results)
    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top", summary["top_score"])
    c2.metric("Avg", summary["avg_score"])
    c3.metric("ms", summary["latency_ms"])
    c4.metric("Method", results[0].retrieval_method if results else "—")
    st.dataframe(dash.to_dataframe(results)[["Rank", "Similarity Score", "Retrieval Method", "Source Document", "Retrieved Chunk"]], use_container_width=True)
    if generation:
        st.subheader("Answer")
        st.markdown(generation.answer)
        st.session_state.last_answer = generation.answer
        if pipeline.last_citation_report:
            st.json(pipeline.last_citation_report.to_dict())
        if pipeline.last_ragas_report:
            st.subheader("RAGAS-style metrics")
            st.json(pipeline.last_ragas_report.to_dict())
        if pipeline.last_retrieval_grade:
            st.caption(f"CRAG grade: {pipeline.last_retrieval_grade.to_dict()}")
        if gemini_key or os.getenv(GEMINI_API_KEY_ENV):
            try:
                judge = LLMJudge(api_key=gemini_key or None)
                st.subheader("LLM Judge")
                st.json(judge.score(query_text, results, generation).to_dict())
            except Exception as exc:
                st.caption(f"LLM judge skipped: {exc}")
    if exp_path:
        st.info(f"Saved `{exp_path.name}`")


with tab_query:
    uploaded = st.file_uploader("Upload", type=["pdf", "txt", "md", "html", "htm", "docx"], accept_multiple_files=True)
    web_url = st.text_input("Single URL")
    crawl_url = st.text_input("Crawl site (same host, max pages)")
    use_data = st.checkbox("Load data/", True)
    b1, b2, b3 = st.columns(3)
    build = b1.button("Build Index", type="primary", use_container_width=True)
    fetch = b2.button("Fetch URL", use_container_width=True)
    crawl = b3.button("Crawl Site", use_container_width=True)

    if build or fetch or crawl:
        p = RAGPipeline(_config())
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if uploaded:
            for uf in uploaded:
                (DATA_DIR / uf.name).write_bytes(uf.getbuffer())
        if fetch and web_url.strip():
            try:
                p.load_from_url(web_url.strip())
            except Exception as e:
                st.error(str(e))
        if crawl and crawl_url.strip():
            try:
                p.crawl_site(crawl_url.strip())
            except Exception as e:
                st.error(str(e))
        if use_data:
            for d in p.loader.load_all():
                if d.document_name not in {x.document_name for x in p.documents}:
                    p.documents.append(d)
        p.documents = p._finalize_documents(p.documents)
        if p.documents:
            with st.spinner("Indexing..."):
                n = p.build_index()
            st.session_state.pipeline = p
            st.session_state.index_built = True
            st.success(f"Indexed {n} chunks from {len(p.documents)} docs")
        else:
            st.warning("No documents")

    if st.session_state.index_built and st.session_state.pipeline:
        p = st.session_state.pipeline
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Docs", len(p.documents))
        m2.metric("Chunks", len(p.chunks))
        m3.metric("Backend", p.config.index_backend)
        m4.metric("Mode", p.config.retrieval_mode)

    query = st.text_input("Question")
    r1, r2 = st.columns(2)
    btn_r = r1.button("Retrieve", disabled=not st.session_state.index_built, use_container_width=True)
    btn_g = r2.button("Retrieve + Generate", type="primary", disabled=not st.session_state.index_built, use_container_width=True)
    if (btn_r or btn_g) and query.strip():
        p: RAGPipeline = st.session_state.pipeline
        p.config = _config()
        p.sync_runtime_config()
        if btn_g:
            api_key = resolve_api_key(
                generator_provider,
                {"gemini": gemini_key, "openai": openai_key, "claude": anthropic_key}.get(
                    generator_provider
                ),
            )
            if not api_key:
                st.error(
                    f"API key required for **{generator_provider}**. "
                    f"Enter it in the sidebar or add it to `.env` "
                    f"({GEMINI_API_KEY_ENV}, {OPENAI_API_KEY_ENV}, or {ANTHROPIC_API_KEY_ENV})."
                )
            else:
                with st.spinner("Generating..."):
                    res, gen, path = p.run_query(
                        query.strip(),
                        save_experiment=save_experiments,
                        top_k=top_k,
                        generate=True,
                        generator_provider=generator_provider,
                        api_key=api_key,
                    )
                _show_results(res, gen, path, p, query.strip())
        else:
            with st.spinner("Retrieving..."):
                res, _, path = p.run_query(query.strip(), save_experiment=save_experiments, top_k=top_k)
            _show_results(res, None, path, p, query.strip())

with tab_compare:
    st.caption("Run the same query across retrieval methods to learn tradeoffs.")
    compare_query = st.text_input("Compare question", key="compare_q")
    compare_modes = st.multiselect(
        "Methods",
        COMPARE_RETRIEVAL_MODES,
        default=["dense", "bm25", "hybrid", "weighted_hybrid"],
        format_func=lambda m: RETRIEVAL_MODES.get(m, m),
    )
    if st.button("Compare retrieval", disabled=not st.session_state.index_built, type="primary"):
        if compare_query.strip() and compare_modes:
            p: RAGPipeline = st.session_state.pipeline
            p.config = _config()
            with st.spinner("Comparing retrieval methods..."):
                comparison = p.compare_retrieval(compare_query.strip(), modes=compare_modes, top_k=top_k)
            rows = []
            for mode, hits in comparison.items():
                for h in hits:
                    rows.append(
                        {
                            "Method": mode,
                            "Rank": h.rank,
                            "Score": round(h.similarity_score, 4),
                            "Source": h.source_document,
                            "Preview": h.chunk.text[:120] + ("…" if len(h.chunk.text) > 120 else ""),
                        }
                    )
            st.dataframe(rows, use_container_width=True)
            summary = {
                mode: {
                    "top_score": round(hits[0].similarity_score, 4) if hits else 0,
                    "latency_ms": round(hits[0].latency_ms, 1) if hits else 0,
                }
                for mode, hits in comparison.items()
            }
            st.json(summary)

with tab_eval:
    st.caption("Labeled queries in data/eval_qa.json")
    if st.button("Run batch eval", disabled=not st.session_state.index_built):
        p = st.session_state.pipeline
        p.config = _config()
        p.sync_runtime_config()
        ev = RetrievalEvaluator(p)
        ds = EvalDataset.load(DATA_DIR / "eval_qa.json")
        per = ev.run(ds, k=top_k)
        st.dataframe(ev.to_dataframe(per), use_container_width=True)
        st.json(ev.summarize(per))

with tab_human:
    st.caption("Manual 1–5 scores for faithfulness and relevance")
    hq = st.text_input("Query (rubric)", value="")
    ha = st.text_area("Answer (rubric)", value=st.session_state.get("last_answer", ""))
    hf = st.slider("Faithfulness", 1, 5, 3)
    hr = st.slider("Relevance", 1, 5, 3)
    notes = st.text_input("Notes")
    if st.button("Save rubric score"):
        if hq.strip() and ha.strip():
            HumanRubricStore().save(query=hq, answer=ha, faithfulness=hf, relevance=hr, notes=notes)
            st.success("Saved")
    recent = HumanRubricStore().list_recent()
    if recent:
        st.dataframe(recent, use_container_width=True)

st.divider()
tracker = ExperimentTracker(EXPERIMENTS_DIR)
ex = tracker.list_experiments(10)
if ex:
    st.dataframe(ex, use_container_width=True)
