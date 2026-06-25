"""
RAG Retrieval Lab — Streamlit UI.

Full workbench: document ingestion, chunking, embeddings, dense/BM25/hybrid
retrieval, cross-encoder reranking, and Gemini-powered answer generation.
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
    DATA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_MODELS,
    EXPERIMENTS_DIR,
    GEMINI_API_KEY_ENV,
    GEMINI_MODELS,
    RETRIEVAL_MODES,
)
from evaluators.dashboard import EvaluationDashboard  # noqa: E402
from evaluators.experiment_tracker import ExperimentTracker  # noqa: E402
from pipeline import PipelineConfig, RAGPipeline  # noqa: E402

st.set_page_config(
    page_title="RAG Retrieval Lab",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 RAG Retrieval Lab")
st.caption(
    "End-to-end RAG workbench: PDF/TXT/MD ingestion → chunking → embeddings → "
    "dense / BM25 / hybrid retrieval → optional reranking → Gemini generation."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Settings")

    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        options=["recursive", "fixed"],
        help="Recursive respects paragraphs/sentences; fixed uses character windows.",
    )

    embedding_key = st.selectbox(
        "Embedding Model",
        options=list(EMBEDDING_MODELS.keys()),
        format_func=lambda k: EMBEDDING_MODELS[k],
    )

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=list(RETRIEVAL_MODES.keys()),
        format_func=lambda k: RETRIEVAL_MODES[k],
        index=list(RETRIEVAL_MODES.keys()).index("hybrid"),
    )

    use_reranker = st.checkbox(
        "Cross-encoder reranking",
        value=False,
        help="Retrieve more candidates, then rerank with ms-marco cross-encoder.",
    )

    chunk_size = st.slider("Chunk Size (chars)", 128, 2048, DEFAULT_CHUNK_SIZE, 64)
    chunk_overlap = st.slider(
        "Chunk Overlap (chars)", 0, min(512, chunk_size - 1), DEFAULT_CHUNK_OVERLAP, 16
    )
    top_k = st.slider("Top K", 1, 20, DEFAULT_TOP_K)

    st.divider()
    st.subheader("Gemini (Generation)")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv(GEMINI_API_KEY_ENV, ""),
        type="password",
        help=f"Or set {GEMINI_API_KEY_ENV} in a .env file at project root.",
    )
    gemini_model = st.selectbox(
        "Gemini Model",
        options=list(GEMINI_MODELS.keys()),
        format_func=lambda k: GEMINI_MODELS[k],
    )

    st.divider()
    save_experiments = st.checkbox("Save experiment to JSON", value=True)

# ── Documents ─────────────────────────────────────────────────────────────────
st.subheader("1. Documents")
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="Supports PDF (PyMuPDF), plain text, and Markdown.",
)

col_a, col_b = st.columns(2)
with col_a:
    use_existing = st.checkbox("Also load files from data folder", value=True)
with col_b:
    build_btn = st.button("Build / Rebuild Index", type="primary", use_container_width=True)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.index_built = False

if build_btn:
    config = PipelineConfig(
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_key,
        retrieval_mode=retrieval_mode,
        use_reranker=use_reranker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        data_dir=DATA_DIR,
        experiments_dir=EXPERIMENTS_DIR,
        gemini_model=gemini_model,
        gemini_api_key=gemini_api_key or None,
    )
    pipeline = RAGPipeline(config)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    if uploaded_files:
        for uf in uploaded_files:
            dest = DATA_DIR / uf.name
            dest.write_bytes(uf.getbuffer())
            saved_paths.append(dest)
        st.success(f"Saved {len(uploaded_files)} uploaded file(s) to data/")

    if use_existing or saved_paths:
        if saved_paths and not use_existing:
            pipeline.load_documents_from_paths(saved_paths)
        else:
            pipeline.load_documents()

        if not pipeline.documents:
            st.warning("No documents found. Upload files or add them to data/.")
        else:
            with st.spinner("Chunking and indexing (first run downloads models)..."):
                n_chunks = pipeline.build_index()

            st.session_state.pipeline = pipeline
            st.session_state.index_built = True
            st.session_state.config_snapshot = {
                "chunking_strategy": chunking_strategy,
                "embedding_model": EMBEDDING_MODELS[embedding_key],
                "retrieval_mode": retrieval_mode,
                "use_reranker": use_reranker,
            }

            doc_names = [d.document_name for d in pipeline.documents]
            st.success(
                f"Indexed **{n_chunks}** chunks from **{len(doc_names)}** document(s): "
                + ", ".join(f"`{n}`" for n in doc_names)
            )
    else:
        st.warning("Enable 'load from data folder' or upload at least one file.")

if st.session_state.index_built and st.session_state.pipeline:
    p: RAGPipeline = st.session_state.pipeline
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Documents", len(p.documents))
    m2.metric("Chunks", len(p.chunks))
    m3.metric("Embed Dim", p.embedder.dimension if p.embedder else "—")
    m4.metric("Retrieval", p.config.retrieval_mode)
    m5.metric("Rerank", "on" if p.config.use_reranker else "off")

# ── Query ─────────────────────────────────────────────────────────────────────
st.subheader("2. Query")
query = st.text_input(
    "Enter your question",
    placeholder="What is retrieval-augmented generation?",
)

col_retrieve, col_generate = st.columns(2)
with col_retrieve:
    retrieve_btn = st.button(
        "Retrieve Only",
        type="secondary",
        disabled=not st.session_state.index_built,
        use_container_width=True,
    )
with col_generate:
    generate_btn = st.button(
        "Retrieve + Generate (Gemini)",
        type="primary",
        disabled=not st.session_state.index_built,
        use_container_width=True,
    )

def _run_query(*, generate: bool) -> None:
    if not query.strip():
        st.error("Please enter a query.")
        return

    pipeline: RAGPipeline = st.session_state.pipeline
    pipeline.config.top_k = top_k
    pipeline.config.retrieval_mode = retrieval_mode
    pipeline.config.use_reranker = use_reranker
    pipeline.config.gemini_model = gemini_model
    pipeline.config.gemini_api_key = gemini_api_key or None
    pipeline.sync_runtime_config()

    if generate and not (gemini_api_key or os.getenv(GEMINI_API_KEY_ENV)):
        st.error(f"Enter a Gemini API key or set {GEMINI_API_KEY_ENV} in .env")
        return

    label = "Generating answer..." if generate else "Retrieving..."
    with st.spinner(label):
        results, generation, exp_path = pipeline.run_query(
            query.strip(),
            save_experiment=save_experiments,
            top_k=top_k,
            generate=generate,
            gemini_api_key=gemini_api_key or None,
        )

    dashboard = EvaluationDashboard()
    summary = dashboard.summary(results)

    st.subheader("3. Retrieval Results")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Top Score", summary["top_score"])
    s2.metric("Avg Score", summary["avg_score"])
    s3.metric("Latency (ms)", summary["latency_ms"])
    s4.metric("Method", results[0].retrieval_method if results else "—")

    df = dashboard.to_dataframe(results)
    st.dataframe(
        df[
            [
                "Rank",
                "Similarity Score",
                "Retrieval Method",
                "Source Document",
                "Chunk ID",
                "Retrieval Latency (ms)",
                "Retrieved Chunk",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    if generation is not None:
        st.subheader("4. Generated Answer")
        g1, g2 = st.columns(2)
        g1.metric("Gemini Model", generation.model)
        g2.metric("Generation Latency (ms)", round(generation.latency_ms, 0))
        st.markdown(generation.answer)

    if exp_path:
        st.info(f"Experiment saved to `{exp_path.name}`")

    st.subheader("5. Chunk Details")
    for r in results:
        with st.expander(
            f"Rank {r.rank} · {r.source_document} · {r.retrieval_method} · {r.similarity_score:.4f}"
        ):
            st.markdown(f"**Source:** {r.source_document} (chunk {r.chunk.chunk_id})")
            st.text(r.chunk.text)


if retrieve_btn:
    _run_query(generate=False)
elif generate_btn:
    _run_query(generate=True)

# ── Experiment history ────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent Experiments")
tracker = ExperimentTracker(EXPERIMENTS_DIR)
experiments = tracker.list_experiments(limit=10)
if experiments:
    st.dataframe(experiments, use_container_width=True, hide_index=True)
else:
    st.caption("No experiments saved yet.")

# ── Reference ─────────────────────────────────────────────────────────────────
with st.expander("📚 RAG stack reference"):
    st.markdown(
        """
        | Layer | Technology | Role |
        |-------|------------|------|
        | **Ingestion** | PyMuPDF, UTF-8 text | PDF, TXT, MD → plain text |
        | **Chunking** | Fixed / Recursive | Split long docs into searchable units |
        | **Dense retrieval** | BGE-small / MiniLM + NumPy | Semantic similarity (cosine) |
        | **Sparse retrieval** | BM25 (rank-bm25) | Keyword / exact-term matching |
        | **Hybrid** | RRF fusion | Combine dense + BM25 rankings |
        | **Reranking** | Cross-encoder (ms-marco) | Re-score top candidates jointly |
        | **Generation** | Google Gemini | Grounded answer from retrieved context |
        | **Tracking** | JSON experiment logs | Compare runs over time |

        **Production upgrades:** Qdrant/FAISS vector DB, ColBERT late interaction,
        query expansion, citation verification, nDCG/MRR eval harness.
        """
    )
