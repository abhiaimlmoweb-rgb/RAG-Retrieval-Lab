# RAG Retrieval Lab

An end-to-end experimentation platform for **Retrieval-Augmented Generation (RAG)** — from document ingestion through hybrid retrieval, reranking, and **Gemini-powered answer generation**.

Built for learning and comparison: swap chunking strategies, embedding models, retrieval backends, and inspect every stage before moving to production vector DBs.

---

## What’s included

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Ingestion** | PyMuPDF, UTF-8 text loader | PDF, `.txt`, `.md` → plain text |
| **Chunking** | Fixed & recursive splitters | Tune size/overlap; preserve structure |
| **Dense retrieval** | BGE-small, MiniLM + NumPy cosine | Semantic similarity search |
| **Sparse retrieval** | BM25 (`rank-bm25`) | Keyword / exact-term matching |
| **Hybrid** | Reciprocal Rank Fusion (RRF) | Merge dense + BM25 rankings |
| **Reranking** | Cross-encoder (`ms-marco-MiniLM`) | Re-score top candidates jointly |
| **Generation** | Google Gemini API | Grounded answers from retrieved context |
| **Evaluation** | Dashboard + JSON experiment logs | Compare runs over time |
| **Batch eval** | nDCG@K, MRR, recall@K | Labeled `data/eval_qa.json` harness |
| **Index backends** | NumPy memory or FAISS | Swap via `--index-backend` |
| **Caching** | Disk embedding cache | Faster rebuilds (`cache/embeddings/`) |
| **Observability** | Structured file logging | `logs/pipeline.log` |
| **UI** | Streamlit | Interactive workbench |

### Pipeline

```
PDF / TXT / MD
    → Loader → Chunker → Embed + Index
                              ↓
         Query → Dense / BM25 / Hybrid → [Rerank] → Top-K chunks
                              ↓
                    [Gemini] → Grounded answer
                              ↓
                    Experiment JSON log
```

---

## Project structure

```
rag-retrieval-lab/
├── config/settings.py           # Paths, models, env vars
├── loaders/
│   ├── pdf_loader.py            # PyMuPDF PDF ingestion
│   ├── text_loader.py           # TXT / Markdown
│   ├── html_loader.py           # HTML files
│   ├── docx_loader.py           # Word documents
│   ├── web_loader.py              # URL fetch + extract
│   └── document_loader.py       # Unified loader
├── chunkers/
│   ├── fixed.py                 # Fixed character windows
│   └── recursive.py             # Paragraph/sentence-aware splits
├── embeddings/
│   ├── bge_small.py             # BAAI/bge-small-en-v1.5
│   └── minilm.py                # all-MiniLM-L6-v2
├── retrievers/
│   ├── cosine_retriever.py      # Dense cosine search
│   ├── faiss_retriever.py       # FAISS vector index
│   ├── bm25_retriever.py        # Lexical BM25
│   ├── hybrid_retriever.py      # RRF fusion
│   └── reranker.py              # Cross-encoder reranking
├── generators/
│   ├── gemini_generator.py      # Google Gemini
│   ├── openai_generator.py      # OpenAI Chat
│   └── factory.py                 # Provider factory
├── cache/
│   └── embedding_cache.py       # Disk embedding cache
├── utils/
│   └── observability.py         # Pipeline logging
├── evaluators/
│   ├── metrics.py               # nDCG, MRR, recall@K
│   └── retrieval_eval.py        # Batch eval harness
├── eval.py                      # Batch eval CLI
├── ui/app.py                    # Streamlit workbench
├── pipeline.py                  # Orchestration
├── main.py                      # CLI
├── data/                        # Sample PDF + notes
├── experiments/                 # Saved runs
├── .env.example                 # API key template
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- ~2–3 GB disk for embedding / reranker model caches (first run downloads from Hugging Face)
- [Google AI Studio API key](https://aistudio.google.com/apikey) for generation (retrieval works without it)

### 1. Virtual environment

```bash
cd rag-retrieval-lab
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> On Apple Silicon, install PyTorch first from [pytorch.org](https://pytorch.org) if `pip install torch` fails.

### 2. Gemini API key

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash   # optional
```

Or export in your shell:

```bash
export GEMINI_API_KEY=your_key_here
```

### 3. Sample data

Included in `data/`:

- `sample_rag_guide.pdf`
- `rag_lab_notes.md`

Add your own PDFs, `.txt`, or `.md` files to `data/`.

---

## Run

### Streamlit UI (recommended)

```bash
streamlit run ui/app.py
```

Workflow:

1. Upload documents or use files in `data/`
2. Configure chunking, embeddings, retrieval mode (dense / BM25 / hybrid), reranking
3. **Build / Rebuild Index**
4. Enter a query → **Retrieve Only** or **Retrieve + Generate (Gemini)**
5. Inspect chunks, scores, generated answer, and saved experiments

### CLI

**Retrieval only (hybrid):**

```bash
python main.py --query "What is retrieval-augmented generation?" \
  --retrieval hybrid \
  --strategy recursive \
  --model bge-small \
  --top-k 5
```

**Full RAG with reranking + Gemini:**

```bash
python main.py --query "How does hybrid search work?" \
  --retrieval hybrid \
  --rerank \
  --generate \
  --gemini-model gemini-2.0-flash
```

### Batch evaluation

```bash
python eval.py --dataset data/eval_qa.json --retrieval hybrid --top-k 5
python eval.py --index-backend faiss --rerank
```

### CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `--query` | Search question | (required) |
| `--strategy` | `fixed` or `recursive` | `recursive` |
| `--model` | `bge-small` or `minilm` | `bge-small` |
| `--retrieval` | `dense`, `bm25`, or `hybrid` | `hybrid` |
| `--index-backend` | `memory` or `faiss` | `memory` |
| `--rerank` | Cross-encoder reranking | off |
| `--generate` | LLM answer generation | off |
| `--generator` | `gemini` or `openai` | `gemini` |
| `--gemini-model` | Gemini model name | `gemini-2.0-flash` |
| `--chunk-size` | Characters per chunk | `512` |
| `--chunk-overlap` | Overlap between chunks | `64` |
| `--top-k` | Number of results | `5` |
| `--data-dir` | Document folder | `data/` |
| `--no-save` | Skip experiment JSON | off |

---

## Retrieval modes explained

### Dense (cosine)

Embeds query and chunks with BGE or MiniLM; ranks by cosine similarity. Best for semantic questions and paraphrases.

### BM25 (lexical)

Classic keyword search. Best for exact terms, product codes, and rare strings. No embedding model needed at query time.

### Hybrid (RRF)

Runs dense and BM25, merges with **Reciprocal Rank Fusion** — robust when you need both semantics and keywords. Recommended default.

### Reranking

Retrieves a larger candidate pool (default 15), then a **cross-encoder** re-scores `(query, chunk)` pairs. Slower but more accurate for the final top-K.

---

## Generation (Gemini)

The generator:

1. Formats retrieved chunks as context with source labels
2. Sends a system prompt instructing grounded, cited answers
3. Returns the model response with latency metadata

Requires `GEMINI_API_KEY`. Retrieval and reranking work without an API key.

---

## Experiment JSON schema

```json
{
  "experiment_id": "uuid",
  "timestamp": "ISO-8601",
  "chunking_strategy": "recursive",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "retrieval_mode": "hybrid",
  "use_reranker": true,
  "top_k": 5,
  "query": "...",
  "retrieved_chunks": [...],
  "scores": [0.82, 0.71],
  "latency_ms": 45.2,
  "generation": {
    "answer": "...",
    "model": "gemini-2.0-flash",
    "latency_ms": 820.5,
    "context_chunks": 5
  },
  "metadata": { "chunk_size": 512, "chunk_overlap": 64 }
}
```

---

## Extension roadmap

All planned enhancements are implemented:

| Area | Features |
|------|----------|
| **Vector DB** | FAISS, Qdrant (local/remote), Pinecone |
| **Embeddings** | BGE, MiniLM, OpenAI, Cohere |
| **Retrieval** | Dense, BM25, hybrid RRF, ColBERT late interaction |
| **RAG quality** | Semantic chunking, query expansion (multi/HyDE), citation verification |
| **Generation** | Gemini, OpenAI, Claude |
| **Eval** | nDCG/MRR/recall@K, LLM-as-judge, human rubrics UI |
| **Ingestion** | PDF/TXT/MD/HTML/DOCX+tables, web fetch, site crawler, dedup/versioning |
| **Production** | Streamlit auth, Redis query cache, OpenTelemetry, file logging |

### Optional services (`.env`)

| Variable | Purpose |
|----------|---------|
| `QDRANT_URL` | Remote Qdrant (omit for local `qdrant_data/`) |
| `PINECONE_API_KEY` | Pinecone managed index |
| `REDIS_URL` | Query result cache |
| `STREAMLIT_PASSWORD` | UI login gate |
| `OTEL_ENABLED=true` | Console trace spans |
| `COHERE_API_KEY` / `ANTHROPIC_API_KEY` | Cohere embed / Claude generation |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No documents found | Add `.pdf`, `.txt`, or `.md` to `data/` |
| Gemini error | Check `GEMINI_API_KEY` in `.env` |
| Slow first run | Models download on first embed/rerank |
| Out of memory | Use `minilm`, larger chunks, fewer docs |
| Import errors | Run from project root |

---

## License

MIT — use freely for learning and prototyping.
