"""Central configuration for RAG Retrieval Lab."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root is one level above this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root (GEMINI_API_KEY, etc.)
load_dotenv(PROJECT_ROOT / ".env")

# Default folders for documents and experiment artifacts
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Supported embedding model identifiers (Hugging Face hub names)
EMBEDDING_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

# Retrieval backends
RETRIEVAL_MODES = {
    "dense": "Dense cosine (embedding similarity)",
    "bm25": "BM25 lexical (keyword matching)",
    "hybrid": "Hybrid dense + BM25 (RRF fusion)",
}

# Cross-encoder reranker (downloads on first use)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Google Gemini
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_MODELS = {
    "gemini-2.0-flash": "Gemini 2.0 Flash (fast, recommended)",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite (cheapest)",
    "gemini-1.5-pro": "Gemini 1.5 Pro (higher quality)",
}

# Chunking strategy registry keys
CHUNKING_STRATEGIES = ("fixed", "recursive")

# Default retrieval and chunking parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 5
DEFAULT_RERANK_POOL = 15  # candidates fetched before reranking
