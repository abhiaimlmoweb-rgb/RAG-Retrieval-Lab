"""Central configuration for RAG Retrieval Lab."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
QDRANT_PATH = PROJECT_ROOT / "qdrant_data"

EMBEDDING_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
    "cohere": "embed-english-v3.0",
}

RETRIEVAL_MODES = {
    "dense": "Dense embedding similarity",
    "bm25": "BM25 lexical",
    "hybrid": "Hybrid dense + BM25 (RRF)",
    "colbert": "ColBERT late interaction",
}

INDEX_BACKENDS = {
    "memory": "In-memory NumPy cosine",
    "faiss": "FAISS vector index",
    "qdrant": "Qdrant (local or remote)",
    "pinecone": "Pinecone (managed cloud)",
}

CHUNKING_STRATEGIES = {
    "recursive": "Recursive (paragraph/sentence)",
    "fixed": "Fixed character windows",
    "semantic": "Semantic (embedding breakpoints)",
}

QUERY_EXPANSION_MODES = {
    "none": "No expansion",
    "multi": "Multi-query paraphrases",
    "hyde": "HyDE hypothetical document",
}

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# API keys
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
COHERE_API_KEY_ENV = "COHERE_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
PINECONE_API_KEY_ENV = "PINECONE_API_KEY"
REDIS_URL_ENV = "REDIS_URL"
STREAMLIT_PASSWORD_ENV = "STREAMLIT_PASSWORD"
OTEL_ENABLED_ENV = "OTEL_ENABLED"

# Models
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_COHERE_EMBED_MODEL = "embed-english-v3.0"
DEFAULT_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-latest")

GEMINI_MODELS = {
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
}
OPENAI_CHAT_MODELS = {
    "gpt-4o-mini": "GPT-4o mini",
    "gpt-4o": "GPT-4o",
}
CLAUDE_MODELS = {
    "claude-3-5-haiku-latest": "Claude 3.5 Haiku",
    "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet",
}

GENERATOR_PROVIDERS = {
    "gemini": "Google Gemini",
    "openai": "OpenAI Chat",
    "claude": "Anthropic Claude",
}

# Qdrant / Pinecone
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_lab")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-retrieval-lab")

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 5
DEFAULT_RERANK_POOL = 15
DEFAULT_CRAWL_MAX_PAGES = 10
