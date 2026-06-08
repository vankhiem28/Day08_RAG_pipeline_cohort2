"""Shared configuration for the local RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LANDING_DIR = DATA_DIR / "landing"
STANDARDIZED_DIR = DATA_DIR / "standardized"
INDEX_DIR = DATA_DIR / "index"
LOG_DIR = DATA_DIR / "logs"

DOCUMENTS_PATH = INDEX_DIR / "documents.jsonl"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
RETRIEVAL_LOG_PATH = LOG_DIR / "retrieval.jsonl"
GENERATION_LOG_PATH = LOG_DIR / "generation.jsonl"

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "local-hashing-v1")
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "256"))
EMBEDDING_BACKEND = os.getenv("RAG_EMBEDDING_BACKEND", "local").lower()
VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "local").lower()

VECTOR_SCORE_THRESHOLD = float(os.getenv("RAG_VECTOR_SCORE_THRESHOLD", "0.18"))
HYBRID_SCORE_THRESHOLD = float(os.getenv("RAG_HYBRID_SCORE_THRESHOLD", "0.12"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_lab")
POSTGRES_USER = os.getenv("POSTGRES_USER", "rag")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rag")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
