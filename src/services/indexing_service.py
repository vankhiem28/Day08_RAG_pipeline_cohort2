"""Orchestrates document loading, chunking, embedding, and local indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import CHUNKS_PATH, DOCUMENTS_PATH, STANDARDIZED_DIR
from .chunking_service import ChunkingService, load_jsonl, write_jsonl
from .vector_store_service import VectorStoreService


class IndexingService:
    def __init__(
        self,
        standardized_dir: Path = STANDARDIZED_DIR,
        documents_path: Path = DOCUMENTS_PATH,
        chunks_path: Path = CHUNKS_PATH,
    ) -> None:
        self.standardized_dir = standardized_dir
        self.documents_path = documents_path
        self.chunks_path = chunks_path
        self.chunking_service = ChunkingService(standardized_dir=standardized_dir)
        self.vector_store = VectorStoreService(chunks_path=chunks_path)

    def load_documents(self) -> list[dict[str, Any]]:
        self._ensure_sample_data_if_empty()
        return self.chunking_service.load_documents()

    def chunk_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.chunking_service.chunk_documents(documents)

    def build_index(self, force: bool = False) -> list[dict[str, Any]]:
        if self.chunks_path.exists() and not force:
            existing = load_jsonl(self.chunks_path)
            if existing:
                return existing

        documents = self.load_documents()
        write_jsonl(self.documents_path, documents)
        chunks = self.chunk_documents(documents)
        return self.vector_store.index_chunks(chunks, documents=documents)

    def load_or_build_chunks(self) -> list[dict[str, Any]]:
        chunks = load_jsonl(self.chunks_path)
        if chunks:
            return chunks
        return self.build_index(force=True)

    def _ensure_sample_data_if_empty(self) -> None:
        md_files = list(self.standardized_dir.rglob("*.md")) if self.standardized_dir.exists() else []
        if md_files:
            return
        from ..sample_data import ensure_sample_data

        ensure_sample_data()
