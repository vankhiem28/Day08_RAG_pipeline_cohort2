"""Local vector index backed by JSONL files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import CHUNKS_PATH, VECTOR_BACKEND
from ..models import ChunkRecord
from .chunking_service import load_jsonl, write_jsonl
from .embedding_service import EmbeddingService
from .text_utils import cosine_similarity


class VectorStoreService:
    """Vector search index. Replace this service with Qdrant/pgvector later."""

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        embedding_service: EmbeddingService | None = None,
        backend: str = VECTOR_BACKEND,
    ) -> None:
        self.chunks_path = chunks_path
        self.embedding_service = embedding_service or EmbeddingService()
        self.backend = backend

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        documents: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        texts = [chunk.get("embedding_text") or chunk["content"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        indexed: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings):
            record = dict(chunk)
            record["embedding"] = embedding
            indexed.append(record)
        write_jsonl(self.chunks_path, indexed)
        if self.backend == "pgvector":
            try:
                from .postgres_store_service import PostgresStoreService

                PostgresStoreService(embedding_service=self.embedding_service).upsert_chunks(
                    indexed,
                    documents=documents,
                )
            except Exception:
                pass
        return indexed

    def load_chunks(self) -> list[dict[str, Any]]:
        return load_jsonl(self.chunks_path)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if self.backend == "pgvector":
            try:
                from .postgres_store_service import PostgresStoreService

                return PostgresStoreService(embedding_service=self.embedding_service).vector_search(
                    query,
                    top_k=top_k,
                )
            except Exception:
                pass

        chunks = self.load_chunks()
        if not chunks:
            return []
        query_embedding = self.embedding_service.embed_query(query)
        results: list[dict[str, Any]] = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if not embedding:
                continue
            score = cosine_similarity(query_embedding, embedding)
            record = ChunkRecord(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                content=chunk["content"],
                embedding_text=chunk.get("embedding_text", chunk["content"]),
                metadata=chunk.get("metadata", {}),
                embedding=embedding,
            )
            result = record.to_result(score=score, source="hybrid")
            result["retrieval_method"] = "vector"
            results.append(result)
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
