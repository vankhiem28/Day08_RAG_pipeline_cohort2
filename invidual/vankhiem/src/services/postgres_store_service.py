"""PostgreSQL + pgvector storage/search service."""

from __future__ import annotations

import json
from typing import Any

from ..config import DATABASE_URL, EMBEDDING_DIM
from ..models import ChunkRecord
from .embedding_service import EmbeddingService
from .text_utils import normalize_for_search


class PostgresStoreService:
    """Source-of-truth PostgreSQL store with pgvector search index."""

    def __init__(
        self,
        database_url: str = DATABASE_URL,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.database_url = database_url
        self.embedding_service = embedding_service or EmbeddingService()

    def available(self) -> bool:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception:
            return False

    def init_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );

                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        embedding_text TEXT NOT NULL,
                        search_text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({EMBEDDING_DIM}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_text TEXT NOT NULL DEFAULT ''")
                cur.execute(
                    """
                    UPDATE chunks
                    SET search_text = lower(embedding_text)
                    WHERE search_text = ''
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_document_id_idx
                    ON chunks(document_id);
                    """
                )
                cur.execute("DROP INDEX IF EXISTS chunks_keyword_idx")
                cur.execute(
                    """
                    CREATE INDEX chunks_keyword_idx
                    ON chunks USING gin (to_tsvector('simple', search_text));
                    """
                )
            conn.commit()
        self._create_vector_index()

    def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return
        self.init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                for doc in documents:
                    cur.execute(
                        """
                        INSERT INTO documents (document_id, content, metadata, updated_at)
                        VALUES (%s, %s, %s::jsonb, now())
                        ON CONFLICT (document_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            updated_at = now()
                        """,
                        (
                            doc["document_id"],
                            doc["content"],
                            json.dumps(doc.get("metadata", {}), ensure_ascii=False),
                        ),
                    )
            conn.commit()

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        documents: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []
        self.init_schema()
        if documents:
            self.upsert_documents(documents)

        indexed = self._ensure_embeddings(chunks)
        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk in indexed:
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id, document_id, content, embedding_text, search_text,
                            metadata, embedding, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector, now())
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            content = EXCLUDED.content,
                            embedding_text = EXCLUDED.embedding_text,
                            search_text = EXCLUDED.search_text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            updated_at = now()
                        """,
                        (
                            chunk["chunk_id"],
                            chunk["document_id"],
                            chunk["content"],
                            chunk.get("embedding_text", chunk["content"]),
                            normalize_for_search(chunk.get("embedding_text", chunk["content"])),
                            json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                            self._vector_literal(chunk["embedding"]),
                        ),
                    )
            conn.commit()
        return indexed

    def vector_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query_embedding = self.embedding_service.embed_query(query)
        vector_literal = self._vector_literal(query_embedding)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, document_id, content, embedding_text, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_literal, vector_literal, top_k),
                )
                rows = cur.fetchall()
        return [self._row_to_result(row, source="hybrid", method="pgvector") for row in rows]

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, document_id, content, embedding_text, metadata,
                           ts_rank_cd(
                               to_tsvector('simple', search_text),
                               plainto_tsquery('simple', %s)
                           ) AS score
                    FROM chunks
                    WHERE to_tsvector('simple', search_text) @@ plainto_tsquery('simple', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (
                        normalize_for_search(query),
                        normalize_for_search(query),
                        top_k,
                    ),
                )
                rows = cur.fetchall()
        return [self._row_to_result(row, source="hybrid", method="postgres_keyword") for row in rows]

    def count_chunks(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM chunks")
                row = cur.fetchone()
        return int(row[0]) if row else 0

    def _ensure_embeddings(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        missing = [chunk for chunk in chunks if not chunk.get("embedding")]
        if missing:
            embeddings = self.embedding_service.embed_texts(
                chunk.get("embedding_text") or chunk["content"] for chunk in missing
            )
            for chunk, embedding in zip(missing, embeddings):
                chunk["embedding"] = embedding
        return chunks

    def _row_to_result(
        self,
        row: tuple[Any, ...],
        source: str,
        method: str,
    ) -> dict[str, Any]:
        chunk_id, document_id, content, embedding_text, metadata, score = row
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        record = ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            embedding_text=embedding_text,
            metadata=metadata or {},
        )
        result = record.to_result(score=float(score or 0.0), source=source)
        result["retrieval_method"] = method
        return result

    def _vector_literal(self, embedding: list[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "Missing psycopg. Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        return psycopg.connect(self.database_url, connect_timeout=5)

    def _create_vector_index(self) -> None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                        ON chunks USING hnsw (embedding vector_cosine_ops);
                        """
                    )
                conn.commit()
        except Exception:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS chunks_embedding_ivfflat_idx
                        ON chunks USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                        """
                    )
                conn.commit()
