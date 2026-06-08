"""Initialize PostgreSQL/pgvector and index current sample/markdown data."""

from __future__ import annotations

from .services.chunking_service import write_jsonl
from .config import DOCUMENTS_PATH
from .services.indexing_service import IndexingService
from .services.postgres_store_service import PostgresStoreService
from .services.vector_store_service import VectorStoreService


def main() -> None:
    postgres = PostgresStoreService()
    postgres.init_schema()

    indexing = IndexingService()
    documents = indexing.load_documents()
    write_jsonl(DOCUMENTS_PATH, documents)
    chunks = indexing.chunk_documents(documents)
    indexed_chunks = VectorStoreService(backend="pgvector").index_chunks(
        chunks,
        documents=documents,
    )

    print("PostgreSQL/pgvector is ready.")
    print(f"Documents indexed: {len(documents)}")
    print(f"Chunks indexed: {len(indexed_chunks)}")
    print(f"Chunks in PostgreSQL: {postgres.count_chunks()}")


if __name__ == "__main__":
    main()
