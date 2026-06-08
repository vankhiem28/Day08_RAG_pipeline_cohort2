"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL, VECTOR_BACKEND
from .services.embedding_service import EmbeddingService
from .services.indexing_service import IndexingService
from .services.vector_store_service import VectorStoreService

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn 900 ký tự để một Điều luật ngắn thường nằm trọn trong một chunk,
# overlap 120 giúp giữ ngữ cảnh khi Điều/khoản dài bị tách.
CHUNKING_METHOD = "domain_aware"  # legal articles | news paragraphs | Q&A blocks

# Mặc định dùng local-hashing-v1 để lab chạy offline. Có thể đổi sang
# sentence-transformers bằng biến môi trường RAG_EMBEDDING_BACKEND.

# Source of truth mặc định là JSONL local. Bật RAG_VECTOR_BACKEND=pgvector
# để ghi documents/chunks vào PostgreSQL và search vector bằng pgvector.
VECTOR_STORE = VECTOR_BACKEND


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    return IndexingService().load_documents()


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    return IndexingService().chunk_documents(documents)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    service = EmbeddingService()
    texts = [chunk.get("embedding_text") or chunk["content"] for chunk in chunks]
    embeddings = service.embed_texts(texts)
    embedded = []
    for chunk, embedding in zip(chunks, embeddings):
        item = dict(chunk)
        item["embedding"] = embedding
        embedded.append(item)
    return embedded


def index_to_vectorstore(chunks: list[dict], documents: list[dict] | None = None):
    """
    Lưu chunks vào vector store đã chọn.
    """
    VectorStoreService().index_chunks(chunks, documents=documents)


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks, documents=docs)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
