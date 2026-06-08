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

import hashlib
import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_PATH = Path(__file__).parent.parent / "data" / "rag_index.json"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # 500 ký tự giữ chunk đủ ngắn để cite rõ và phù hợp test.
CHUNK_OVERLAP = 50      # 50 ký tự giúp câu/điều khoản ở ranh giới không bị mất ngữ cảnh.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding nhỏ, nhanh, tốt cho search đa ngôn ngữ.
EMBEDDING_DIM = 1536
EMBEDDING_FALLBACK_MODEL = "local-hashing-tfidf"  # Fallback để test vẫn chạy khi chưa set OPENAI_API_KEY.

# TODO: Chọn vector store
VECTOR_STORE = "local-json"  # Lưu vector OpenAI vào JSON local; có thể thay bằng Weaviate khi deploy.


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        rel_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(rel_path),
                "type": doc_type,
            },
        })
    return documents


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive-ish splitter: ưu tiên paragraph, line, sentence, space."""
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    pieces = [text.strip()]
    for sep in separators:
        next_pieces = []
        changed = False
        for piece in pieces:
            if len(piece) <= chunk_size:
                next_pieces.append(piece)
                continue
            parts = [p.strip() for p in piece.split(sep) if p.strip()]
            if len(parts) == 1:
                next_pieces.append(piece)
            else:
                changed = True
                suffix = sep.strip()
                next_pieces.extend((p + suffix).strip() for p in parts)
        pieces = next_pieces
        if changed and all(len(p) <= chunk_size for p in pieces):
            break

    chunks = []
    current = ""
    for piece in pieces:
        if len(piece) > chunk_size:
            step = max(1, chunk_size - overlap)
            for start in range(0, len(piece), step):
                part = piece[start:start + chunk_size].strip()
                if part:
                    chunks.append(part)
            continue

        candidate = f"{current}\n{piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            prefix = current[-overlap:] if overlap and current else ""
            current = f"{prefix} {piece}".strip()
            if len(current) > chunk_size:
                chunks.append(piece)
                current = ""
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    for doc_id, doc in enumerate(documents):
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            )
            splits = splitter.split_text(doc["content"])
        except Exception:
            splits = _split_long_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(s.strip() for s in splits if s.strip()):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "doc_id": doc_id, "chunk_index": i},
            })
    return chunks


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho tiếng Việt: lowercase, bỏ dấu câu, giữ số điều luật."""
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def hashing_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Tạo vector băm TF-IDF-lite để semantic search chạy offline ổn định."""
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def current_embedding_backend() -> str:
    """Backend đang dùng thật sự: OpenAI nếu có key, local nếu chưa cấu hình key."""
    return EMBEDDING_MODEL if os.getenv("OPENAI_API_KEY") else EMBEDDING_FALLBACK_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts bằng OpenAI; fallback local khi chưa có OPENAI_API_KEY."""
    if not texts:
        return []

    if not os.getenv("OPENAI_API_KEY"):
        return [hashing_embedding(text) for text in texts]

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings: list[list[float]] = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def embed_text(text: str) -> list[float]:
    """Embed một query/document bằng cùng backend với index hiện tại."""
    return embed_texts([text])[0]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    backend = current_embedding_backend()
    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        chunk["metadata"] = {
            **chunk.get("metadata", {}),
            "embedding_model": backend,
            "embedding_dim": len(embedding),
        }
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return INDEX_PATH


def load_or_build_chunks(with_embeddings: bool = False) -> list[dict]:
    """Shared helper cho Task 5-9."""
    backend = current_embedding_backend()
    if with_embeddings and INDEX_PATH.exists():
        try:
            data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if (
                data
                and "embedding" in data[0]
                and data[0].get("metadata", {}).get("embedding_model") == backend
            ):
                return data
        except json.JSONDecodeError:
            pass

    chunks = chunk_documents(load_documents())
    if with_embeddings:
        chunks = embed_chunks(chunks)
        index_to_vectorstore(chunks)
    return chunks


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

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
