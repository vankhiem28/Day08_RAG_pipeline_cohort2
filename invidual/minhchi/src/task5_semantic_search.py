"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from .task4_chunking_indexing import embed_text, load_or_build_chunks


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    chunks = load_or_build_chunks(with_embeddings=True)
    query_embedding = embed_text(query)
    results = []
    for chunk in chunks:
        score = _cosine(query_embedding, chunk.get("embedding", []))
        results.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk.get("metadata", {}),
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:max(0, top_k)]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
