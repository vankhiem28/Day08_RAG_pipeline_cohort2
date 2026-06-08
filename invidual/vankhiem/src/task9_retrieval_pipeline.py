"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Pipeline:
    semantic vector search + keyword BM25
    -> RRF merge + deduplicate
    -> local rerank
    -> vectorless fallback for exact/low-score/vector errors
"""

from .config import HYBRID_SCORE_THRESHOLD
from .services.retrieval_service import RetrievalService


SCORE_THRESHOLD = HYBRID_SCORE_THRESHOLD
DEFAULT_TOP_K = 5
RERANK_METHOD = "lexical"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.
    """
    return RetrievalService().retrieve(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
    )


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
