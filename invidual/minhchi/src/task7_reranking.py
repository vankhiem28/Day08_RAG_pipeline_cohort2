"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os

from dotenv import load_dotenv

from .task4_chunking_indexing import tokenize

load_dotenv()
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

QWEN_RERANK_MODEL = os.getenv("QWEN_RERANK_MODEL_PATH", "Qwen/Qwen3-Reranker-0.6B")
_QWEN_RERANKER = None


def rerank_local_overlap(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Fallback reranker local khi Qwen chưa load được."""
    query_terms = set(tokenize(query))
    rescored = []
    for rank, candidate in enumerate(candidates, 1):
        doc_terms = set(tokenize(candidate.get("content", "")))
        overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
        original_score = float(candidate.get("score", 0.0))
        rank_bonus = 1.0 / (rank + 1)
        score = 0.65 * overlap + 0.25 * original_score + 0.10 * rank_bonus
        item = candidate.copy()
        item["score"] = float(score)
        item["metadata"] = {**item.get("metadata", {}), "reranker": "local-token-overlap"}
        rescored.append(item)
    rescored.sort(key=lambda item: item["score"], reverse=True)
    return rescored[:max(0, top_k)]


def load_qwen_reranker():
    """Lazy-load Qwen reranker local/Hugging Face cache."""
    global _QWEN_RERANKER
    if _QWEN_RERANKER is None:
        from sentence_transformers import CrossEncoder

        _QWEN_RERANKER = CrossEncoder(
            QWEN_RERANK_MODEL,
            trust_remote_code=True,
        )
    return _QWEN_RERANKER


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    try:
        model = load_qwen_reranker()
        pairs = [(query, candidate.get("content", "")) for candidate in candidates]
        scores = model.predict(pairs)
    except Exception:
        return rerank_local_overlap(query, candidates, top_k)

    results = []
    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["score"] = float(score)
        item["metadata"] = {
            **item.get("metadata", {}),
            "reranker": QWEN_RERANK_MODEL,
        }
        results.append(item)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:max(0, top_k)]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            emb = candidates[idx].get("embedding", [])
            relevance = cosine(query_embedding, emb) if emb else float(candidates[idx].get("score", 0))
            diversity_penalty = 0.0
            for selected_idx in selected:
                other = candidates[selected_idx].get("embedding", [])
                if emb and other:
                    diversity_penalty = max(diversity_penalty, cosine(emb, other))
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    results = []
    for content, score in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        candidate = content_map[content].copy()
        candidate["score"] = float(score)
        results.append(candidate)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
