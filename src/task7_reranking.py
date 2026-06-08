"""
Task 7 — Reranking Module.

Default path is local and deterministic:
- Jina cross-encoder is used only when JINA_API_KEY is configured.
- Otherwise lexical reranking keeps the lab runnable offline.
- RRF is used for hybrid fusion.
"""

import math

from .config import JINA_API_KEY
from .services.reranking_service import RerankingService
from .services.text_utils import cosine_similarity


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates with Jina API when available, otherwise local rerank.
    """
    if JINA_API_KEY:
        try:
            import requests

            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k,
                },
                timeout=20,
            )
            response.raise_for_status()
            reranked = response.json()["results"]
            return [
                {**candidates[item["index"]], "score": float(item["relevance_score"])}
                for item in reranked
            ]
        except Exception:
            pass
    return RerankingService().lexical_rerank(query, candidates, top_k=top_k)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    selected: list[int] = []
    remaining = [i for i, item in enumerate(candidates) if item.get("embedding")]
    if not remaining:
        return candidates[:top_k]

    for _ in range(min(top_k, len(remaining))):
        best_idx: int | None = None
        best_score = -math.inf
        for idx in remaining:
            relevance = cosine_similarity(query_embedding, candidates[idx]["embedding"])
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    cosine_similarity(candidates[idx]["embedding"], candidates[sel]["embedding"])
                    for sel in selected
                )
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        item = dict(candidates[best_idx])
        item["score"] = float(best_score)
        candidates[best_idx] = item
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.
    """
    return RerankingService().rrf(ranked_lists, top_k=top_k, k=k)


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "lexical":
        return RerankingService().lexical_rerank(query, candidates, top_k=top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
