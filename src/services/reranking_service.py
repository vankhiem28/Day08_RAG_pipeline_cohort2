"""Reranking and fusion helpers."""

from __future__ import annotations

from typing import Any

from .text_utils import tokenize


class RerankingService:
    """Local reranking strategies used by the retrieval pipeline."""

    def rrf(
        self,
        ranked_lists: list[list[dict[str, Any]]],
        top_k: int = 5,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        items: dict[str, dict[str, Any]] = {}
        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list, 1):
                key = self._key(item)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                existing = items.get(key)
                if existing is None or item.get("score", 0.0) > existing.get("score", 0.0):
                    items[key] = dict(item)

        fused = []
        for key, score in scores.items():
            item = items[key]
            item["score"] = float(score)
            item["fusion_score"] = float(score)
            item["source"] = item.get("source", "hybrid")
            fused.append(item)
        fused.sort(key=lambda item: item["score"], reverse=True)
        return fused[:top_k]

    def lexical_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_terms = set(tokenize(query))
        reranked: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            content_terms = set(tokenize(candidate.get("content", "")))
            overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            base_score = float(candidate.get("score", 0.0))
            score = 0.65 * base_score + 0.35 * overlap
            item = dict(candidate)
            item["score"] = float(score)
            item["rerank_score"] = float(score)
            item["original_rank"] = index + 1
            reranked.append(item)
        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]

    def _key(self, item: dict[str, Any]) -> str:
        metadata = item.get("metadata", {})
        return metadata.get("chunk_id") or item.get("chunk_id") or item.get("content", "")

