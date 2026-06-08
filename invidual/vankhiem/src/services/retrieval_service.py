"""Hybrid retrieval with vectorless fallback and structured logging."""

from __future__ import annotations

import re
from typing import Any

from ..config import HYBRID_SCORE_THRESHOLD, VECTOR_SCORE_THRESHOLD
from .indexing_service import IndexingService
from .keyword_search_service import KeywordSearchService
from .logging_service import LoggingService
from .reranking_service import RerankingService
from .vector_store_service import VectorStoreService
from .vectorless_fallback_service import VectorlessFallbackService


class RetrievalService:
    EXACT_QUERY_RE = re.compile(
        r"(Điều\s+\d+[a-zA-Z]?|\d{2,4}/\d{4}/[A-ZĐ-]+|\d{1,2}/\d{1,2}/\d{4})",
        re.IGNORECASE,
    )

    def __init__(
        self,
        indexing_service: IndexingService | None = None,
        vector_service: VectorStoreService | None = None,
        keyword_service: KeywordSearchService | None = None,
        reranking_service: RerankingService | None = None,
        fallback_service: VectorlessFallbackService | None = None,
        logging_service: LoggingService | None = None,
    ) -> None:
        self.indexing_service = indexing_service or IndexingService()
        self.vector_service = vector_service or VectorStoreService()
        self.keyword_service = keyword_service or KeywordSearchService()
        self.reranking_service = reranking_service or RerankingService()
        self.fallback_service = fallback_service or VectorlessFallbackService(self.keyword_service)
        self.logging_service = logging_service or LoggingService()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = HYBRID_SCORE_THRESHOLD,
        use_reranking: bool = True,
    ) -> list[dict[str, Any]]:
        self.indexing_service.load_or_build_chunks()
        exact_search = self.is_exact_search(query)

        vector_results: list[dict[str, Any]] = []
        vector_error = False
        try:
            vector_results = self.vector_service.search(query, top_k=top_k * 2)
        except Exception:
            vector_error = True

        keyword_results = self.keyword_service.search(query, top_k=top_k * 2)
        vector_best = vector_results[0]["score"] if vector_results else 0.0

        should_fallback = (
            exact_search
            or vector_error
            or not vector_results
            or vector_best < VECTOR_SCORE_THRESHOLD
        )
        if should_fallback and keyword_results:
            final_results = self._fallback(query, top_k)
            self._log(query, "vectorless_fallback", final_results, fallback=True)
            return final_results

        merged = self.reranking_service.rrf(
            [vector_results, keyword_results],
            top_k=top_k * 2,
        )
        for result in merged:
            result["source"] = "hybrid"
            result["retrieval_method"] = "hybrid_rrf"

        if use_reranking and merged:
            final_results = self.reranking_service.lexical_rerank(query, merged, top_k=top_k)
        else:
            final_results = merged[:top_k]

        if not final_results or final_results[0].get("score", 0.0) < score_threshold:
            final_results = self._fallback(query, top_k)
            self._log(query, "low_score_vectorless_fallback", final_results, fallback=True)
            return final_results

        self._log(query, "hybrid", final_results, fallback=False)
        return final_results[:top_k]

    def is_exact_search(self, query: str) -> bool:
        return bool(self.EXACT_QUERY_RE.search(query or ""))

    def _fallback(self, query: str, top_k: int) -> list[dict[str, Any]]:
        return self.fallback_service.search(query, top_k=top_k)

    def _log(
        self,
        query: str,
        retrieval_mode: str,
        results: list[dict[str, Any]],
        fallback: bool,
    ) -> None:
        self.logging_service.log_retrieval(
            {
                "question": query,
                "retrieval_mode": retrieval_mode,
                "retrieved_chunk_ids": [
                    result.get("metadata", {}).get("chunk_id") for result in results
                ],
                "scores": [float(result.get("score", 0.0)) for result in results],
                "fallback": fallback,
            }
        )
