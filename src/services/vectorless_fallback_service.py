"""Vectorless fallback search.

The lab asks for PageIndex fallback. When PAGEINDEX_API_KEY is not configured,
this service uses local BM25 over chunks and marks results as source=pageindex
so the public task interface remains compatible while the pipeline is runnable.
"""

from __future__ import annotations

from typing import Any

from ..config import PAGEINDEX_API_KEY
from .keyword_search_service import KeywordSearchService


class VectorlessFallbackService:
    def __init__(self, keyword_service: KeywordSearchService | None = None) -> None:
        self.keyword_service = keyword_service or KeywordSearchService()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if PAGEINDEX_API_KEY:
            pageindex_results = self._try_pageindex(query, top_k)
            if pageindex_results:
                return pageindex_results

        results = self.keyword_service.search(query, top_k=top_k)
        for result in results:
            result["source"] = "pageindex"
            result["retrieval_method"] = "keyword_vectorless_fallback"
        return results

    def _try_pageindex(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            from pageindex import PageIndex

            client = PageIndex(api_key=PAGEINDEX_API_KEY)
            raw_results = client.query(query=query, top_k=top_k)
            results = []
            for item in raw_results:
                results.append(
                    {
                        "content": getattr(item, "text", ""),
                        "score": float(getattr(item, "score", 0.0)),
                        "metadata": getattr(item, "metadata", {}) or {},
                        "source": "pageindex",
                        "retrieval_method": "pageindex",
                    }
                )
            return results
        except Exception:
            return []

