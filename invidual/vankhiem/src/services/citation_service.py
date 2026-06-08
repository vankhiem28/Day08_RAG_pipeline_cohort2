"""Citation formatting for retrieved chunks."""

from __future__ import annotations

from typing import Any


class CitationService:
    def attach_citation_ids(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cited = []
        for index, chunk in enumerate(chunks, 1):
            item = dict(chunk)
            metadata = dict(item.get("metadata", {}))
            metadata["citation_id"] = index
            item["metadata"] = metadata
            cited.append(item)
        return cited

    def citation_label(self, chunk: dict[str, Any]) -> str:
        metadata = chunk.get("metadata", {})
        citation_id = metadata.get("citation_id", "?")
        source = metadata.get("source") or metadata.get("title") or "Nguồn"
        article = metadata.get("article")
        if article:
            return f"[{citation_id}] {source}, {article}"
        return f"[{citation_id}] {source}"

    def source_list(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            sources.append(
                {
                    "citation": f"[{metadata.get('citation_id', '?')}]",
                    "chunk_id": metadata.get("chunk_id"),
                    "document_id": metadata.get("document_id"),
                    "source": metadata.get("source"),
                    "title": metadata.get("title"),
                    "article": metadata.get("article"),
                    "score": chunk.get("score"),
                }
            )
        return sources

