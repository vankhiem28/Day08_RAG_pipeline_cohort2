"""Prompt construction for citation-grounded generation."""

from __future__ import annotations

from typing import Any

from .citation_service import CitationService


class PromptBuilderService:
    SYSTEM_PROMPT = """Bạn là trợ lý RAG trả lời bằng tiếng Việt.
Chỉ sử dụng thông tin trong CONTEXT.
Nếu context không đủ, nói rõ là không đủ dữ liệu từ nguồn hiện có.
Không suy đoán ngoài context.
Không hướng dẫn sản xuất, mua bán, vận chuyển, che giấu hoặc né tránh pháp luật liên quan đến ma túy.
Mỗi nhận định thực tế phải có citation dạng [1], [2]."""

    def __init__(self, citation_service: CitationService | None = None) -> None:
        self.citation_service = citation_service or CitationService()

    def format_context(self, chunks: list[dict[str, Any]]) -> str:
        parts = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            citation_id = metadata.get("citation_id", "?")
            source = metadata.get("source", f"Source {citation_id}")
            doc_type = metadata.get("type", "unknown")
            title = metadata.get("title", "")
            article = metadata.get("article", "")
            header = (
                f"[{citation_id}] Source: {source} | Type: {doc_type}"
                f"{' | Title: ' + title if title else ''}"
                f"{' | Article: ' + article if article else ''}"
            )
            parts.append(f"{header}\n{chunk['content']}")
        return "\n\n---\n\n".join(parts)

    def build_messages(self, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        context = self.format_context(chunks)
        user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

