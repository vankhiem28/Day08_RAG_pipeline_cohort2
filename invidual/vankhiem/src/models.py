"""Typed records used by the RAG services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentRecord:
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    content: str
    embedding_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "embedding_text": self.embedding_text,
            "metadata": self.metadata,
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding
        return data

    def to_result(self, score: float, source: str = "hybrid") -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata.setdefault("chunk_id", self.chunk_id)
        metadata.setdefault("document_id", self.document_id)
        return {
            "content": self.content,
            "score": float(score),
            "metadata": metadata,
            "source": source,
        }


@dataclass
class RetrievalTrace:
    question: str
    retrieval_mode: str
    retrieved_chunk_ids: list[str]
    scores: list[float]
    fallback: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "retrieval_mode": self.retrieval_mode,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "scores": self.scores,
            "fallback": self.fallback,
        }

