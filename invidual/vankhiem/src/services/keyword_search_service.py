"""Keyword retrieval using a compact BM25 implementation."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import CHUNKS_PATH, VECTOR_BACKEND
from ..models import ChunkRecord
from .chunking_service import load_jsonl
from .text_utils import tokenize


class KeywordSearchService:
    """BM25 search over locally indexed chunks."""

    def __init__(self, chunks_path: Path = CHUNKS_PATH, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks_path = chunks_path
        self.k1 = k1
        self.b = b
        self.backend = VECTOR_BACKEND

    def load_corpus(self) -> list[dict[str, Any]]:
        return load_jsonl(self.chunks_path)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if self.backend == "pgvector":
            try:
                from .postgres_store_service import PostgresStoreService

                return PostgresStoreService().keyword_search(query, top_k=top_k)
            except Exception:
                pass

        corpus = self.load_corpus()
        if not corpus:
            return []
        return self.search_corpus(query, corpus, top_k=top_k)

    def search_corpus(
        self,
        query: str,
        corpus: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        tokenized_docs = [tokenize(doc.get("embedding_text") or doc["content"]) for doc in corpus]
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        doc_count = len(tokenized_docs)
        avgdl = sum(len(tokens) for tokens in tokenized_docs) / max(doc_count, 1)
        df: Counter[str] = Counter()
        for tokens in tokenized_docs:
            df.update(set(tokens))

        query_counter = Counter(query_tokens)
        scored: list[tuple[int, float]] = []
        for idx, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            tf = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term, query_weight in query_counter.items():
                if term not in tf:
                    continue
                idf = math.log(1 + (doc_count - df[term] + 0.5) / (df[term] + 0.5))
                numerator = tf[term] * (self.k1 + 1)
                denominator = tf[term] + self.k1 * (1 - self.b + self.b * doc_len / max(avgdl, 1))
                score += query_weight * idf * numerator / denominator
            if score > 0:
                scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        results: list[dict[str, Any]] = []
        for idx, score in scored[:top_k]:
            chunk = corpus[idx]
            record = ChunkRecord(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                content=chunk["content"],
                embedding_text=chunk.get("embedding_text", chunk["content"]),
                metadata=chunk.get("metadata", {}),
                embedding=chunk.get("embedding"),
            )
            result = record.to_result(score=score, source="hybrid")
            result["retrieval_method"] = "keyword"
            results.append(result)
        return results
