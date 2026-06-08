"""Embedding service with a deterministic local fallback."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

from ..config import EMBEDDING_BACKEND, EMBEDDING_DIM, EMBEDDING_MODEL
from .text_utils import tokenize


class EmbeddingService:
    """
    Embed text for retrieval.

    Default backend is a local hashing embedding so the lab runs without
    downloading a model. Set RAG_EMBEDDING_BACKEND=sentence-transformers and
    RAG_EMBEDDING_MODEL to use a real local SentenceTransformer model.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIM,
        backend: str = EMBEDDING_BACKEND,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.backend = backend
        self._model = None

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        texts_list = list(texts)
        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                if self._model is None:
                    self._model = SentenceTransformer(self.model_name)
                embeddings = self._model.encode(texts_list, show_progress_bar=False)
                return [list(map(float, emb)) for emb in embeddings]
            except Exception:
                self.backend = "local"
        return [self._hashing_embedding(text) for text in texts_list]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def _hashing_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            self._add_feature(vector, token, 1.0)
            for ngram in self._char_ngrams(token):
                self._add_feature(vector, ngram, 0.25)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _add_feature(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "big")
        index = raw % self.dimension
        sign = 1.0 if (raw >> 1) % 2 == 0 else -1.0
        vector[index] += sign * weight

    def _char_ngrams(self, token: str, n: int = 3) -> list[str]:
        if len(token) <= n:
            return [token]
        return [token[i : i + n] for i in range(len(token) - n + 1)]

