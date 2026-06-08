"""Small text helpers shared by retrieval, chunking, and generation."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from html import unescape
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[0-9a-zA-ZÀ-ỹ]+", re.UNICODE)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_for_search(text: str) -> str:
    text = unescape(text or "")
    text = strip_accents(text).lower()
    return normalize_whitespace(text)


def tokenize(text: str) -> list[str]:
    normalized = normalize_for_search(text)
    return TOKEN_RE.findall(normalized)


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def stable_id(*parts: str, length: int = 16) -> str:
    raw = "::".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:length]


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if not a_list or not b_list or len(a_list) != len(b_list):
        return 0.0
    dot = sum(x * y for x, y in zip(a_list, b_list))
    norm_a = math.sqrt(sum(x * x for x in a_list))
    norm_b = math.sqrt(sum(y * y for y in b_list))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def slugify(text: str, fallback: str = "document") -> str:
    tokens = tokenize(text)
    slug = "-".join(tokens[:12])
    return slug or fallback

