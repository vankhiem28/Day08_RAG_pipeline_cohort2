"""Cleaning utilities for raw and standardized documents."""

from __future__ import annotations

import re
from html import unescape

from .text_utils import normalize_whitespace


class CleanerService:
    """Clean HTML/text while preserving source content for citations."""

    SCRIPT_RE = re.compile(r"<(script|style).*?>.*?</\1>", re.IGNORECASE | re.DOTALL)
    TAG_RE = re.compile(r"<[^>]+>")

    def clean_html(self, html: str) -> str:
        text = self.SCRIPT_RE.sub(" ", html or "")
        text = self.TAG_RE.sub(" ", text)
        return self.clean_text(unescape(text))

    def clean_text(self, text: str) -> str:
        text = (text or "").replace("\ufeff", " ")
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def build_embedding_text(self, content: str, metadata: dict) -> str:
        title = metadata.get("title") or metadata.get("source") or ""
        doc_type = metadata.get("type") or metadata.get("doc_type") or ""
        combined = f"{title}\n{doc_type}\n{content}"
        return normalize_whitespace(combined.lower())

