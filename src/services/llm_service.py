"""LLM generation with offline extractive fallback."""

from __future__ import annotations

import os
from typing import Any

from ..config import OPENAI_API_KEY, OPENAI_MODEL
from .prompt_builder_service import PromptBuilderService
from .text_utils import sentence_split, tokenize


class LLMService:
    def __init__(self, prompt_builder: PromptBuilderService | None = None) -> None:
        self.prompt_builder = prompt_builder or PromptBuilderService()

    def generate(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        if not chunks:
            return "Không đủ dữ liệu từ nguồn hiện có để trả lời câu hỏi này."

        api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        if api_key:
            answer = self._try_openai(query, chunks, api_key, temperature, top_p)
            if answer:
                return answer
        return self._extractive_answer(query, chunks)

    def _try_openai(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        api_key: str,
        temperature: float,
        top_p: float,
    ) -> str:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self.prompt_builder.build_messages(query, chunks),
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def _extractive_answer(self, query: str, chunks: list[dict[str, Any]]) -> str:
        query_terms = set(tokenize(query))
        lines: list[str] = []
        for chunk in chunks[:4]:
            citation_id = chunk.get("metadata", {}).get("citation_id", "?")
            sentence = self._best_sentence(chunk["content"], query_terms)
            if sentence:
                lines.append(f"{sentence} [{citation_id}]")

        if not lines:
            return "Không đủ dữ liệu từ nguồn hiện có để trả lời câu hỏi này."

        prefix = "Dựa trên các nguồn đã truy xuất, "
        return prefix + "\n".join(lines)

    def _best_sentence(self, content: str, query_terms: set[str]) -> str:
        sentences = sentence_split(content)
        if not sentences:
            return content[:500].strip()

        def score(sentence: str) -> float:
            terms = set(tokenize(sentence))
            if not terms:
                return 0.0
            return len(terms & query_terms) / len(query_terms or terms)

        best = max(sentences, key=score)
        if len(best) > 650:
            return best[:647].rstrip() + "..."
        return best.strip()
