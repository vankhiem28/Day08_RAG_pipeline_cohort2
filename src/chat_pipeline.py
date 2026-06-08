"""Conversation-aware wrapper for the group Streamlit chatbot."""

from __future__ import annotations

from typing import Any

from .task10_generation import generate_with_citation


def build_contextual_query(
    question: str,
    history: list[dict[str, Any]] | None = None,
    max_turns: int = 2,
) -> str:
    """Attach recent turns so follow-up questions have enough retrieval context."""
    if not history:
        return question

    recent_turns = history[-max_turns:]
    lines = ["Ngữ cảnh hội thoại gần đây:"]
    for turn in recent_turns:
        previous_question = turn.get("question", "")
        previous_answer = turn.get("result", {}).get("answer", "")
        if previous_question:
            lines.append(f"Người dùng: {previous_question}")
        if previous_answer:
            lines.append(f"Trợ lý: {previous_answer[:700]}")

    lines.append(f"Câu hỏi hiện tại: {question}")
    lines.append("Hãy trả lời câu hỏi hiện tại dựa trên ngữ cảnh và nguồn truy xuất.")
    return "\n".join(lines)


def generate_chat_response(
    question: str,
    history: list[dict[str, Any]] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run citation-grounded generation with lightweight conversation memory."""
    retrieval_query = build_contextual_query(question, history=history)
    result = generate_with_citation(retrieval_query, top_k=top_k)
    result["display_question"] = question
    result["retrieval_query"] = retrieval_query
    return result
