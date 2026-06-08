"""Safety rules for drug-related RAG answers."""

from __future__ import annotations

from .text_utils import normalize_for_search


class SafetyService:
    """Block requests that ask for illegal drug facilitation."""

    ILLEGAL_PATTERNS = (
        "cach san xuat",
        "cach dieu che",
        "cong thuc",
        "mua ma tuy",
        "ban ma tuy",
        "van chuyen ma tuy",
        "che tao ma tuy",
        "trong can sa",
        "qua mat cong an",
        "ne tranh phap luat",
    )

    def is_disallowed(self, query: str) -> bool:
        normalized = normalize_for_search(query)
        return any(pattern in normalized for pattern in self.ILLEGAL_PATTERNS)

    def refusal(self) -> str:
        return (
            "Tôi không thể hỗ trợ hướng dẫn sản xuất, mua bán, vận chuyển, "
            "che giấu hoặc né tránh xử lý pháp luật liên quan đến ma túy. "
            "Tôi có thể hỗ trợ thông tin pháp luật, phòng chống, cai nghiện "
            "và tác hại dựa trên nguồn được cung cấp."
        )

