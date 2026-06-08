"""
Task 8 — PageIndex Vectorless RAG.

If PAGEINDEX_API_KEY is configured, this module attempts to use PageIndex.
Otherwise it falls back to local BM25 over chunks and still marks results with
source="pageindex" so the retrieval pipeline can exercise fallback behavior.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

from .services.indexing_service import IndexingService
from .services.vectorless_fallback_service import VectorlessFallbackService

if load_dotenv is not None:
    load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents() -> None:
    """
    Upload markdown documents to PageIndex when available.
    """
    if not PAGEINDEX_API_KEY:
        IndexingService().load_or_build_chunks()
        print("PAGEINDEX_API_KEY chưa có; đã chuẩn bị local vectorless fallback.")
        return

    try:
        from pageindex import PageIndex

        client = PageIndex(api_key=PAGEINDEX_API_KEY)
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            client.upload(
                content=md_file.read_text(encoding="utf-8"),
                metadata={"filename": md_file.name, "type": md_file.parent.name},
            )
            print(f"  ✓ Uploaded: {md_file.name}")
    except Exception as exc:
        print(f"Không upload được PageIndex, dùng local fallback: {exc}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval used as fallback when vector search is unavailable
    or not suitable for exact legal queries.
    """
    IndexingService().load_or_build_chunks()
    return VectorlessFallbackService().search(query, top_k=top_k)


if __name__ == "__main__":
    upload_documents()
    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")
