"""
Task 10 — Generation Có Citation.

The module can call OpenAI when OPENAI_API_KEY is present. Without an API key,
it returns an extractive answer grounded in retrieved chunks, with citations.
"""

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

from .services.citation_service import CitationService
from .services.llm_service import LLMService
from .services.logging_service import LoggingService
from .services.prompt_builder_service import PromptBuilderService
from .services.safety_service import SafetyService
from .task9_retrieval_pipeline import retrieve

if load_dotenv is not None:
    load_dotenv()


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = PromptBuilderService.SYSTEM_PROMPT


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Place strongest evidence at the beginning and near the end to reduce
    lost-in-the-middle.
    """
    if len(chunks) <= 2:
        return chunks
    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])
    start = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for i in range(start, 0, -2):
        reordered.append(chunks[i])
    return reordered


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks with stable citation ids.
    """
    cited_chunks = CitationService().attach_citation_ids(chunks)
    return PromptBuilderService().format_context(cited_chunks)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.
    """
    safety = SafetyService()
    citation_service = CitationService()
    logging_service = LoggingService()

    if safety.is_disallowed(query):
        answer = safety.refusal()
        logging_service.log_generation(
            {
                "question": query,
                "retrieval_mode": "blocked_by_safety",
                "retrieved_chunk_ids": [],
                "scores": [],
                "final_answer": answer,
                "citations": [],
                "fallback": False,
            }
        )
        return {"answer": answer, "sources": [], "retrieval_source": "safety", "citations": []}

    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    cited_chunks = citation_service.attach_citation_ids(reordered)
    answer = LLMService(PromptBuilderService(citation_service)).generate(
        query=query,
        chunks=cited_chunks,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    citations = citation_service.source_list(cited_chunks)
    retrieval_source = cited_chunks[0].get("source", "none") if cited_chunks else "none"
    fallback = retrieval_source == "pageindex"

    logging_service.log_generation(
        {
            "question": query,
            "retrieval_mode": retrieval_source,
            "retrieved_chunk_ids": [
                chunk.get("metadata", {}).get("chunk_id") for chunk in cited_chunks
            ],
            "scores": [float(chunk.get("score", 0.0)) for chunk in cited_chunks],
            "final_answer": answer,
            "citations": citations,
            "fallback": fallback,
        }
    )
    return {
        "answer": answer,
        "sources": cited_chunks,
        "retrieval_source": retrieval_source,
        "citations": citations,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
