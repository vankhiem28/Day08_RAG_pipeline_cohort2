"""Offline RAG evaluation for the group chatbot.

The class assignment suggests DeepEval/RAGAS/TruLens. For a classroom demo that
must run without external LLM credentials, this script uses deterministic
token-overlap metrics with the same four evaluation dimensions:
faithfulness, answer relevance, context recall, and context precision.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.md"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.citation_service import CitationService
from src.services.llm_service import LLMService
from src.services.prompt_builder_service import PromptBuilderService
from src.services.text_utils import tokenize
from src.task10_generation import reorder_for_llm
from src.task9_retrieval_pipeline import retrieve


STOPWORDS = {
    "a",
    "an",
    "and",
    "bi",
    "bị",
    "của",
    "các",
    "cái",
    "cho",
    "chất",
    "có",
    "do",
    "duoc",
    "được",
    "gì",
    "hay",
    "hoặc",
    "khi",
    "la",
    "là",
    "ma",
    "ma túy",
    "ma tuý",
    "mot",
    "một",
    "những",
    "nào",
    "the",
    "theo",
    "trong",
    "tu",
    "từ",
    "va",
    "và",
    "ve",
    "về",
}


@dataclass
class EvalCaseResult:
    question: str
    expected_context: str
    answer: str
    sources: list[dict[str, Any]]
    metrics: dict[str, float]

    @property
    def average(self) -> float:
        return statistics.mean(self.metrics.values())


def load_golden_dataset() -> list[dict[str, str]]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def meaningful_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 1 and token not in STOPWORDS}


def overlap_ratio(needle: str, haystack: str) -> float:
    needle_tokens = meaningful_tokens(needle)
    if not needle_tokens:
        return 0.0
    haystack_tokens = meaningful_tokens(haystack)
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)


def precision_ratio(needle: str, haystack: str) -> float:
    haystack_tokens = meaningful_tokens(haystack)
    if not haystack_tokens:
        return 0.0
    needle_tokens = meaningful_tokens(needle)
    return len(needle_tokens & haystack_tokens) / len(haystack_tokens)


def run_rag(question: str, use_reranking: bool, top_k: int = 5) -> dict[str, Any]:
    chunks = retrieve(question, top_k=top_k, use_reranking=use_reranking)
    reordered = reorder_for_llm(chunks)
    cited_chunks = CitationService().attach_citation_ids(reordered)
    answer = LLMService(PromptBuilderService()).generate(question, cited_chunks)
    return {
        "answer": answer,
        "sources": cited_chunks,
        "retrieval_source": cited_chunks[0].get("source", "none") if cited_chunks else "none",
    }


def compute_metrics(item: dict[str, str], result: dict[str, Any]) -> dict[str, float]:
    answer = result["answer"]
    source_texts = [source.get("content", "") for source in result.get("sources", [])]
    joined_context = "\n".join(source_texts)
    expected_answer = item["expected_answer"]
    expected_context = item["expected_context"]
    question = item["question"]

    answer_context_overlap = overlap_ratio(answer, joined_context)
    citation_bonus = 0.15 if "[" in answer and "]" in answer else 0.0
    faithfulness = min(1.0, answer_context_overlap + citation_bonus)

    answer_relevance = min(
        1.0,
        0.55 * overlap_ratio(question, answer)
        + 0.45 * overlap_ratio(expected_answer, answer),
    )

    context_recall = min(
        1.0,
        0.65 * overlap_ratio(expected_answer, joined_context)
        + 0.35 * overlap_ratio(expected_context, joined_context),
    )

    relevance_basis = f"{question} {expected_answer} {expected_context}"
    useful_contexts = [
        context
        for context in source_texts
        if overlap_ratio(relevance_basis, context) >= 0.12
        or precision_ratio(relevance_basis, context) >= 0.08
    ]
    context_precision = len(useful_contexts) / len(source_texts) if source_texts else 0.0

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevance": round(answer_relevance, 4),
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4),
    }


def evaluate_config(
    name: str,
    golden_dataset: list[dict[str, str]],
    *,
    use_reranking: bool,
) -> dict[str, Any]:
    cases: list[EvalCaseResult] = []
    for item in golden_dataset:
        result = run_rag(item["question"], use_reranking=use_reranking)
        metrics = compute_metrics(item, result)
        cases.append(
            EvalCaseResult(
                question=item["question"],
                expected_context=item["expected_context"],
                answer=result["answer"],
                sources=result["sources"],
                metrics=metrics,
            )
        )

    metric_names = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    averages = {
        metric: round(statistics.mean(case.metrics[metric] for case in cases), 4)
        for metric in metric_names
    }
    averages["average"] = round(statistics.mean(averages.values()), 4)
    return {"name": name, "averages": averages, "cases": cases}


def compare_configs(golden_dataset: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "hybrid_rerank": evaluate_config(
            "hybrid_rerank",
            golden_dataset,
            use_reranking=True,
        ),
        "hybrid_no_rerank": evaluate_config(
            "hybrid_no_rerank",
            golden_dataset,
            use_reranking=False,
        ),
    }


def evaluate_with_deepeval(golden_dataset: list[dict[str, str]]) -> Any:
    """Optional DeepEval run for environments with LLM credentials configured."""
    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    test_cases = []
    for item in golden_dataset:
        result = run_rag(item["question"], use_reranking=True)
        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=result["answer"],
                expected_output=item["expected_answer"],
                retrieval_context=[source.get("content", "") for source in result["sources"]],
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
    ]
    return evaluate(test_cases, metrics)


def failure_stage(metrics: dict[str, float]) -> str:
    if metrics["context_recall"] < 0.45:
        return "retrieval_recall"
    if metrics["context_precision"] < 0.45:
        return "retrieval_precision"
    if metrics["faithfulness"] < 0.65:
        return "generation_grounding"
    if metrics["answer_relevance"] < 0.45:
        return "answer_relevance"
    return "mixed"


def source_summary(case: EvalCaseResult) -> str:
    labels = []
    for source in case.sources[:3]:
        metadata = source.get("metadata", {})
        labels.append(metadata.get("source") or metadata.get("title") or "unknown")
    return ", ".join(labels) if labels else "none"


def export_results(comparison: dict[str, Any]) -> None:
    config_a = comparison["hybrid_rerank"]
    config_b = comparison["hybrid_no_rerank"]
    avg_a = config_a["averages"]
    avg_b = config_b["averages"]

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "average": "Average",
    }

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework sử dụng",
        "",
        "Default run: offline heuristic evaluator with deterministic token-overlap metrics mapped to the four required RAG dimensions. Optional run: set `RAG_EVAL_FRAMEWORK=deepeval` to execute DeepEval metrics when LLM credentials are available.",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Delta |",
        "|--------|-----------------------------|------------------------------|-------|",
    ]

    for metric, label in metric_labels.items():
        delta = avg_a[metric] - avg_b[metric]
        lines.append(f"| {label} | {avg_a[metric]:.3f} | {avg_b[metric]:.3f} | {delta:+.3f} |")

    best = "Config A" if avg_a["average"] >= avg_b["average"] else "Config B"
    lines.extend(
        [
            "",
            "## A/B Comparison Analysis",
            "",
            "**Config A:** hybrid retrieval with RRF merge and local lexical reranking.",
            "",
            "**Config B:** hybrid retrieval with RRF merge, without the final reranking pass.",
            "",
            f"**Kết luận:** {best} có điểm trung bình cao hơn trong lần chạy này. Nếu điểm recall thấp, ưu tiên cải thiện chunking/metadata; nếu precision thấp, ưu tiên reranking và lọc nhiễu từ HTML crawl.",
            "",
            "## Worst Performers (Bottom 3)",
            "",
            "| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Top Sources |",
            "|---|----------|--------------|-----------|--------|-----------|---------------|-------------|",
        ]
    )

    worst_cases = sorted(config_a["cases"], key=lambda case: case.average)[:3]
    for index, case in enumerate(worst_cases, 1):
        metrics = case.metrics
        question = case.question.replace("|", "\\|")
        lines.append(
            f"| {index} | {question} | {metrics['faithfulness']:.3f} | "
            f"{metrics['answer_relevance']:.3f} | {metrics['context_recall']:.3f} | "
            f"{metrics['context_precision']:.3f} | {failure_stage(metrics)} | "
            f"{source_summary(case).replace('|', '/') } |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "### Cải tiến 1",
            "**Action:** Làm sạch markdown báo chí sau crawl để bỏ menu, footer, quảng cáo và link liên quan.",
            "**Expected impact:** Tăng context precision và giảm câu trả lời trích nhầm nội dung nhiễu.",
            "",
            "### Cải tiến 2",
            "**Action:** Bổ sung metadata thủ công cho văn bản pháp luật: số điều, chương, tên văn bản, năm ban hành.",
            "**Expected impact:** Tăng context recall cho câu hỏi pháp luật có trích điều/khoản.",
            "",
            "### Cải tiến 3",
            "**Action:** Thử reranker thật như Jina hoặc cross-encoder multilingual khi có API/GPU.",
            "**Expected impact:** Cải thiện thứ tự source chunks cho câu hỏi tin tức nhiều thực thể.",
            "",
        ]
    )

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    golden_dataset = load_golden_dataset()
    if os.getenv("RAG_EVAL_FRAMEWORK", "").lower() == "deepeval":
        print("Running optional DeepEval evaluation...")
        evaluate_with_deepeval(golden_dataset)

    comparison = compare_configs(golden_dataset)
    export_results(comparison)
    print(f"Loaded {len(golden_dataset)} test cases")
    print(f"Results written to {RESULTS_PATH}")
    for name, payload in comparison.items():
        print(name, payload["averages"])


if __name__ == "__main__":
    main()
