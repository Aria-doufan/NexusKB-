"""Evaluate EnterpriseRAG-Bench generation records with RAGAS judges."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.enterprise_rag_graph import EnterpriseRagGraph
from app.schemas.rag import RagResponse, RagState
from scripts.evaluate_enterprise_hybrid_retrieval import DEFAULT_QUESTIONS_PATH, Question, load_questions
from scripts.rag_eval_reporting import load_json_if_exists, short_git_commit, utc_run_id

DEFAULT_OUTPUT_ROOT = BACKEND_DIR / "data" / "eval_outputs" / "generation"
DEFAULT_BASELINE_DIR = BACKEND_DIR / "data" / "eval_baselines" / "generation" / "current"
RAGAS_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
]


class CapturingTraceStore:
    """Trace store test double that keeps traces in memory for record assembly."""

    def __init__(self) -> None:
        self.saved: list[Any] = []

    async def save(self, trace: Any) -> None:
        self.saved.append(trace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble EnterpriseRAG-Bench generation evaluation records.")
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--judge-provider", default=os.getenv("RAGAS_JUDGE_PROVIDER", "openai"))
    parser.add_argument("--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o"))
    parser.add_argument("--ci", action="store_true", help="Record CI mode in config without enforcing regressions yet.")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Reserved for future generation metric threshold enforcement; no-op in this version.",
    )
    return parser.parse_args()


def normalize_generation_rag_intent(question_type: str) -> str:
    mapping = {
        "semantic": "semantic_query",
        "semantic_query": "semantic_query",
        "multi_hop": "multi_hop",
        "comparison": "comparison",
        "conflicting_info": "comparison",
        "project_related": "multi_hop",
        "procedure": "procedure",
        "constrained": "constrained",
        "completeness": "constrained",
        "high_level": "semantic_query",
        "fact_lookup": "fact_lookup",
        "lookup": "fact_lookup",
    }
    return mapping.get((question_type or "").strip(), question_type or "unknown")


def build_rag_state(question: Question) -> RagState:
    suffix = uuid4().hex[:12]
    return RagState(
        request_id=f"rag-generation-eval-{question.question_id}-{suffix}",
        debug_id=f"rag-generation-eval-debug-{question.question_id}-{suffix}",
        session_id="rag-eval",
        user_id="rag-eval-user",
        original_query=question.question,
        current_query=question.question,
        rag_intent=normalize_generation_rag_intent(question.question_type),
        source_hints=question.source_types,
        router_confidence=0.9,
        router_reason="EnterpriseRAG-Bench generation evaluation fixture.",
    )


def final_trace_documents(trace: Any, response: RagResponse) -> list[Any]:
    by_chunk_or_source_id: dict[str, Any] = {}
    by_parent_doc_id: dict[str, Any] = {}
    for document in _trace_selected_documents(trace):
        for field_name in ("parent_chunk_id", "source_id", "candidate_id", "id"):
            value = (getattr(document, field_name, None) or "").strip()
            if value and value not in by_chunk_or_source_id:
                by_chunk_or_source_id[value] = document

        parent_doc_id = (getattr(document, "parent_doc_id", None) or "").strip()
        if parent_doc_id and parent_doc_id not in by_parent_doc_id:
            by_parent_doc_id[parent_doc_id] = document

    documents: list[Any] = []
    seen_document_ids: set[int] = set()
    for source in response.sources or []:
        parent_chunk_id = (getattr(source, "parent_chunk_id", None) or "").strip()
        source_id = (getattr(source, "source_id", None) or "").strip()
        parent_doc_id = (getattr(source, "parent_doc_id", None) or "").strip()

        document = None
        for exact_id in (parent_chunk_id, source_id):
            if exact_id:
                document = by_chunk_or_source_id.get(exact_id)
                if document is not None:
                    break
        if document is None and not parent_chunk_id and not source_id and parent_doc_id:
            document = by_parent_doc_id.get(parent_doc_id)

        if document is not None and id(document) not in seen_document_ids:
            documents.append(document)
            seen_document_ids.add(id(document))
    return documents


def trace_contexts(trace: Any, response: RagResponse) -> list[str]:
    contexts: list[str] = []
    seen: set[str] = set()
    for document in final_trace_documents(trace, response):
        text = (getattr(document, "text", None) or getattr(document, "child_text", None) or "").strip()
        if text and text not in seen:
            contexts.append(text)
            seen.add(text)
    return contexts


def trace_source_ids(trace: Any, response: RagResponse) -> tuple[list[str], list[str]]:
    doc_ids: list[str] = []
    chunk_ids: list[str] = []
    seen_doc_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()
    for document in final_trace_documents(trace, response):
        doc_id = (getattr(document, "parent_doc_id", None) or "").strip()
        chunk_id = (getattr(document, "parent_chunk_id", None) or "").strip()
        if doc_id and doc_id not in seen_doc_ids:
            doc_ids.append(doc_id)
            seen_doc_ids.add(doc_id)
        if chunk_id and chunk_id not in seen_chunk_ids:
            chunk_ids.append(chunk_id)
            seen_chunk_ids.add(chunk_id)
    return doc_ids, chunk_ids


def response_to_generation_record(
    question: Question,
    response: RagResponse,
    trace: Any,
    latency_ms: float,
) -> dict[str, Any]:
    source_doc_ids, source_chunk_ids = trace_source_ids(trace, response)
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "reference": question.gold_answer,
        "answer": response.answer,
        "retrieved_contexts": trace_contexts(trace, response),
        "source_doc_ids": source_doc_ids,
        "source_chunk_ids": source_chunk_ids,
        "latency_ms": round(float(latency_ms), 4),
    }


def build_ragas_sample_dict(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_input": record["question"],
        "retrieved_contexts": record["retrieved_contexts"],
        "response": record["answer"],
        "reference": record["reference"],
    }


def apply_ragas_scores(records: list[dict[str, Any]], evaluator) -> list[dict[str, Any]]:
    samples = [build_ragas_sample_dict(record) for record in records if not record.get("generation_error")]
    sample_indexes = [index for index, record in enumerate(records) if not record.get("generation_error")]
    try:
        results = evaluator(samples)
    except Exception as exc:
        for index in sample_indexes:
            records[index]["ragas_error"] = str(exc)
        return records

    for index, result in zip(sample_indexes, results):
        if isinstance(result, Exception):
            records[index]["ragas_error"] = str(result)
        else:
            records[index]["ragas_scores"] = {
                key: float(value)
                for key, value in dict(result).items()
                if value is not None
            }
    return records


def summarize_generation_records(records: list[dict[str, Any]], intended_count: int) -> dict[str, Any]:
    valid_records = [record for record in records if isinstance(record.get("ragas_scores"), dict)]
    latencies = [float(record["latency_ms"]) for record in records if isinstance(record.get("latency_ms"), int | float)]
    valid_ratio = len(valid_records) / intended_count if intended_count else 1.0
    summary: dict[str, Any] = {
        "questions": len(records),
        "intended_questions": intended_count,
        "valid_ragas_scores": len(valid_records),
        "status": "complete" if valid_ratio >= 0.8 else "incomplete",
        "average_latency_ms": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
    }

    for metric_name in RAGAS_METRIC_NAMES:
        values = []
        for record in valid_records:
            value = record["ragas_scores"].get(metric_name)
            if isinstance(value, int | float):
                values.append(float(value))
        if values:
            summary[metric_name] = round(sum(values) / len(values), 4)
    return summary


async def generate_records(questions: list[Question]) -> list[dict[str, Any]]:
    trace_store = CapturingTraceStore()
    graph = EnterpriseRagGraph(trace_store=trace_store)
    records: list[dict[str, Any]] = []

    for question in questions:
        started = perf_counter()
        try:
            response = await graph.run(build_rag_state(question))
            latency_ms = (perf_counter() - started) * 1000
            trace = trace_store.saved[-1] if trace_store.saved else None
            records.append(response_to_generation_record(question, response, trace, latency_ms=latency_ms))
        except Exception as exc:
            latency_ms = (perf_counter() - started) * 1000
            records.append(
                {
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "question": question.question,
                    "reference": question.gold_answer,
                    "answer": "",
                    "retrieved_contexts": [],
                    "source_doc_ids": [],
                    "source_chunk_ids": [],
                    "latency_ms": round(latency_ms, 4),
                    "generation_error": str(exc),
                }
            )
    return records


def _build_ragas_evaluation_components(judge_model: str):
    from datasets import Dataset
    from langchain_openai.chat_models import ChatOpenAI
    from langchain_openai.embeddings import OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]
    return Dataset, evaluate, metrics, llm


def run_ragas_evaluation(records: list[dict[str, Any]], judge_provider: str, judge_model: str) -> list[dict[str, Any]]:
    if judge_provider != "openai":
        raise ValueError(f"Unsupported RAGAS judge provider for this version: {judge_provider}")

    def evaluator(samples: list[dict[str, Any]]) -> list[dict[str, float]]:
        Dataset, evaluate, metrics, llm = _build_ragas_evaluation_components(judge_model)

        dataset = Dataset.from_list(samples)
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
        )
        dataframe = result.to_pandas()
        return [
            {
                metric: float(row[metric])
                for metric in RAGAS_METRIC_NAMES
                if metric in row and row[metric] == row[metric]
            }
            for row in dataframe.to_dict(orient="records")
        ]

    return apply_ragas_scores(records, evaluator)


def render_generation_report(
    config: dict[str, Any],
    summary: dict[str, Any],
    baseline_summary: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# RAG Generation Evaluation Report",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Summary",
        "",
        "| Metric | Current | Baseline |",
        "| --- | ---: | ---: |",
    ]
    for metric_name in ["questions", "valid_ragas_scores", "average_latency_ms", *RAGAS_METRIC_NAMES]:
        current = summary.get(metric_name, "")
        baseline = "" if baseline_summary is None else baseline_summary.get(metric_name, "")
        lines.append(f"| {metric_name} | {current} | {baseline} |")
    return "\n".join(lines) + "\n"


def write_generation_outputs(
    output_root: Path,
    baseline_dir: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    run_dir = output_root / utc_run_id("generation")
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline_summary = load_json_if_exists(baseline_dir / "generation_ragas_summary.json")
    report = render_generation_report(config, summary, baseline_summary)
    failures = [record for record in records if record.get("generation_error") or record.get("ragas_error")]

    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "generation_ragas_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "generation_ragas_details.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (run_dir / "generation_ragas_failures.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in failures),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return run_dir


async def async_main() -> int:
    args = parse_args()
    questions = load_questions(args.questions_path, limit=args.limit)
    records = await generate_records(questions)
    records = run_ragas_evaluation(records, args.judge_provider, args.judge_model)
    summary = summarize_generation_records(records, intended_count=len(questions))
    config = {
        "questions_path": str(args.questions_path),
        "limit": args.limit,
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "ci": args.ci,
        "fail_on_regression": args.fail_on_regression,
        "git_commit": short_git_commit(),
        "ragas_status": "wired",
    }
    run_dir = write_generation_outputs(args.output_root, args.baseline_dir, config, records, summary)
    print(f"Wrote generation evaluation records to {run_dir}")
    return 0


def main() -> int:
    return asyncio.run(async_main())


def _trace_selected_documents(trace: Any) -> list[Any]:
    if trace is None:
        return []

    documents: list[Any] = []
    for retrieval_attempt in getattr(trace, "retrieval_attempts", []) or []:
        attempt = getattr(retrieval_attempt, "attempt", retrieval_attempt)
        documents.extend(getattr(attempt, "selected_documents", []) or [])
    return documents


if __name__ == "__main__":
    raise SystemExit(main())
