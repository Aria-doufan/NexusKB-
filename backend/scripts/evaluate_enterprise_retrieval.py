"""Evaluate EnterpriseRAG-Bench retrieval against expected document ids.

Baseline 0 uses only Chroma vector search over child chunks and evaluates
returned `parent_doc_id` values against the benchmark's `expected_doc_ids`.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "questions.jsonl"
DEFAULT_PERSIST_DIR = BACKEND_DIR / "data" / "chromadb_enterprise_parent_child"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "enterprise_rag_bench" / "eval"
DEFAULT_COLLECTION_NAME = "enterprise_rag_bench_parent_child"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_K_VALUES = [1, 5, 10, 20]


@dataclass(slots=True)
class Question:
    question_id: str
    question_type: str
    source_types: list[str]
    question: str
    expected_doc_ids: list[str]
    gold_answer: str
    answer_facts: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EnterpriseRAG-Bench retrieval.")
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--search-k",
        type=int,
        default=50,
        help="Number of child chunks fetched from Chroma before parent_doc_id dedup.",
    )
    parser.add_argument(
        "--k-values",
        default="1,5,10,20",
        help="Comma-separated K values for Recall@K and Hit@K after parent_doc_id dedup.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--where-source-type",
        default=None,
        help="Optional single source_type metadata filter. Baseline keeps this empty.",
    )
    return parser.parse_args()


def load_questions(path: Path, limit: int | None = None) -> list[Question]:
    questions: list[Question] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if limit is not None and len(questions) >= limit:
                break
            row = json.loads(line)
            questions.append(
                Question(
                    question_id=row["question_id"],
                    question_type=row.get("question_type", ""),
                    source_types=list(row.get("source_types") or []),
                    question=row["question"],
                    expected_doc_ids=list(row.get("expected_doc_ids") or []),
                    gold_answer=row.get("gold_answer", ""),
                    answer_facts=list(row.get("answer_facts") or []),
                )
            )
    return questions


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--k-values must contain positive integers")
    return values


def unique_parent_doc_ids(results: list[tuple[Any, float]]) -> tuple[list[str], list[dict[str, Any]]]:
    seen: set[str] = set()
    doc_ids: list[str] = []
    hits: list[dict[str, Any]] = []

    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata
        parent_doc_id = metadata.get("parent_doc_id", "")
        if not parent_doc_id or parent_doc_id in seen:
            continue
        seen.add(parent_doc_id)
        doc_ids.append(parent_doc_id)
        hits.append(
            {
                "rank": len(doc_ids),
                "raw_child_rank": rank,
                "score": score,
                "parent_doc_id": parent_doc_id,
                "parent_chunk_id": metadata.get("parent_chunk_id", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "source_type": metadata.get("source_type", ""),
                "title": metadata.get("title", ""),
                "section_heading": metadata.get("section_heading", ""),
                "preview": document.page_content[:240].replace("\n", " "),
            }
        )
    return doc_ids, hits


def reciprocal_rank(ranked_doc_ids: list[str], expected_doc_ids: set[str], max_k: int) -> float:
    for index, doc_id in enumerate(ranked_doc_ids[:max_k], start=1):
        if doc_id in expected_doc_ids:
            return 1.0 / index
    return 0.0


def evaluate_question(
    store: Chroma,
    question: Question,
    search_k: int,
    k_values: list[int],
    where: dict[str, str] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    results = store.similarity_search_with_score(
        question.question,
        k=search_k,
        filter=where,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    ranked_doc_ids, hits = unique_parent_doc_ids(results)
    expected = set(question.expected_doc_ids)

    detail: dict[str, Any] = {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "source_types": question.source_types,
        "question": question.question,
        "expected_doc_ids": question.expected_doc_ids,
        "retrieved_parent_doc_ids": ranked_doc_ids,
        "top_hits": hits[: max(k_values)],
        "latency_ms": elapsed_ms,
        "raw_child_results": len(results),
        "dedup_parent_results": len(ranked_doc_ids),
    }

    for k in k_values:
        retrieved_at_k = set(ranked_doc_ids[:k])
        matched = sorted(expected.intersection(retrieved_at_k))
        detail[f"hit@{k}"] = 1 if matched else 0
        detail[f"recall@{k}"] = len(matched) / max(len(expected), 1)
        detail[f"matched_doc_ids@{k}"] = matched

    max_mrr_k = max(k_values)
    detail[f"rr@{max_mrr_k}"] = reciprocal_rank(ranked_doc_ids, expected, max_mrr_k)
    return detail


def summarize(details: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    total = len(details)
    summary: dict[str, Any] = {
        "questions": total,
        "k_values": k_values,
        "average_latency_ms": round(
            sum(item["latency_ms"] for item in details) / max(total, 1),
            2,
        ),
    }

    for k in k_values:
        summary[f"hit@{k}"] = round(
            sum(item[f"hit@{k}"] for item in details) / max(total, 1),
            4,
        )
        summary[f"recall@{k}"] = round(
            sum(item[f"recall@{k}"] for item in details) / max(total, 1),
            4,
        )

    max_mrr_k = max(k_values)
    summary[f"mrr@{max_mrr_k}"] = round(
        sum(item[f"rr@{max_mrr_k}"] for item in details) / max(total, 1),
        4,
    )
    return summary


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    k_values: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "baseline_chroma_child_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    details_path = output_dir / "baseline_chroma_child_details.jsonl"
    with details_path.open("w", encoding="utf-8") as file:
        for row in details:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / "baseline_chroma_child_details.csv"
    fieldnames = [
        "question_id",
        "question_type",
        "source_types",
        "expected_doc_ids",
        "top_parent_doc_ids",
        "latency_ms",
        "dedup_parent_results",
    ]
    for k in k_values:
        fieldnames.extend([f"hit@{k}", f"recall@{k}", f"matched_doc_ids@{k}"])
    fieldnames.append(f"rr@{max(k_values)}")

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            csv_row = {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "source_types": "|".join(row["source_types"]),
                "expected_doc_ids": "|".join(row["expected_doc_ids"]),
                "top_parent_doc_ids": "|".join(row["retrieved_parent_doc_ids"][: max(k_values)]),
                "latency_ms": row["latency_ms"],
                "dedup_parent_results": row["dedup_parent_results"],
                f"rr@{max(k_values)}": row[f"rr@{max(k_values)}"],
            }
            for k in k_values:
                csv_row[f"hit@{k}"] = row[f"hit@{k}"]
                csv_row[f"recall@{k}"] = row[f"recall@{k}"]
                csv_row[f"matched_doc_ids@{k}"] = "|".join(row[f"matched_doc_ids@{k}"])
            writer.writerow(csv_row)


def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.k_values)
    questions = load_questions(args.questions_path, args.limit)
    if not questions:
        raise ValueError(f"No questions loaded from {args.questions_path}")

    embeddings = OllamaEmbeddings(
        model=args.embedding_model,
        base_url=args.ollama_base_url,
    )
    store = Chroma(
        collection_name=args.collection_name,
        embedding_function=embeddings,
        persist_directory=str(args.persist_dir.resolve()),
    )
    where = {"source_type": args.where_source_type} if args.where_source_type else None

    started = time.perf_counter()
    details: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        detail = evaluate_question(
            store=store,
            question=question,
            search_k=args.search_k,
            k_values=k_values,
            where=where,
        )
        details.append(detail)
        print(
            f"[{index}/{len(questions)}] {question.question_id} "
            f"hit@{max(k_values)}={detail[f'hit@{max(k_values)}']} "
            f"latency_ms={detail['latency_ms']}",
            flush=True,
        )

    summary = summarize(details, k_values)
    summary.update(
        {
            "method": "baseline_chroma_child",
            "collection_name": args.collection_name,
            "persist_dir": str(args.persist_dir.resolve()),
            "embedding_model": args.embedding_model,
            "search_k_child_chunks": args.search_k,
            "where_source_type": args.where_source_type,
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }
    )
    write_outputs(args.output_dir, summary, details, k_values)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
