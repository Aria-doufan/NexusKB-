"""Deterministic retrieval metric helpers for RAG evaluation."""

from __future__ import annotations

from collections import defaultdict
from math import log2
from numbers import Number
from typing import Any, Iterable


def _top_k(ranked_ids: Iterable[str], k: int) -> list[str]:
    """Return the first k non-empty ranked ids."""
    if k <= 0:
        return []

    return [ranked_id for ranked_id in ranked_ids if ranked_id][:k]


def precision_at_k(ranked_ids: Iterable[str], expected_ids: set[str], k: int) -> float:
    """Compute binary precision@k for ranked retrieval ids."""
    top_ids = _top_k(ranked_ids, k)
    if not top_ids or not expected_ids or k <= 0:
        return 0.0

    hits = len({ranked_id for ranked_id in top_ids if ranked_id in expected_ids})
    return hits / k


def recall_at_k(ranked_ids: Iterable[str], expected_ids: set[str], k: int) -> float:
    """Compute binary recall@k for ranked retrieval ids."""
    if not expected_ids:
        return 0.0

    top_ids = _top_k(ranked_ids, k)
    hits = len({ranked_id for ranked_id in top_ids if ranked_id in expected_ids})
    return hits / len(expected_ids)


def reciprocal_rank_at_k(ranked_ids: Iterable[str], expected_ids: set[str], k: int) -> float:
    """Compute reciprocal rank@k from the first relevant retrieved id."""
    if not expected_ids:
        return 0.0

    for rank, ranked_id in enumerate(_top_k(ranked_ids, k), start=1):
        if ranked_id in expected_ids:
            return 1 / rank

    return 0.0


def average_precision_at_k(ranked_ids: Iterable[str], expected_ids: set[str], k: int) -> float:
    """Compute average precision@k over first-seen relevant hits."""
    if not expected_ids or k <= 0:
        return 0.0

    seen_relevant: set[str] = set()
    precision_sum = 0.0
    for rank, ranked_id in enumerate(_top_k(ranked_ids, k), start=1):
        if ranked_id in expected_ids and ranked_id not in seen_relevant:
            seen_relevant.add(ranked_id)
            precision_sum += len(seen_relevant) / rank

    return precision_sum / min(len(expected_ids), k)


def ndcg_at_k(ranked_ids: Iterable[str], expected_ids: set[str], k: int) -> float:
    """Compute binary normalized discounted cumulative gain@k."""
    if not expected_ids or k <= 0:
        return 0.0

    seen_relevant: set[str] = set()
    dcg = 0.0
    for rank, ranked_id in enumerate(_top_k(ranked_ids, k), start=1):
        if ranked_id in expected_ids and ranked_id not in seen_relevant:
            seen_relevant.add(ranked_id)
            dcg += 1 / log2(rank + 1)

    ideal_relevant_count = min(len(expected_ids), k)
    idcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def average_numeric(rows: Iterable[dict[str, Any]], key: str) -> float:
    """Average numeric row values for key, returning 0.0 when absent."""
    values = [row[key] for row in rows if isinstance(row.get(key), Number)]
    if not values:
        return 0.0

    return sum(values) / len(values)


def build_question_type_summary(
    details: Iterable[dict[str, Any]], k_values: Iterable[int]
) -> dict[str, dict[str, float | int]]:
    """Average retrieval metrics grouped by question_type."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[str(row.get("question_type", "unknown"))].append(row)

    k_list = list(k_values)
    max_k = max(k_list) if k_list else 0
    summary: dict[str, dict[str, float | int]] = {}

    for question_type, rows in grouped.items():
        evidence_coverage_rows = [
            row for row in rows if row.get("required_evidence_groups_count", 0) > 0
        ]
        type_summary: dict[str, float | int] = {
            "questions": len(rows),
            "evidence_coverage_questions": len(evidence_coverage_rows),
            "average_latency_ms": average_numeric(rows, "latency_ms"),
            f"mrr@{max_k}": average_numeric(rows, f"rr@{max_k}"),
            f"map@{max_k}": average_numeric(rows, f"ap@{max_k}"),
        }

        for k in k_list:
            for metric in (
                "hit",
                "precision",
                "recall",
                "f1",
                "ndcg",
                "ap",
            ):
                type_summary[f"{metric}@{k}"] = average_numeric(rows, f"{metric}@{k}")
            type_summary[f"map@{k}"] = average_numeric(rows, f"ap@{k}")
            type_summary[f"evidence_coverage@{k}"] = average_numeric(
                evidence_coverage_rows, f"evidence_coverage@{k}"
            )

        summary[question_type] = type_summary

    return summary
