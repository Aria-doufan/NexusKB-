"""Reporting helpers for RAG evaluation runs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def short_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def utc_run_id(prefix: str = "") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    commit = short_git_commit()
    if prefix:
        return f"{timestamp}-{prefix}-{commit}"
    return f"{timestamp}-{commit}"


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_delta(current: dict[str, Any], baseline: dict[str, Any] | None, metric: str) -> float | None:
    if baseline is None or metric not in current or metric not in baseline:
        return None
    current_value = current.get(metric)
    baseline_value = baseline.get(metric)
    if not isinstance(current_value, int | float) or not isinstance(baseline_value, int | float):
        return None
    return round(float(current_value) - float(baseline_value), 4)


def render_metric_table(
    summary: dict[str, Any],
    metrics: list[str],
    baseline: dict[str, Any] | None = None,
) -> str:
    lines = ["| Metric | Current | Baseline | Delta |", "| --- | ---: | ---: | ---: |"]
    for metric in metrics:
        current_value = summary.get(metric, "")
        baseline_value = "" if baseline is None else baseline.get(metric, "")
        delta = metric_delta(summary, baseline, metric)
        lines.append(
            f"| {metric} | {_format_value(current_value)} | {_format_value(baseline_value)} | {_format_value(delta)} |"
        )
    return "\n".join(lines)


def render_retrieval_report(
    config: dict[str, Any],
    summary: dict[str, Any],
    failures: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
) -> str:
    k_values = summary.get("k_values") or config.get("k_values") or []
    metrics = ["questions", "average_latency_ms", "evidence_coverage"]
    for k in k_values:
        metrics.extend([f"hit@{k}", f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"map@{k}", f"evidence_coverage@{k}"])
    if k_values:
        metrics.append(f"mrr@{max(k_values)}")

    question_type_summary = summary.get("question_type_summary") or {}
    baseline_question_types = (baseline_summary or {}).get("question_type_summary") or {}
    doc_semantic_type_summary = summary.get("doc_semantic_type_summary") or {}
    baseline_doc_semantic_types = (baseline_summary or {}).get("doc_semantic_type_summary") or {}
    question_type_metrics = [metric for metric in metrics if metric != "average_latency_ms"]

    sections = [
        "# Retrieval Evaluation Report",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Summary",
        "",
        render_metric_table(summary, metrics, baseline_summary),
        "",
        "## Question Type Summary",
        "",
        _render_group_summary_table("Question Type", question_type_summary, question_type_metrics, baseline_question_types),
        "",
        "## Document Semantic Type Summary",
        "",
        _render_group_summary_table(
            "Document Semantic Type",
            doc_semantic_type_summary,
            question_type_metrics,
            baseline_doc_semantic_types,
        ),
        "",
        "## Failure Examples",
        "",
        _render_failure_examples(failures[:10]),
    ]
    return "\n".join(sections) + "\n"


def _render_group_summary_table(
    group_label: str,
    group_summary: dict[str, Any],
    metrics: list[str],
    baseline_groups: dict[str, Any],
) -> str:
    if not group_summary:
        return f"No {group_label.lower()} summary available."

    lines = [
        f"| {group_label} | Metric | Current | Baseline | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for group in sorted(group_summary):
        row = group_summary[group]
        baseline_row = baseline_groups.get(group)
        for metric in metrics:
            if metric not in row:
                continue
            delta = metric_delta(row, baseline_row, metric)
            baseline_value = "" if baseline_row is None else baseline_row.get(metric, "")
            lines.append(
                f"| {_escape_table_cell(group)} | {_escape_table_cell(metric)} | "
                f"{_format_value(row.get(metric, ''))} | "
                f"{_format_value(baseline_value)} | {_format_value(delta)} |"
            )
    return "\n".join(lines)


def _render_failure_examples(failures: list[dict[str, Any]]) -> str:
    if not failures:
        return "No failure examples."

    lines = ["| Question ID | Question Type | Failure Reasons | Question |", "| --- | --- | --- | --- |"]
    for row in failures:
        reasons = row.get("failure_reasons") or []
        if isinstance(reasons, list):
            reasons_text = ", ".join(str(reason) for reason in reasons)
        else:
            reasons_text = str(reasons)
        lines.append(
            f"| {_escape_table_cell(row.get('question_id', ''))} | "
            f"{_escape_table_cell(row.get('question_type', ''))} | "
            f"{_escape_table_cell(reasons_text)} | "
            f"{_escape_table_cell(row.get('question', ''))} |"
        )
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(round(value, 4))
    return str(value)


def _escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
