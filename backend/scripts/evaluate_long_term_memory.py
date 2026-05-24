"""Evaluate long-term memory through the public FastAPI API.

The script runs golden cases end to end:
1. Send setup turns that should create long-term memories.
2. Wait for background extraction.
3. Search memories and score Recall-style term hits.
4. Ask a cross-session query and score answer term hits.
5. Optionally delete the matched memory and verify it no longer appears.

It writes summary JSON, detail JSONL, and detail CSV files so memory changes can
be compared across versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from jose import jwt


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = BACKEND_DIR / "scripts" / "memory_eval_golden_cases.jsonl"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "memory_eval" / "eval"


@dataclass(slots=True)
class MemoryEvalCase:
    case_id: str
    category: str
    setup_turns: list[str]
    search_query: str
    cross_session_query: str
    expected_memory_terms: list[str]
    expected_answer_terms: list[str]
    delete_after: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate long-term memory E2E.")
    parser.add_argument("--base-url", default=os.getenv("MEMORY_EVAL_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--token", default=os.getenv("MEMORY_EVAL_TOKEN"))
    parser.add_argument("--other-token", default=os.getenv("MEMORY_EVAL_OTHER_TOKEN"))
    parser.add_argument("--user-id", default=os.getenv("MEMORY_EVAL_USER_ID", "memory-eval-user"))
    parser.add_argument("--other-user-id", default=os.getenv("MEMORY_EVAL_OTHER_USER_ID", "memory-eval-other-user"))
    parser.add_argument("--secret-key", default=os.getenv("SECRET_KEY"))
    parser.add_argument("--algorithm", default=os.getenv("ALGORITHM", "HS256"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def load_cases(path: Path, limit: int | None = None) -> list[MemoryEvalCase]:
    cases: list[MemoryEvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if limit is not None and len(cases) >= limit:
                break
            row = json.loads(line)
            cases.append(
                MemoryEvalCase(
                    case_id=row["case_id"],
                    category=row.get("category", "other"),
                    setup_turns=list(row.get("setup_turns") or []),
                    search_query=row["search_query"],
                    cross_session_query=row["cross_session_query"],
                    expected_memory_terms=list(row.get("expected_memory_terms") or []),
                    expected_answer_terms=list(row.get("expected_answer_terms") or []),
                    delete_after=bool(row.get("delete_after", False)),
                )
            )
    return cases


def make_token(user_id: str, secret_key: str | None, algorithm: str) -> str:
    if not secret_key:
        raise ValueError("Provide --token or set SECRET_KEY so the script can generate a JWT")
    now = datetime.now(timezone.utc)
    payload = {
        "user_id": user_id,
        "jti": f"memory-eval-{uuid.uuid4()}",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=2)).timestamp()),
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def unwrap_response(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("code") == 200 and "data" in data:
        return data.get("data") or {}
    return data


def post_router_query(
    base_url: str,
    token: str,
    query: str,
    session_id: str,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/api/agent/router/query",
        headers=headers(token),
        json={"session_id": session_id, "query": query},
        timeout=timeout,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return unwrap_response(response), elapsed_ms


def search_memories(
    base_url: str,
    token: str,
    query: str,
    timeout: float,
    limit: int = 8,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    response = requests.get(
        f"{base_url.rstrip('/')}/api/memories/search",
        headers=headers(token),
        params={"q": query, "limit": limit},
        timeout=timeout,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    data = unwrap_response(response)
    return list(data.get("memories") or []), elapsed_ms


def delete_memory(base_url: str, token: str, memory_id: str, timeout: float) -> float:
    started = time.perf_counter()
    response = requests.delete(
        f"{base_url.rstrip('/')}/api/memories/{memory_id}",
        headers=headers(token),
        timeout=timeout,
    )
    response.raise_for_status()
    return round((time.perf_counter() - started) * 1000, 2)


def normalize(text: str) -> str:
    return text.lower().replace(" ", "")


def terms_hit(text: str, terms: list[str]) -> bool:
    normalized_text = normalize(text)
    return all(normalize(term) in normalized_text for term in terms)


def memory_hit(memories: list[dict[str, Any]], terms: list[str]) -> tuple[bool, dict[str, Any] | None]:
    for memory in memories:
        if terms_hit(str(memory.get("memory") or ""), terms):
            return True, memory
    return False, None


def evaluate_case(
    case: MemoryEvalCase,
    base_url: str,
    token: str,
    run_id: str,
    settle_seconds: float,
    timeout: float,
    other_token: str | None = None,
) -> dict[str, Any]:
    setup_session_id = f"memory-eval-{run_id}-{case.case_id}-setup"
    query_session_id = f"memory-eval-{run_id}-{case.case_id}-query"
    setup_results: list[dict[str, Any]] = []

    for turn_index, setup_query in enumerate(case.setup_turns, start=1):
        response, latency_ms = post_router_query(
            base_url=base_url,
            token=token,
            query=setup_query,
            session_id=setup_session_id,
            timeout=timeout,
        )
        setup_results.append(
            {
                "turn_index": turn_index,
                "query": setup_query,
                "route": response.get("route"),
                "latency_ms": latency_ms,
                "response_preview": str(response.get("response") or "")[:240],
                "error": response.get("error"),
            }
        )

    if settle_seconds > 0:
        time.sleep(settle_seconds)

    memories, search_latency_ms = search_memories(
        base_url=base_url,
        token=token,
        query=case.search_query,
        timeout=timeout,
    )
    search_pass, matched_memory = memory_hit(memories, case.expected_memory_terms)

    answer_response, answer_latency_ms = post_router_query(
        base_url=base_url,
        token=token,
        query=case.cross_session_query,
        session_id=query_session_id,
        timeout=timeout,
    )
    answer_text = str(answer_response.get("response") or "")
    answer_pass = terms_hit(answer_text, case.expected_answer_terms)

    delete_pass: bool | None = None
    delete_latency_ms: float | None = None
    post_delete_memories: list[dict[str, Any]] = []
    if case.delete_after and matched_memory:
        delete_latency_ms = delete_memory(
            base_url=base_url,
            token=token,
            memory_id=str(matched_memory["id"]),
            timeout=timeout,
        )
        if settle_seconds > 0:
            time.sleep(min(settle_seconds, 2.0))
        post_delete_memories, _ = search_memories(
            base_url=base_url,
            token=token,
            query=case.search_query,
            timeout=timeout,
        )
        delete_pass = not memory_hit(post_delete_memories, case.expected_memory_terms)[0]
    elif case.delete_after:
        delete_pass = False

    isolation_pass: bool | None = None
    other_user_memories: list[dict[str, Any]] = []
    if other_token:
        other_user_memories, _ = search_memories(
            base_url=base_url,
            token=other_token,
            query=case.search_query,
            timeout=timeout,
        )
        isolation_pass = not memory_hit(other_user_memories, case.expected_memory_terms)[0]

    return {
        "case_id": case.case_id,
        "category": case.category,
        "setup_session_id": setup_session_id,
        "query_session_id": query_session_id,
        "setup_results": setup_results,
        "search_query": case.search_query,
        "cross_session_query": case.cross_session_query,
        "expected_memory_terms": case.expected_memory_terms,
        "expected_answer_terms": case.expected_answer_terms,
        "retrieved_memory_count": len(memories),
        "retrieved_memories": memories,
        "matched_memory": matched_memory,
        "search_pass": search_pass,
        "answer": answer_text,
        "answer_route": answer_response.get("route"),
        "answer_pass": answer_pass,
        "delete_after": case.delete_after,
        "delete_pass": delete_pass,
        "post_delete_memory_count": len(post_delete_memories),
        "isolation_pass": isolation_pass,
        "other_user_memory_count": len(other_user_memories),
        "search_latency_ms": search_latency_ms,
        "answer_latency_ms": answer_latency_ms,
        "delete_latency_ms": delete_latency_ms,
        "passed": bool(
            search_pass
            and answer_pass
            and (delete_pass is not False)
            and (isolation_pass is not False)
        ),
    }


def average(values: list[float]) -> float:
    return round(sum(values) / max(len(values), 1), 2)


def summarize(details: list[dict[str, Any]], run_id: str, args: argparse.Namespace) -> dict[str, Any]:
    total = len(details)
    delete_cases = [row for row in details if row["delete_after"]]
    isolation_cases = [row for row in details if row["isolation_pass"] is not None]
    return {
        "run_id": run_id,
        "cases": total,
        "passed": sum(1 for row in details if row["passed"]),
        "pass_rate": round(sum(1 for row in details if row["passed"]) / max(total, 1), 4),
        "memory_search_hit_rate": round(sum(1 for row in details if row["search_pass"]) / max(total, 1), 4),
        "answer_hit_rate": round(sum(1 for row in details if row["answer_pass"]) / max(total, 1), 4),
        "delete_pass_rate": None
        if not delete_cases
        else round(sum(1 for row in delete_cases if row["delete_pass"]) / len(delete_cases), 4),
        "isolation_pass_rate": None
        if not isolation_cases
        else round(sum(1 for row in isolation_cases if row["isolation_pass"]) / len(isolation_cases), 4),
        "average_search_latency_ms": average([row["search_latency_ms"] for row in details]),
        "average_answer_latency_ms": average([row["answer_latency_ms"] for row in details]),
        "base_url": args.base_url,
        "cases_path": str(args.cases_path.resolve()),
        "settle_seconds": args.settle_seconds,
    }


def write_outputs(output_dir: Path, output_name: str, summary: dict[str, Any], details: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{output_name}_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    with (output_dir / f"{output_name}_details.jsonl").open("w", encoding="utf-8") as file:
        for row in details:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / f"{output_name}_details.csv"
    fieldnames = [
        "case_id",
        "category",
        "passed",
        "search_pass",
        "answer_pass",
        "delete_pass",
        "isolation_pass",
        "retrieved_memory_count",
        "search_latency_ms",
        "answer_latency_ms",
        "answer_route",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    token = args.token or make_token(args.user_id, args.secret_key, args.algorithm)
    other_token = args.other_token
    if other_token is None and args.secret_key:
        other_token = make_token(args.other_user_id, args.secret_key, args.algorithm)

    cases = load_cases(args.cases_path, args.limit)
    if not cases:
        raise ValueError(f"No cases loaded from {args.cases_path}")

    details: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        detail = evaluate_case(
            case=case,
            base_url=args.base_url,
            token=token,
            run_id=run_id,
            settle_seconds=args.settle_seconds,
            timeout=args.timeout,
            other_token=other_token,
        )
        details.append(detail)
        print(
            f"[{index}/{len(cases)}] {case.case_id} "
            f"search={detail['search_pass']} answer={detail['answer_pass']} "
            f"delete={detail['delete_pass']} isolation={detail['isolation_pass']}",
            flush=True,
        )

    summary = summarize(details, run_id, args)
    output_name = args.output_name or f"long_term_memory_{run_id}"
    write_outputs(args.output_dir, output_name, summary, details)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
