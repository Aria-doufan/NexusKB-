"""Run recall-focused enterprise chunking profile evaluations."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = BACKEND_DIR / "data" / "chunking_eval_outputs"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_METHOD = "chroma_bm25_rrf_reranker"
DEFAULT_K_VALUES = "1,5,10,20"
PREPARE_SCRIPT = BACKEND_DIR / "scripts" / "prepare_enterprise_rag_bench.py"
INDEX_SCRIPT = BACKEND_DIR / "scripts" / "index_enterprise_chunks_chroma.py"
EVAL_SCRIPT = BACKEND_DIR / "scripts" / "evaluate_enterprise_hybrid_retrieval.py"


@dataclass(frozen=True, slots=True)
class ChunkProfile:
    name: str
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int

    @property
    def slug(self) -> str:
        return (
            f"{self.name}_"
            f"p{self.parent_chunk_size}-o{self.parent_chunk_overlap}_"
            f"c{self.child_chunk_size}-o{self.child_chunk_overlap}"
        )

    def to_record(self) -> dict[str, int | str]:
        return asdict(self)


def stage1_profiles() -> list[ChunkProfile]:
    return [
        ChunkProfile("baseline", 3000, 300, 700, 100),
        ChunkProfile("smaller_child", 3000, 300, 500, 80),
        ChunkProfile("larger_child", 3000, 300, 900, 120),
        ChunkProfile("larger_parent", 4000, 400, 700, 100),
    ]


def build_prepare_command(
    profile: ChunkProfile,
    output_dir: Path,
    sample_size: int,
) -> list[str]:
    return [
        "python",
        str(PREPARE_SCRIPT),
        "--strategy",
        "parent_child",
        "--output-dir",
        str(output_dir),
        "--sample-size",
        str(sample_size),
        "--parent-chunk-size",
        str(profile.parent_chunk_size),
        "--parent-chunk-overlap",
        str(profile.parent_chunk_overlap),
        "--child-chunk-size",
        str(profile.child_chunk_size),
        "--child-chunk-overlap",
        str(profile.child_chunk_overlap),
    ]


def build_index_command(
    chunks_path: Path,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    reset: bool,
) -> list[str]:
    command = [
        "python",
        str(INDEX_SCRIPT),
        "--chunks-path",
        str(chunks_path),
        "--persist-dir",
        str(persist_dir),
        "--collection-name",
        collection_name,
        "--embedding-model",
        embedding_model,
    ]
    if reset:
        command.append("--reset")
    return command


def build_eval_command(
    prepared_dir: Path,
    persist_dir: Path,
    collection_name: str,
    output_dir: Path,
    method: str,
    k_values: str,
    embedding_model: str,
    limit: int | None,
) -> list[str]:
    command = [
        "python",
        str(EVAL_SCRIPT),
        "--method",
        method,
        "--questions-path",
        str(prepared_dir / "questions.jsonl"),
        "--child-chunks-path",
        str(prepared_dir / "child_chunks_parent_child.jsonl"),
        "--parent-chunks-path",
        str(prepared_dir / "parent_chunks_parent_child.jsonl"),
        "--persist-dir",
        str(persist_dir),
        "--collection-name",
        collection_name,
        "--output-dir",
        str(output_dir),
        "--k-values",
        k_values,
        "--embedding-model",
        embedding_model,
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def parse_k_values(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",")]
    if any(item == "" for item in items):
        raise ValueError(f"--k-values must not include empty values: {value}")

    try:
        k_values = {int(item) for item in items}
    except ValueError as exc:
        raise ValueError(f"Invalid k-value in --k-values: {value}") from exc

    if not k_values:
        raise ValueError("--k-values must include at least one integer")

    non_positive_values = sorted(k_value for k_value in k_values if k_value <= 0)
    if non_positive_values:
        invalid_values = ", ".join(str(k_value) for k_value in non_positive_values)
        raise ValueError(f"--k-values must be positive integers; got: {invalid_values}")

    return sorted(k_values)


def build_profile_plan(
    stage: str,
    profile: ChunkProfile,
    output_root: Path,
    sample_size: int,
    embedding_model: str,
    method: str,
    k_values: str,
    limit: int | None,
    reset_index: bool,
) -> dict[str, Any]:
    profile_root = output_root / stage / profile.name
    prepared_dir = profile_root / "prepared"
    persist_dir = profile_root / "chroma"
    eval_dir = profile_root / "eval"
    collection_name = f"enterprise_chunking_{stage}_{profile.name}"
    return {
        "run_id": f"enterprise_chunking_{stage}_{profile.name}",
        "profile": profile,
        "profile_root": profile_root,
        "prepared_dir": prepared_dir,
        "persist_dir": persist_dir,
        "eval_dir": eval_dir,
        "collection_name": collection_name,
        "prepare_command": build_prepare_command(profile, prepared_dir, sample_size),
        "index_command": build_index_command(
            chunks_path=prepared_dir / "child_chunks_parent_child.jsonl",
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            reset=reset_index,
        ),
        "eval_command": build_eval_command(
            prepared_dir=prepared_dir,
            persist_dir=persist_dir,
            collection_name=collection_name,
            output_dir=eval_dir,
            method=method,
            k_values=k_values,
            embedding_model=embedding_model,
            limit=limit,
        ),
    }


def validate_report_k_values(k_values: list[int]) -> None:
    required_k_values = {5, 10}
    missing_k_values = sorted(required_k_values.difference(k_values))
    if missing_k_values:
        missing = ", ".join(str(k_value) for k_value in missing_k_values)
        raise ValueError(f"Stage 1 reports require k_values to include: {missing}")


def _mrr_key_for_k_values(k_values: list[int]) -> str:
    if not k_values:
        return "mrr@20"
    return f"mrr@{max(k_values)}"


def _mrr_keys(metrics: dict[str, Any]) -> list[str]:
    def k_value(key: str) -> int:
        try:
            return int(key.split("@", 1)[1])
        except (IndexError, ValueError):
            return -1

    return sorted(
        (key for key in metrics if key.startswith("mrr@")),
        key=k_value,
        reverse=True,
    )


def common_mrr_key(records: list[dict[str, Any]]) -> str:
    record_mrr_keys = []
    for record in records:
        keys = _mrr_keys(record.get("summary_metrics", {}))
        if len(keys) > 1:
            raise ValueError(f"Record contains incompatible MRR keys: {', '.join(keys)}")
        if keys:
            record_mrr_keys.append(keys[0])

    unique_mrr_keys = sorted(set(record_mrr_keys))
    if len(unique_mrr_keys) > 1:
        raise ValueError(f"Records contain incompatible MRR keys: {', '.join(unique_mrr_keys)}")
    return unique_mrr_keys[0] if unique_mrr_keys else "mrr@20"


def _summary_subset(summary: dict[str, Any], k_values: list[int]) -> dict[str, Any]:
    keys = [
        "questions",
        "recall@10",
        "evidence_coverage@10",
        "hit@5",
        _mrr_key_for_k_values(k_values),
        "ndcg@10",
        "average_latency_ms",
    ]
    return {key: summary[key] for key in keys}


def _chunk_statistics(chunk_stats: dict[str, Any]) -> dict[str, Any]:
    parent_child = chunk_stats.get("parent_child", {})
    return {
        "total_child_chunks": parent_child.get("total_child_chunks", 0),
        "average_child_chunk_chars": parent_child.get("average_child_chunk_chars", 0.0),
        "max_child_chunk_chars": parent_child.get("max_child_chunk_chars", 0),
        "total_parent_chunks": parent_child.get("total_parent_chunks", 0),
        "average_parent_chunk_chars": parent_child.get("average_parent_chunk_chars", 0.0),
        "max_parent_chunk_chars": parent_child.get("max_parent_chunk_chars", 0),
    }


def build_run_record(
    run_id: str,
    source_type: str,
    profile: ChunkProfile,
    collection_name: str,
    embedding_model: str,
    retrieval_method: str,
    k_values: list[int],
    summary: dict[str, Any],
    chunk_stats: dict[str, Any],
    details_path: Path,
    report_path: Path | None,
) -> dict[str, Any]:
    validate_report_k_values(k_values)
    return {
        "run_id": run_id,
        "source_type": source_type,
        "chunk_profile": {
            "profile_name": profile.name,
            "parent_chunk_size": profile.parent_chunk_size,
            "parent_chunk_overlap": profile.parent_chunk_overlap,
            "child_chunk_size": profile.child_chunk_size,
            "child_chunk_overlap": profile.child_chunk_overlap,
        },
        "index_profile": {
            "collection": collection_name,
            "embedding_model": embedding_model,
        },
        "retrieval_method": retrieval_method,
        "k_values": k_values,
        "summary_metrics": _summary_subset(summary, k_values),
        "question_type_summary": summary.get("question_type_summary", {}),
        "chunk_statistics": _chunk_statistics(chunk_stats),
        "details_path": str(details_path),
        "report_path": str(report_path) if report_path is not None else None,
    }


def select_winner(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot select a winner without records")

    report_mrr_key = common_mrr_key(records)

    def score(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
        metrics = record.get("summary_metrics", {})
        return (
            float(metrics["recall@10"]),
            float(metrics["evidence_coverage@10"]),
            float(metrics["hit@5"]),
            float(metrics[report_mrr_key]),
            float(metrics["ndcg@10"]),
        )

    return max(records, key=score)


def render_comparison_report(stage: str, records: list[dict[str, Any]]) -> str:
    winner = select_winner(records) if records else None
    report_mrr_key = common_mrr_key(records)
    lines = [
        f"# Enterprise Chunking Recall Evaluation: {stage}",
        "",
        f"| Profile | Parent size / overlap | Child size / overlap | recall@10 | evidence_coverage@10 | hit@5 | {report_mrr_key} | ndcg@10 | avg latency ms | child chunks | avg child chars | max child chars | parent chunks | avg parent chars | max parent chars |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        profile = record["chunk_profile"]
        metrics = record["summary_metrics"]
        stats = record.get("chunk_statistics", {})
        lines.append(
            "| {name} | {parent_size} / {parent_overlap} | {child_size} / {child_overlap} | {recall} | {coverage} | {hit} | {mrr} | {ndcg} | {latency} | {child_chunks} | {avg_child_chars} | {max_child_chars} | {parent_chunks} | {avg_parent_chars} | {max_parent_chars} |".format(
                name=profile["profile_name"],
                parent_size=profile["parent_chunk_size"],
                parent_overlap=profile["parent_chunk_overlap"],
                child_size=profile["child_chunk_size"],
                child_overlap=profile["child_chunk_overlap"],
                recall=metrics["recall@10"],
                coverage=metrics["evidence_coverage@10"],
                hit=metrics["hit@5"],
                mrr=metrics[report_mrr_key],
                ndcg=metrics["ndcg@10"],
                latency=metrics["average_latency_ms"],
                child_chunks=stats.get("total_child_chunks", 0),
                avg_child_chars=stats.get("average_child_chunk_chars", 0.0),
                max_child_chars=stats.get("max_child_chunk_chars", 0),
                parent_chunks=stats.get("total_parent_chunks", 0),
                avg_parent_chars=stats.get("average_parent_chunk_chars", 0.0),
                max_parent_chars=stats.get("max_parent_chunk_chars", 0),
            )
        )
    lines.extend(["", f"Recommended winner: `{winner['chunk_profile']['profile_name']}`" if winner else "Recommended winner: unavailable"])
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run enterprise chunking recall profile evaluations.")
    parser.add_argument("--stage", choices=["stage1"], default="stage1")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--k-values", default=DEFAULT_K_VALUES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-reset-index", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    printable = " ".join(command)
    print(printable, flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def execute_profile(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.skip_prepare:
        run_command(plan["prepare_command"], args.dry_run)
    if not args.skip_index:
        run_command(plan["index_command"], args.dry_run)
    if not args.skip_eval:
        run_command(plan["eval_command"], args.dry_run)
    if args.dry_run:
        return None

    eval_dir = plan["eval_dir"]
    summary_path = eval_dir / f"{args.method}_summary.json"
    details_path = eval_dir / f"{args.method}_details.jsonl"
    profile_report_path = eval_dir / "report.md"
    report_path = profile_report_path if profile_report_path.exists() else None
    summary = load_json(summary_path)
    chunk_stats = load_json(plan["prepared_dir"] / "dataset_stats.json")
    record = build_run_record(
        run_id=plan["run_id"],
        source_type="enterprise",
        profile=plan["profile"],
        collection_name=plan["collection_name"],
        embedding_model=args.embedding_model,
        retrieval_method=args.method,
        k_values=parse_k_values(args.k_values),
        summary=summary,
        chunk_stats=chunk_stats,
        details_path=details_path,
        report_path=report_path,
    )
    write_json(plan["profile_root"] / "run_record.json", record)
    return record


def main() -> None:
    args = parse_args()
    validate_report_k_values(parse_k_values(args.k_values))
    profiles = stage1_profiles()
    records = []
    for profile in profiles:
        plan = build_profile_plan(
            stage=args.stage,
            profile=profile,
            output_root=args.output_root,
            sample_size=args.sample_size,
            embedding_model=args.embedding_model,
            method=args.method,
            k_values=args.k_values,
            limit=args.limit,
            reset_index=not args.no_reset_index,
        )
        record = execute_profile(plan, args)
        if record is not None:
            records.append(record)

    if records:
        stage_dir = args.output_root / args.stage
        write_json(stage_dir / "comparison_summary.json", {"stage": args.stage, "records": records})
        (stage_dir / "report.md").write_text(render_comparison_report(args.stage, records), encoding="utf-8")



if __name__ == "__main__":
    main()
