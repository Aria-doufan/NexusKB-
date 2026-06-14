import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_enterprise_chunking_profiles import (
    BACKEND_DIR,
    ChunkProfile,
    build_eval_command,
    build_index_command,
    build_prepare_command,
    build_profile_plan,
    build_run_record,
    build_sample_fingerprint,
    ensure_matching_fingerprints,
    main,
    parse_k_values,
    render_comparison_report,
    select_winner,
    stage2_semantic_profiles,
    validate_report_k_values,
    stage1_profiles,
)


def test_stage1_profiles_match_chroma_fixed_size_matrix():
    profiles = stage1_profiles()

    assert [profile.name for profile in profiles] == [
        "fixed_baseline",
        "fixed_smaller_child",
        "fixed_larger_child",
        "fixed_larger_parent",
    ]
    assert profiles[0] == ChunkProfile("fixed_baseline", 3000, 300, 700, 100, "recursive")
    assert profiles[1] == ChunkProfile("fixed_smaller_child", 3000, 300, 500, 80, "recursive")
    assert profiles[2] == ChunkProfile("fixed_larger_child", 3000, 300, 900, 120, "recursive")
    assert profiles[3] == ChunkProfile("fixed_larger_parent", 4000, 400, 700, 100, "recursive")
    assert {profile.child_boundary_mode for profile in profiles} == {"recursive"}


def test_stage2_semantic_profiles_match_chroma_semantic_matrix():
    profiles = stage2_semantic_profiles()

    assert [profile.name for profile in profiles] == [
        "semantic_baseline_threshold",
        "semantic_smaller_child",
        "semantic_larger_child",
    ]
    assert profiles[0] == ChunkProfile("semantic_baseline_threshold", 3000, 300, 700, 100, "semantic")
    assert profiles[1] == ChunkProfile("semantic_smaller_child", 3000, 300, 500, 80, "semantic")
    assert profiles[2] == ChunkProfile("semantic_larger_child", 3000, 300, 900, 120, "semantic")
    assert {profile.child_boundary_mode for profile in profiles} == {"semantic"}


def test_chunk_profile_slug_is_safe_for_paths():
    profile = ChunkProfile("smaller_child", 3000, 300, 500, 80)

    assert profile.slug == "smaller_child_p3000-o300_c500-o80"


def test_build_prepare_command_uses_parent_child_profile_paths():
    profile = ChunkProfile("baseline", 3000, 300, 700, 100)
    output_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/prepared")

    command = build_prepare_command(profile=profile, output_dir=output_dir, sample_size=10000, seed=123)

    assert command[:4] == ["python", str(BACKEND_DIR / "scripts" / "prepare_enterprise_rag_bench.py"), "--strategy", "parent_child"]
    assert "--output-dir" in command
    assert str(output_dir) in command
    assert command[command.index("--sample-size") + 1] == "10000"
    assert command[command.index("--seed") + 1] == "123"
    assert command[command.index("--parent-chunk-size") + 1] == "3000"
    assert command[command.index("--parent-chunk-overlap") + 1] == "300"
    assert command[command.index("--child-chunk-size") + 1] == "700"
    assert command[command.index("--child-chunk-overlap") + 1] == "100"
    assert command[command.index("--child-boundary-mode") + 1] == "recursive"


def test_build_sample_fingerprint_is_stable_for_same_document_and_question_ids(tmp_path):
    documents_a = tmp_path / "documents_a.jsonl"
    questions_a = tmp_path / "questions_a.jsonl"
    documents_b = tmp_path / "documents_b.jsonl"
    questions_b = tmp_path / "questions_b.jsonl"
    documents_a.write_text('{"doc_id":"doc-2","text":"A"}\n{"doc_id":"doc-1","text":"B"}\n', encoding="utf-8")
    questions_a.write_text('{"question_id":"q-1","question":"A?"}\n{"question_id":"q-2","question":"B?"}\n', encoding="utf-8")
    documents_b.write_text('{"doc_id":"doc-2","text":"changed"}\n{"doc_id":"doc-1","text":"content"}\n', encoding="utf-8")
    questions_b.write_text('{"question_id":"q-1","question":"Changed?"}\n{"question_id":"q-2","question":"Content?"}\n', encoding="utf-8")

    fingerprint_a = build_sample_fingerprint(documents_a, questions_a, sample_size=2, seed=42)
    fingerprint_b = build_sample_fingerprint(documents_b, questions_b, sample_size=2, seed=42)

    assert fingerprint_a == fingerprint_b
    assert fingerprint_a["documents"]["count"] == 2
    assert fingerprint_a["questions"]["count"] == 2
    assert fingerprint_a["sample_size"] == 2
    assert fingerprint_a["seed"] == 42


def test_ensure_matching_fingerprints_rejects_profile_mismatch(tmp_path):
    documents_a = tmp_path / "documents_a.jsonl"
    questions_a = tmp_path / "questions_a.jsonl"
    documents_b = tmp_path / "documents_b.jsonl"
    questions_b = tmp_path / "questions_b.jsonl"
    documents_a.write_text('{"doc_id":"doc-1"}\n', encoding="utf-8")
    questions_a.write_text('{"question_id":"q-1"}\n', encoding="utf-8")
    documents_b.write_text('{"doc_id":"doc-2"}\n', encoding="utf-8")
    questions_b.write_text('{"question_id":"q-1"}\n', encoding="utf-8")
    fingerprints = [
        build_sample_fingerprint(documents_a, questions_a, sample_size=1, seed=42),
        build_sample_fingerprint(documents_b, questions_b, sample_size=1, seed=42),
    ]

    with pytest.raises(ValueError, match="Sample fingerprint mismatch"):
        ensure_matching_fingerprints(fingerprints)


def test_main_detects_profile_sample_mismatch_before_second_index_eval(monkeypatch, tmp_path):
    profiles = [
        ChunkProfile("profile_a", 3000, 300, 700, 100),
        ChunkProfile("profile_b", 3000, 300, 700, 100),
    ]
    commands = []

    def write_prepared_artifacts(prepared_dir: Path, doc_id: str) -> None:
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared_dir.joinpath("documents_sample.jsonl").write_text(f'{{"doc_id":"{doc_id}"}}\n', encoding="utf-8")
        prepared_dir.joinpath("questions.jsonl").write_text('{"question_id":"q-1"}\n', encoding="utf-8")
        prepared_dir.joinpath("dataset_stats.json").write_text('{"parent_child":{}}', encoding="utf-8")
        eval_dir = prepared_dir.parent / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.joinpath("chroma_bm25_rrf_reranker_summary.json").write_text(
            '{"questions":1,"recall@10":1.0,"evidence_coverage@10":1.0,"hit@5":1.0,"mrr@20":1.0,"ndcg@10":1.0,"average_latency_ms":1.0}',
            encoding="utf-8",
        )

    def fake_run_command(command, dry_run):
        commands.append(command)
        if str(BACKEND_DIR / "scripts" / "prepare_enterprise_rag_bench.py") in command:
            prepared_dir = Path(command[command.index("--output-dir") + 1])
            doc_id = "doc-2" if "profile_b" in str(prepared_dir) else "doc-1"
            write_prepared_artifacts(prepared_dir, doc_id)

    monkeypatch.setattr(sys, "argv", ["evaluate_enterprise_chunking_profiles.py", "--output-root", str(tmp_path), "--sample-size", "1"])
    monkeypatch.setattr("scripts.evaluate_enterprise_chunking_profiles.stage1_profiles", lambda: profiles)
    monkeypatch.setattr("scripts.evaluate_enterprise_chunking_profiles.run_command", fake_run_command)

    with pytest.raises(ValueError, match="Sample fingerprint mismatch"):
        main()

    second_expensive_commands = [
        command
        for command in commands
        if (str(BACKEND_DIR / "scripts" / "index_enterprise_chunks_chroma.py") in command or str(BACKEND_DIR / "scripts" / "evaluate_enterprise_hybrid_retrieval.py") in command)
        and any("profile_b" in part for part in command)
    ]
    assert second_expensive_commands == []


def test_main_successful_run_writes_stage_sample_fingerprint_and_passes_cli_seed(monkeypatch, tmp_path):
    profile = ChunkProfile("semantic/variant", 3000, 300, 700, 100, "semantic")
    commands = []

    def write_prepared_artifacts(prepared_dir: Path) -> None:
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared_dir.joinpath("documents_sample.jsonl").write_text('{"doc_id":"doc-1"}\n', encoding="utf-8")
        prepared_dir.joinpath("questions.jsonl").write_text('{"question_id":"q-1"}\n', encoding="utf-8")
        prepared_dir.joinpath("dataset_stats.json").write_text(
            json.dumps(
                {
                    "parent_child": {
                        "total_child_chunks": 1,
                        "average_child_chunk_chars": 10.0,
                        "max_child_chunk_chars": 10,
                        "total_parent_chunks": 1,
                        "average_parent_chunk_chars": 20.0,
                        "max_parent_chunk_chars": 20,
                    }
                }
            ),
            encoding="utf-8",
        )

    def write_eval_artifacts(eval_dir: Path, method: str) -> None:
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.joinpath(f"{method}_summary.json").write_text(
            json.dumps(
                {
                    "questions": 1,
                    "recall@10": 1.0,
                    "evidence_coverage@10": 1.0,
                    "hit@5": 1.0,
                    "mrr@20": 1.0,
                    "ndcg@10": 1.0,
                    "average_latency_ms": 1.0,
                }
            ),
            encoding="utf-8",
        )
        eval_dir.joinpath(f"{method}_details.jsonl").write_text("", encoding="utf-8")

    def fake_run_command(command, dry_run):
        commands.append(command)
        if str(BACKEND_DIR / "scripts" / "prepare_enterprise_rag_bench.py") in command:
            write_prepared_artifacts(Path(command[command.index("--output-dir") + 1]))
        if str(BACKEND_DIR / "scripts" / "evaluate_enterprise_hybrid_retrieval.py") in command:
            write_eval_artifacts(Path(command[command.index("--output-dir") + 1]), command[command.index("--method") + 1])

    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_enterprise_chunking_profiles.py", "--output-root", str(tmp_path), "--sample-size", "1", "--seed", "987"],
    )
    monkeypatch.setattr("scripts.evaluate_enterprise_chunking_profiles.stage1_profiles", lambda: [profile])
    monkeypatch.setattr("scripts.evaluate_enterprise_chunking_profiles.run_command", fake_run_command)

    main()

    stage_fingerprint_path = tmp_path / "stage1" / "sample_fingerprint.json"
    assert stage_fingerprint_path.exists()
    stage_fingerprint = json.loads(stage_fingerprint_path.read_text(encoding="utf-8"))
    assert stage_fingerprint["sample_size"] == 1
    assert stage_fingerprint["seed"] == 987
    assert stage_fingerprint["documents"]["count"] == 1
    assert stage_fingerprint["questions"]["count"] == 1

    prepare_commands = [command for command in commands if str(BACKEND_DIR / "scripts" / "prepare_enterprise_rag_bench.py") in command]
    assert len(prepare_commands) == 1
    assert prepare_commands[0][prepare_commands[0].index("--seed") + 1] == "987"

    run_record_path = tmp_path / "stage1" / profile.slug / "run_record.json"
    run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    assert run_record["sample_fingerprint"] == stage_fingerprint
    assert run_record["chunk_profile"]["profile_name"] == "semantic/variant"
    assert run_record["chunk_profile"]["profile_name"] != profile.slug


def test_build_index_command_uses_profile_specific_collection_and_paths():
    chunks_path = Path("backend/data/chunking_eval_outputs/stage1/baseline/prepared/child_chunks_parent_child.jsonl")
    persist_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/chroma")

    command = build_index_command(
        chunks_path=chunks_path,
        persist_dir=persist_dir,
        collection_name="enterprise_chunking_baseline",
        embedding_model="qwen3-embedding:latest",
        reset=True,
    )

    assert command[:2] == ["python", str(BACKEND_DIR / "scripts" / "index_enterprise_chunks_chroma.py")]
    assert command[command.index("--chunks-path") + 1] == str(chunks_path)
    assert command[command.index("--persist-dir") + 1] == str(persist_dir)
    assert command[command.index("--collection-name") + 1] == "enterprise_chunking_baseline"
    assert "--reset" in command


def test_build_profile_plan_contains_prepare_index_eval_commands():
    profile = ChunkProfile("baseline", 3000, 300, 700, 100)
    output_root = Path("backend/data/chunking_eval_outputs")
    plan = build_profile_plan(
        stage="stage1",
        profile=profile,
        output_root=output_root,
        sample_size=10000,
        embedding_model="qwen3-embedding:latest",
        method="chroma_bm25_rrf_reranker",
        k_values="1,5,10,20",
        limit=None,
        reset_index=True,
    )
    expected_profile_root = output_root.resolve() / "stage1" / profile.slug

    assert plan["run_id"] == "enterprise_chunking_stage1_baseline"
    assert plan["collection_name"] == "enterprise_chunking_stage1_baseline"
    assert plan["prepared_dir"] == expected_profile_root / "prepared"
    assert plan["persist_dir"] == expected_profile_root / "chroma"
    assert plan["eval_dir"] == expected_profile_root / "eval"
    assert plan["prepare_command"][1] == str(BACKEND_DIR / "scripts" / "prepare_enterprise_rag_bench.py")
    assert plan["index_command"][1] == str(BACKEND_DIR / "scripts" / "index_enterprise_chunks_chroma.py")
    assert plan["eval_command"][1] == str(BACKEND_DIR / "scripts" / "evaluate_enterprise_hybrid_retrieval.py")
    assert plan["eval_command"][plan["eval_command"].index("--embedding-model") + 1] == "qwen3-embedding:latest"


def test_build_profile_plan_uses_profile_slug_for_artifact_paths(tmp_path):
    profile = ChunkProfile("semantic/variant", 3000, 300, 700, 100, "semantic")
    plan = build_profile_plan(
        stage="stage2_semantic",
        profile=profile,
        output_root=tmp_path,
        sample_size=10000,
        embedding_model="qwen3-embedding:latest",
        method="chroma_bm25_rrf_reranker",
        k_values="1,5,10,20",
        limit=None,
        reset_index=True,
    )

    assert plan["profile_root"].parent == tmp_path.resolve() / "stage2_semantic"
    assert plan["profile_root"].name == profile.slug
    assert plan["profile"].name == "semantic/variant"


def test_build_profile_plan_resolves_relative_output_root_for_artifact_paths():
    profile = ChunkProfile("baseline", 3000, 300, 700, 100)
    output_root = Path("relative/chunking_outputs")
    plan = build_profile_plan(
        stage="stage1",
        profile=profile,
        output_root=output_root,
        sample_size=10000,
        embedding_model="qwen3-embedding:latest",
        method="chroma_bm25_rrf_reranker",
        k_values="1,5,10,20",
        limit=None,
        reset_index=True,
    )
    expected_profile_root = output_root.resolve() / "stage1" / profile.slug
    expected_prepared_dir = expected_profile_root / "prepared"
    expected_persist_dir = expected_profile_root / "chroma"
    expected_eval_dir = expected_profile_root / "eval"

    assert plan["prepared_dir"] == expected_prepared_dir
    assert plan["persist_dir"] == expected_persist_dir
    assert plan["eval_dir"] == expected_eval_dir
    assert plan["prepared_dir"].is_absolute()
    assert plan["persist_dir"].is_absolute()
    assert plan["eval_dir"].is_absolute()
    assert plan["prepare_command"][plan["prepare_command"].index("--output-dir") + 1] == str(expected_prepared_dir)
    assert plan["index_command"][plan["index_command"].index("--chunks-path") + 1] == str(expected_prepared_dir / "child_chunks_parent_child.jsonl")
    assert plan["index_command"][plan["index_command"].index("--persist-dir") + 1] == str(expected_persist_dir)
    assert plan["eval_command"][plan["eval_command"].index("--questions-path") + 1] == str(expected_prepared_dir / "questions.jsonl")
    assert plan["eval_command"][plan["eval_command"].index("--child-chunks-path") + 1] == str(expected_prepared_dir / "child_chunks_parent_child.jsonl")
    assert plan["eval_command"][plan["eval_command"].index("--parent-chunks-path") + 1] == str(expected_prepared_dir / "parent_chunks_parent_child.jsonl")
    assert plan["eval_command"][plan["eval_command"].index("--persist-dir") + 1] == str(expected_persist_dir)
    assert plan["eval_command"][plan["eval_command"].index("--output-dir") + 1] == str(expected_eval_dir)


def test_build_eval_command_points_at_profile_index_and_chunks():
    prepared_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/prepared")
    persist_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/chroma")
    output_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/eval")

    command = build_eval_command(
        prepared_dir=prepared_dir,
        persist_dir=persist_dir,
        collection_name="enterprise_chunking_baseline",
        output_dir=output_dir,
        method="chroma_bm25_rrf_reranker",
        k_values="1,5,10,20",
        embedding_model="qwen3-embedding:latest",
        limit=25,
    )

    assert command[:2] == ["python", str(BACKEND_DIR / "scripts" / "evaluate_enterprise_hybrid_retrieval.py")]
    assert command[command.index("--questions-path") + 1] == str(prepared_dir / "questions.jsonl")
    assert command[command.index("--child-chunks-path") + 1] == str(prepared_dir / "child_chunks_parent_child.jsonl")
    assert command[command.index("--parent-chunks-path") + 1] == str(prepared_dir / "parent_chunks_parent_child.jsonl")
    assert command[command.index("--persist-dir") + 1] == str(persist_dir)
    assert command[command.index("--output-dir") + 1] == str(output_dir)
    assert command[command.index("--embedding-model") + 1] == "qwen3-embedding:latest"
    assert command[command.index("--limit") + 1] == "25"


def test_build_run_record_uses_source_agnostic_shape():
    profile = ChunkProfile("baseline", 3000, 300, 700, 100)
    summary = {
        "questions": 10,
        "recall@10": 0.7,
        "evidence_coverage@10": 0.6,
        "hit@5": 0.8,
        "mrr@20": 0.5,
        "ndcg@10": 0.55,
        "average_latency_ms": 120.0,
        "question_type_summary": {"fact_lookup": {"questions": 5, "recall@10": 0.8}},
    }
    chunk_stats = {
        "parent_child": {
            "total_child_chunks": 123,
            "average_child_chunk_chars": 650.5,
            "max_child_chunk_chars": 700,
        }
    }

    record = build_run_record(
        run_id="enterprise_chunking_stage1_baseline",
        source_type="enterprise",
        profile=profile,
        collection_name="enterprise_chunking_baseline",
        embedding_model="qwen3-embedding:latest",
        retrieval_method="chroma_bm25_rrf_reranker",
        k_values=[1, 5, 10, 20],
        summary=summary,
        chunk_stats=chunk_stats,
        details_path=Path("details.jsonl"),
        report_path=None,
    )

    assert record["run_id"] == "enterprise_chunking_stage1_baseline"
    assert record["source_type"] == "enterprise"
    assert record["chunk_profile"]["child_chunk_size"] == 700
    assert record["index_profile"]["collection"] == "enterprise_chunking_baseline"
    assert record["summary_metrics"]["recall@10"] == 0.7
    assert record["summary_metrics"]["mrr@20"] == 0.5
    assert "mrr@10" not in record["summary_metrics"]
    assert record["chunk_statistics"]["total_child_chunks"] == 123
    assert record["details_path"] == "details.jsonl"
    assert record["report_path"] is None


def test_parse_k_values_returns_sorted_deduped_positive_values():
    assert parse_k_values("10,5,5,1") == [1, 5, 10]


def test_parse_k_values_rejects_zero_values():
    with pytest.raises(ValueError):
        parse_k_values("0,5,10")


def test_parse_k_values_rejects_negative_values():
    with pytest.raises(ValueError):
        parse_k_values("-1,5,10")


def test_parse_k_values_rejects_invalid_text():
    with pytest.raises(ValueError):
        parse_k_values("five,10")


@pytest.mark.parametrize("value", ["5,,10", "5,", ",5,10", "   "])
def test_parse_k_values_rejects_empty_tokens(value):
    with pytest.raises(ValueError, match="empty values"):
        parse_k_values(value)


def test_main_rejects_empty_k_values_before_running_child_commands(monkeypatch):
    def fail_if_called(command, dry_run):
        raise AssertionError(f"Unexpected child command: {command}")

    monkeypatch.setattr(sys, "argv", ["evaluate_enterprise_chunking_profiles.py", "--dry-run", "--k-values", "5,,10"])
    monkeypatch.setattr("scripts.evaluate_enterprise_chunking_profiles.run_command", fail_if_called)

    with pytest.raises(ValueError, match="empty values"):
        main()


def test_validate_report_k_values_requires_stage_report_columns():
    validate_report_k_values([1, 5, 10, 20])

    with pytest.raises(ValueError, match="include"):
        validate_report_k_values([1, 5, 20])

    with pytest.raises(ValueError, match="include"):
        validate_report_k_values([1, 10, 20])


def test_build_run_record_rejects_missing_required_report_k_values():
    profile = ChunkProfile("baseline", 3000, 300, 700, 100)
    summary = {
        "questions": 10,
        "recall@10": 0.7,
        "evidence_coverage@10": 0.6,
        "hit@5": 0.8,
        "mrr@20": 0.5,
        "ndcg@10": 0.55,
        "average_latency_ms": 120.0,
    }

    with pytest.raises(ValueError, match="include"):
        build_run_record(
            run_id="enterprise_chunking_stage1_baseline",
            source_type="enterprise",
            profile=profile,
            collection_name="enterprise_chunking_baseline",
            embedding_model="qwen3-embedding:latest",
            retrieval_method="chroma_bm25_rrf_reranker",
            k_values=[1, 10, 20],
            summary=summary,
            chunk_stats={},
            details_path=Path("details.jsonl"),
            report_path=Path("report.md"),
        )

    with pytest.raises(ValueError, match="include"):
        build_run_record(
            run_id="enterprise_chunking_stage1_baseline",
            source_type="enterprise",
            profile=profile,
            collection_name="enterprise_chunking_baseline",
            embedding_model="qwen3-embedding:latest",
            retrieval_method="chroma_bm25_rrf_reranker",
            k_values=[1, 5, 20],
            summary=summary,
            chunk_stats={},
            details_path=Path("details.jsonl"),
            report_path=Path("report.md"),
        )


def test_select_winner_rejects_empty_records():
    with pytest.raises(ValueError):
        select_winner([])


def test_select_winner_prioritizes_recall_then_coverage_then_hit():
    records = [
        {
            "chunk_profile": {"profile_name": "baseline"},
            "summary_metrics": {"recall@10": 0.70, "evidence_coverage@10": 0.50, "hit@5": 0.80, "mrr@20": 0.50, "ndcg@10": 0.65},
        },
        {
            "chunk_profile": {"profile_name": "smaller_child"},
            "summary_metrics": {"recall@10": 0.75, "evidence_coverage@10": 0.50, "hit@5": 0.80, "mrr@20": 0.50, "ndcg@10": 0.65},
        },
        {
            "chunk_profile": {"profile_name": "larger_child"},
            "summary_metrics": {"recall@10": 0.75, "evidence_coverage@10": 0.55, "hit@5": 0.75, "mrr@20": 0.50, "ndcg@10": 0.65},
        },
    ]

    assert select_winner(records)["chunk_profile"]["profile_name"] == "larger_child"


def test_select_winner_uses_available_mrr_key_as_tiebreaker():
    records = [
        {
            "chunk_profile": {"profile_name": "baseline"},
            "summary_metrics": {"recall@10": 0.75, "evidence_coverage@10": 0.55, "hit@5": 0.80, "mrr@20": 0.50, "ndcg@10": 0.65},
        },
        {
            "chunk_profile": {"profile_name": "smaller_child"},
            "summary_metrics": {"recall@10": 0.75, "evidence_coverage@10": 0.55, "hit@5": 0.80, "mrr@20": 0.60, "ndcg@10": 0.60},
        },
    ]

    assert select_winner(records)["chunk_profile"]["profile_name"] == "smaller_child"


def test_render_comparison_report_rejects_mixed_mrr_keys():
    records = [
        {
            "chunk_profile": {"profile_name": "baseline"},
            "summary_metrics": {"recall@10": 0.75, "evidence_coverage@10": 0.55, "hit@5": 0.80, "mrr@10": 0.50, "ndcg@10": 0.65},
        },
        {
            "chunk_profile": {"profile_name": "smaller_child"},
            "summary_metrics": {"recall@10": 0.75, "evidence_coverage@10": 0.55, "hit@5": 0.80, "mrr@20": 0.60, "ndcg@10": 0.60},
        },
    ]

    with pytest.raises(ValueError, match="incompatible MRR keys"):
        render_comparison_report(stage="stage1", records=records)


def test_render_comparison_report_includes_primary_metrics_and_winner():
    records = [
        {
            "run_id": "enterprise_chunking_stage1_baseline",
            "chunk_profile": {"profile_name": "baseline", "parent_chunk_size": 3000, "parent_chunk_overlap": 300, "child_chunk_size": 700, "child_chunk_overlap": 100},
            "summary_metrics": {"recall@10": 0.70, "evidence_coverage@10": 0.50, "hit@5": 0.80, "mrr@20": 0.60, "ndcg@10": 0.65, "average_latency_ms": 100.0},
            "chunk_statistics": {
                "total_child_chunks": 1000,
                "average_child_chunk_chars": 650.0,
                "max_child_chunk_chars": 700,
                "total_parent_chunks": 250,
                "average_parent_chunk_chars": 2800.0,
                "max_parent_chunk_chars": 3000,
            },
        },
        {
            "run_id": "enterprise_chunking_stage1_smaller_child",
            "chunk_profile": {"profile_name": "smaller_child", "parent_chunk_size": 3000, "parent_chunk_overlap": 300, "child_chunk_size": 500, "child_chunk_overlap": 80},
            "summary_metrics": {"recall@10": 0.75, "evidence_coverage@10": 0.55, "hit@5": 0.82, "mrr@20": 0.61, "ndcg@10": 0.66, "average_latency_ms": 130.0},
            "chunk_statistics": {
                "total_child_chunks": 1300,
                "average_child_chunk_chars": 480.0,
                "max_child_chunk_chars": 500,
                "total_parent_chunks": 250,
                "average_parent_chunk_chars": 2800.0,
                "max_parent_chunk_chars": 3000,
            },
        },
    ]

    report = render_comparison_report(stage="stage1", records=records)

    assert "# Enterprise Chunking Recall Evaluation: stage1" in report
    assert "| child chunks | avg child chars | max child chars | parent chunks | avg parent chars | max parent chars |" in report
    assert "| mrr@20 |" in report
    assert "mrr@10" not in report
    assert "| baseline | 3000 / 300 | 700 / 100 | 0.7 | 0.5 | 0.8 | 0.6 | 0.65 | 100.0 | 1000 | 650.0 | 700 | 250 | 2800.0 | 3000 |" in report
    assert "Recommended winner: `smaller_child`" in report
