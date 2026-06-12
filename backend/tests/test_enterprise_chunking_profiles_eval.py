import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_enterprise_chunking_profiles import (
    ChunkProfile,
    build_eval_command,
    build_index_command,
    build_prepare_command,
    build_run_record,
    render_comparison_report,
    select_winner,
    stage1_profiles,
)


def test_stage1_profiles_match_design_matrix():
    profiles = stage1_profiles()

    assert [profile.name for profile in profiles] == [
        "baseline",
        "smaller_child",
        "larger_child",
        "larger_parent",
    ]
    assert profiles[0] == ChunkProfile("baseline", 3000, 300, 700, 100)
    assert profiles[1] == ChunkProfile("smaller_child", 3000, 300, 500, 80)
    assert profiles[2] == ChunkProfile("larger_child", 3000, 300, 900, 120)
    assert profiles[3] == ChunkProfile("larger_parent", 4000, 400, 700, 100)


def test_chunk_profile_slug_is_safe_for_paths():
    profile = ChunkProfile("smaller_child", 3000, 300, 500, 80)

    assert profile.slug == "smaller_child_p3000-o300_c500-o80"


def test_build_prepare_command_uses_parent_child_profile_paths():
    profile = ChunkProfile("baseline", 3000, 300, 700, 100)
    output_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/prepared")

    command = build_prepare_command(profile=profile, output_dir=output_dir, sample_size=10000)

    assert command[:4] == ["python", "backend/scripts/prepare_enterprise_rag_bench.py", "--strategy", "parent_child"]
    assert "--output-dir" in command
    assert str(output_dir) in command
    assert command[command.index("--parent-chunk-size") + 1] == "3000"
    assert command[command.index("--parent-chunk-overlap") + 1] == "300"
    assert command[command.index("--child-chunk-size") + 1] == "700"
    assert command[command.index("--child-chunk-overlap") + 1] == "100"


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

    assert command[:2] == ["python", "backend/scripts/index_enterprise_chunks_chroma.py"]
    assert command[command.index("--chunks-path") + 1] == str(chunks_path)
    assert command[command.index("--persist-dir") + 1] == str(persist_dir)
    assert command[command.index("--collection-name") + 1] == "enterprise_chunking_baseline"
    assert "--reset" in command


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
        limit=25,
    )

    assert command[:2] == ["python", "backend/scripts/evaluate_enterprise_hybrid_retrieval.py"]
    assert command[command.index("--questions-path") + 1] == str(prepared_dir / "questions.jsonl")
    assert command[command.index("--child-chunks-path") + 1] == str(prepared_dir / "child_chunks_parent_child.jsonl")
    assert command[command.index("--parent-chunks-path") + 1] == str(prepared_dir / "parent_chunks_parent_child.jsonl")
    assert command[command.index("--persist-dir") + 1] == str(persist_dir)
    assert command[command.index("--output-dir") + 1] == str(output_dir)
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
        report_path=Path("report.md"),
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
