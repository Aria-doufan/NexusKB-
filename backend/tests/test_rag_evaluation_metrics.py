import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rag_eval_metrics import (
    average_precision_at_k,
    build_question_type_summary,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_precision_and_recall_at_k_with_multiple_gold_documents():
    ranked = ["doc_a", "doc_x", "doc_b", "doc_c"]
    expected = {"doc_a", "doc_b", "doc_z"}

    assert precision_at_k(ranked, expected, 3) == 2 / 3
    assert recall_at_k(ranked, expected, 3) == 2 / 3


def test_precision_and_recall_return_zero_without_gold_documents():
    ranked = ["doc_a", "doc_b"]
    expected = set()

    assert precision_at_k(ranked, expected, 2) == 0.0
    assert recall_at_k(ranked, expected, 2) == 0.0


def test_reciprocal_rank_uses_first_relevant_document_only():
    ranked = ["doc_x", "doc_y", "doc_b", "doc_a"]
    expected = {"doc_a", "doc_b"}

    assert reciprocal_rank_at_k(ranked, expected, 4) == 1 / 3
    assert reciprocal_rank_at_k(ranked, expected, 2) == 0.0


def test_average_precision_at_k_scores_all_relevant_documents():
    ranked = ["doc_a", "doc_x", "doc_b", "doc_c"]
    expected = {"doc_a", "doc_b", "doc_z"}

    assert average_precision_at_k(ranked, expected, 4) == (1 / 1 + 2 / 3) / 3
    assert average_precision_at_k(ranked, expected, 2) == (1 / 1) / 2


def test_ndcg_at_k_rewards_better_ranked_relevant_documents():
    best = ["doc_a", "doc_b", "doc_x", "doc_y"]
    worse = ["doc_x", "doc_y", "doc_a", "doc_b"]
    expected = {"doc_a", "doc_b"}

    assert ndcg_at_k(best, expected, 4) == 1.0
    assert 0.0 < ndcg_at_k(worse, expected, 4) < 1.0
    assert ndcg_at_k(worse, expected, 2) == 0.0


def test_question_type_summary_averages_metrics_by_type():
    details = [
        {
            "question_type": "fact_lookup",
            "latency_ms": 100.0,
            "hit@1": 1,
            "precision@1": 1.0,
            "recall@1": 0.5,
            "ndcg@1": 1.0,
            "ap@1": 1.0,
            "evidence_coverage@1": 0.0,
            "required_evidence_groups_count": 0,
            "rr@1": 1.0,
        },
        {
            "question_type": "fact_lookup",
            "latency_ms": 300.0,
            "hit@1": 0,
            "precision@1": 0.0,
            "recall@1": 0.0,
            "ndcg@1": 0.0,
            "ap@1": 0.0,
            "evidence_coverage@1": 0.0,
            "required_evidence_groups_count": 0,
            "rr@1": 0.0,
        },
        {
            "question_type": "comparison",
            "latency_ms": 500.0,
            "hit@1": 1,
            "precision@1": 1.0,
            "recall@1": 1.0,
            "ndcg@1": 1.0,
            "ap@1": 1.0,
            "evidence_coverage@1": 1.0,
            "required_evidence_groups_count": 1,
            "rr@1": 1.0,
        },
    ]

    summary = build_question_type_summary(details, [1])

    assert summary["fact_lookup"]["questions"] == 2
    assert summary["fact_lookup"]["average_latency_ms"] == 200.0
    assert summary["fact_lookup"]["hit@1"] == 0.5
    assert summary["fact_lookup"]["precision@1"] == 0.5
    assert summary["fact_lookup"]["recall@1"] == 0.25
    assert summary["comparison"]["questions"] == 1
    assert summary["comparison"]["evidence_coverage@1"] == 1.0


def test_question_type_summary_averages_evidence_coverage_only_for_annotated_questions():
    details = [
        {
            "question_type": "comparison",
            "latency_ms": 100.0,
            "hit@1": 1,
            "precision@1": 1.0,
            "recall@1": 1.0,
            "ndcg@1": 1.0,
            "ap@1": 1.0,
            "evidence_coverage@1": 1.0,
            "required_evidence_groups_count": 1,
            "rr@1": 1.0,
        },
        {
            "question_type": "comparison",
            "latency_ms": 300.0,
            "hit@1": 0,
            "precision@1": 0.0,
            "recall@1": 0.0,
            "ndcg@1": 0.0,
            "ap@1": 0.0,
            "evidence_coverage@1": 0.0,
            "required_evidence_groups_count": 0,
            "rr@1": 0.0,
        },
    ]

    summary = build_question_type_summary(details, [1])

    assert summary["comparison"]["questions"] == 2
    assert summary["comparison"]["precision@1"] == 0.5
    assert summary["comparison"]["evidence_coverage@1"] == 1.0
    assert summary["comparison"]["evidence_coverage_questions"] == 1
