import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.decomposition import (
    SubQuery,
    SubQueryPlan,
    build_fallback_plan,
    merge_decomposed_scores,
    parse_sub_query_plan,
    should_decompose_intent,
)


def test_parse_sub_query_plan_accepts_valid_json_array():
    raw = json.dumps(
        {
            "sub_queries": [
                {
                    "id": "sq1",
                    "query": "试用期员工请假审批流程是什么？",
                    "purpose": "fact",
                    "depends_on": [],
                },
                {
                    "id": "sq2",
                    "query": "正式员工请假审批流程是什么？",
                    "purpose": "comparison_dimension",
                    "depends_on": [],
                },
            ]
        },
        ensure_ascii=False,
    )

    plan = parse_sub_query_plan(raw, original_query="试用期和正式员工请假流程有什么区别？")

    assert plan.original_query == "试用期和正式员工请假流程有什么区别？"
    assert [item.id for item in plan.sub_queries] == ["sq1", "sq2"]
    assert plan.sub_queries[0].purpose == "fact"
    assert plan.fallback_reason is None


def test_parse_sub_query_plan_trims_valid_sub_query_content():
    raw = json.dumps(
        {
            "sub_queries": [
                {
                    "id": " sq1 ",
                    "query": " 试用期员工请假审批流程是什么？ ",
                    "purpose": "fact",
                    "depends_on": [],
                },
                {
                    "id": " sq2 ",
                    "query": " 正式员工请假审批流程是什么？ ",
                    "purpose": "comparison_dimension",
                    "depends_on": [" sq1 "],
                },
            ]
        },
        ensure_ascii=False,
    )

    plan = parse_sub_query_plan(raw, original_query="试用期和正式员工请假流程有什么区别？")

    assert [item.id for item in plan.sub_queries] == ["sq1", "sq2"]
    assert [item.query for item in plan.sub_queries] == [
        "试用期员工请假审批流程是什么？",
        "正式员工请假审批流程是什么？",
    ]
    assert plan.sub_queries[1].depends_on == ["sq1"]
    assert plan.fallback_reason is None


def test_parse_sub_query_plan_rejects_unknown_dependency():
    raw = json.dumps(
        {
            "sub_queries": [
                {
                    "id": "sq1",
                    "query": "试用期员工请假审批流程是什么？",
                    "purpose": "fact",
                    "depends_on": [],
                },
                {
                    "id": "sq2",
                    "query": "正式员工请假审批流程是什么？",
                    "purpose": "comparison_dimension",
                    "depends_on": ["sq3"],
                },
            ]
        },
        ensure_ascii=False,
    )

    plan = parse_sub_query_plan(raw, original_query="试用期和正式员工请假流程有什么区别？")

    assert plan.fallback_reason == "invalid_sub_query_content"
    assert plan.sub_queries == []


def test_parse_sub_query_plan_rejects_single_sub_query():
    raw = json.dumps(
        {
            "sub_queries": [
                {
                    "id": "sq1",
                    "query": "试用期员工请假审批流程是什么？",
                    "purpose": "fact",
                    "depends_on": [],
                }
            ]
        },
        ensure_ascii=False,
    )

    plan = parse_sub_query_plan(raw, original_query="试用期和正式员工请假流程有什么区别？")

    assert plan.sub_queries == []
    assert plan.fallback_reason == "invalid_sub_query_count"


def test_parse_sub_query_plan_rejects_empty_or_duplicate_queries():
    raw = json.dumps(
        {
            "sub_queries": [
                {"id": "sq1", "query": "", "purpose": "fact", "depends_on": []},
                {"id": "sq2", "query": "正式员工请假审批流程是什么？", "purpose": "fact", "depends_on": []},
                {"id": "sq3", "query": "正式员工请假审批流程是什么？", "purpose": "fact", "depends_on": []},
            ]
        },
        ensure_ascii=False,
    )

    plan = parse_sub_query_plan(raw, original_query="请比较试用期和正式员工请假流程")

    assert plan.sub_queries == []
    assert plan.fallback_reason == "invalid_sub_query_content"


def test_parse_sub_query_plan_rejects_duplicate_queries_without_empty_query():
    raw = json.dumps(
        {
            "sub_queries": [
                {"id": "sq1", "query": "试用期员工请假审批流程是什么？", "purpose": "fact", "depends_on": []},
                {"id": "sq2", "query": "正式员工请假审批流程是什么？", "purpose": "fact", "depends_on": []},
                {"id": "sq3", "query": "正式员工请假审批流程是什么？", "purpose": "comparison_dimension", "depends_on": []},
            ]
        },
        ensure_ascii=False,
    )

    plan = parse_sub_query_plan(raw, original_query="请比较试用期和正式员工请假流程")

    assert plan.fallback_reason == "invalid_sub_query_content"
    assert plan.sub_queries == []


def test_parse_sub_query_plan_rejects_non_json_text():
    plan = parse_sub_query_plan("我认为可以拆成两个问题", original_query="A 和 B 有什么区别？")

    assert plan.sub_queries == []
    assert plan.fallback_reason == "invalid_json"


def test_build_fallback_plan_keeps_original_query_and_reason():
    plan = build_fallback_plan("VPN 怎么申请？", "decomposition_failed")

    assert plan.original_query == "VPN 怎么申请？"
    assert plan.sub_queries == []
    assert plan.fallback_reason == "decomposition_failed"


def test_should_decompose_intent_only_allows_complex_intents():
    assert should_decompose_intent("multi_hop") is True
    assert should_decompose_intent("comparison") is True
    assert should_decompose_intent("semantic_query") is False
    assert should_decompose_intent("constrained") is False
    assert should_decompose_intent("unknown") is False


def test_sub_query_model_rejects_unknown_purpose():
    try:
        SubQuery(id="sq1", query="问题", purpose="unsupported", depends_on=[])
    except ValueError as exc:
        assert "purpose" in str(exc)
    else:
        raise AssertionError("SubQuery accepted unsupported purpose")


def test_sub_query_plan_helper_returns_texts():
    plan = SubQueryPlan(
        original_query="比较 A 和 B",
        sub_queries=[
            SubQuery(id="sq1", query="A 是什么？", purpose="fact"),
            SubQuery(id="sq2", query="B 是什么？", purpose="fact"),
        ],
    )

    assert plan.query_texts() == ["A 是什么？", "B 是什么？"]


def test_merge_decomposed_scores_rewards_evidence_coverage():
    merged = merge_decomposed_scores(
        {
            "sq1": {"parent_a": 0.0200, "parent_b": 0.0195},
            "sq2": {"parent_b": 0.0195},
        },
        total_sub_queries=2,
        coverage_weight=0.25,
    )

    assert merged["parent_a"].matched_sub_query_ids == ["sq1"]
    assert merged["parent_a"].coverage_score == 0.5
    assert merged["parent_b"].matched_sub_query_ids == ["sq1", "sq2"]
    assert merged["parent_b"].coverage_score == 1.0
    assert merged["parent_b"].final_score > merged["parent_a"].final_score


def test_merge_decomposed_scores_keeps_max_base_score_per_candidate():
    merged = merge_decomposed_scores(
        {
            "sq1": {"parent_a": 0.0100},
            "sq2": {"parent_a": 0.0200},
        },
        total_sub_queries=2,
    )

    assert merged["parent_a"].fused_score == 0.0200
    assert merged["parent_a"].coverage_score == 1.0
    assert merged["parent_a"].final_score == 0.025
