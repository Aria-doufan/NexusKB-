import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_planner_uses_hard_filter_for_explicit_source_and_policy_type():
    from app.rag.metadata_filter_planner import plan_metadata_filter

    decision = plan_metadata_filter("Find the Confluence PTO policy", rag_intent="constrained", source_hints=[])

    assert decision.mode == "hard"
    assert decision.source_types == ["confluence"]
    assert decision.doc_semantic_types == ["policy_rule"]
    assert decision.confidence >= 0.8


def test_planner_uses_soft_filter_for_implicit_policy_query():
    from app.rag.metadata_filter_planner import plan_metadata_filter

    decision = plan_metadata_filter("What is our internal PTO rule?", rag_intent="procedure", source_hints=[])

    assert decision.mode == "soft"
    assert decision.source_types == []
    assert decision.doc_semantic_types == ["policy_rule"]
    assert 0.5 <= decision.confidence < 0.8


def test_planner_uses_none_for_open_ended_cross_source_query():
    from app.rag.metadata_filter_planner import plan_metadata_filter

    decision = plan_metadata_filter("Why did the contractor conversion process slip?", rag_intent="multi_hop", source_hints=[])

    assert decision.mode == "none"
    assert decision.source_types == []
    assert decision.doc_semantic_types == []


def test_planner_normalizes_router_source_hints_to_soft_filter():
    from app.rag.metadata_filter_planner import plan_metadata_filter

    decision = plan_metadata_filter("Where was the rollout discussed?", rag_intent="semantic_query", source_hints=["slack", "unknown"])

    assert decision.mode == "soft"
    assert decision.source_types == ["slack"]
    assert decision.doc_semantic_types == []
