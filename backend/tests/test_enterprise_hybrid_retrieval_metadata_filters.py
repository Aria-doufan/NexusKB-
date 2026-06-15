import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_build_metadata_filter_decision_accepts_allowed_values():
    from scripts.evaluate_enterprise_hybrid_retrieval import build_metadata_filter_decision

    decision = build_metadata_filter_decision(
        filter_mode="hard",
        source_types="confluence,jira",
        doc_semantic_types="policy_rule,issue_ticket",
    )

    assert decision.mode == "hard"
    assert decision.source_types == ["confluence", "jira"]
    assert decision.doc_semantic_types == ["policy_rule", "issue_ticket"]


def test_build_metadata_filter_decision_rejects_unknown_values():
    from scripts.evaluate_enterprise_hybrid_retrieval import build_metadata_filter_decision

    try:
        build_metadata_filter_decision(filter_mode="hard", source_types="drop_index", doc_semantic_types="policy_rule")
    except ValueError as exc:
        assert "Unsupported source_type" in str(exc)
    else:
        raise AssertionError("Unknown source_type should be rejected")
