import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import prepare_enterprise_rag_bench as prep


def test_policy_rule_numbered_rules_split_into_separate_chunks_near_threshold():
    text = (
        "1. Employees must complete security training before requesting production access.\n"
        "2. Managers must review access exceptions within one business day.\n"
        "3. Contractors must renew credentials every thirty calendar days."
    )

    chunks = prep.semantic_child_split(text, "policy_rule", child_chunk_size=86)

    assert len(chunks) == 3
    assert chunks[0].startswith("1. Employees")
    assert chunks[1].startswith("2. Managers")
    assert chunks[2].startswith("3. Contractors")
    assert all(len(chunk) <= 86 for chunk in chunks)


def test_meeting_notes_split_decision_action_item_and_notes_boundaries():
    text = (
        "Decision: ship the beta to design partners on Monday.\n"
        "Action item: Priya will publish the rollout checklist by Friday.\n"
        "Notes: Keep support staffed during the first import window."
    )

    chunks = prep.semantic_child_split(text, "meeting_notes", child_chunk_size=76)

    assert len(chunks) == 3
    assert chunks[0].startswith("Decision:")
    assert chunks[1].startswith("Action item:")
    assert chunks[2].startswith("Notes:")
    assert all(len(chunk) <= 76 for chunk in chunks)


def test_short_adjacent_policy_rules_merge_until_threshold():
    text = (
        "1. Use SSO.\n"
        "2. Rotate keys.\n"
        "3. Log access.\n"
        "4. Review exceptions monthly."
    )

    chunks = prep.semantic_child_split(text, "policy_rule", child_chunk_size=45)

    assert chunks == [
        "1. Use SSO.\n2. Rotate keys.\n3. Log access.",
        "4. Review exceptions monthly.",
    ]


def test_generic_doc_fallback_splits_long_generic_text_recursively():
    text = "Alpha beta gamma. " * 20

    chunks = prep.semantic_child_split(text, "generic_doc", child_chunk_size=70)

    assert chunks == prep.recursive_split(text, 70, prep.STRUCTURAL_SEPARATORS)
    assert len(chunks) > 1


def test_parent_child_chunking_uses_semantic_mode_when_configured():
    document = {
        "doc_id": "doc-semantic",
        "source_type": "notion",
        "doc_semantic_type": "meeting_notes",
        "title": "Launch sync",
        "text": (
            "Decision: release on Tuesday after final QA signoff.\n"
            "Action item: Omar will notify support about the release window.\n"
            "Notes: Watch import latency during the rollout."
        ),
        "is_gold": True,
    }

    _parent_chunks, child_chunks = prep.chunk_document_parent_child(
        document,
        prep.ParentChildConfig(
            parent_chunk_size=500,
            parent_chunk_overlap=0,
            child_chunk_size=76,
            child_chunk_overlap=0,
            child_boundary_mode="semantic",
        ),
    )

    assert [chunk["text"].splitlines()[0].split(":", 1)[0] for chunk in child_chunks] == [
        "Decision",
        "Action item",
        "Notes",
    ]


def test_parent_child_default_config_still_uses_recursive_child_splitting(monkeypatch):
    def fail_if_semantic_split_is_used(*_args, **_kwargs):
        raise AssertionError("default parent-child config must not use semantic child splitting")

    monkeypatch.setattr(
        "scripts.prepare_enterprise_rag_bench.semantic_child_split",
        fail_if_semantic_split_is_used,
    )
    document = {
        "doc_id": "doc-default",
        "source_type": "notion",
        "title": "Launch sync",
        "text": "Decision: go.\nAction item: tell support.\nNotes: watch latency.",
        "is_gold": False,
    }

    parent_chunks, child_chunks = prep.chunk_document_parent_child(
        document,
        prep.ParentChildConfig(
            parent_chunk_size=500,
            parent_chunk_overlap=0,
            child_chunk_size=500,
            child_chunk_overlap=0,
        ),
    )

    assert len(child_chunks) == 1
    assert child_chunks[0]["text"] == document["text"]
    assert {chunk["doc_semantic_type"] for chunk in parent_chunks} == {"generic_doc"}
    assert {chunk["doc_semantic_type"] for chunk in child_chunks} == {"generic_doc"}
