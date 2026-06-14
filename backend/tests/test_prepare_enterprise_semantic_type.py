import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import prepare_enterprise_rag_bench as prep


def infer_doc_semantic_type(source_type: str, title: str, content: str) -> str:
    assert hasattr(prep, "infer_doc_semantic_type")
    return prep.infer_doc_semantic_type(source_type, title, content)


@pytest.mark.parametrize(
    ("source_type", "expected"),
    [
        ("fireflies", "meeting_notes"),
        ("slack", "chat_thread"),
        ("gmail", "email_thread"),
        ("hubspot", "account_notes"),
        ("jira", "issue_ticket"),
        ("linear", "issue_ticket"),
    ],
)
def test_infer_doc_semantic_type_uses_source_specific_labels(source_type, expected):
    assert infer_doc_semantic_type(source_type, "Quarterly RFC", "Architecture proposal") == expected


def test_infer_doc_semantic_type_classifies_github_code_changes_before_issue_patterns():
    assert infer_doc_semantic_type(
        "github",
        "Pull request: fix regression in sync job",
        "LGTM after reviewing the diff and merge commit.",
    ) == "code_change"


def test_infer_doc_semantic_type_classifies_github_issue_tickets():
    assert infer_doc_semantic_type(
        "github",
        "Bug: checkout failure incident",
        "Description: users see an error after deploy.",
    ) == "issue_ticket"


def test_infer_doc_semantic_type_prefers_github_issue_indicators_over_generic_review():
    assert infer_doc_semantic_type(
        "github",
        "Bug: security review failure",
        "Description: users see an error after deploy.",
    ) == "issue_ticket"


def test_infer_doc_semantic_type_defaults_github_to_code_change():
    assert infer_doc_semantic_type(
        "github",
        "Repository maintenance",
        "Adjusted labels and ownership notes for team visibility.",
    ) == "code_change"


def test_infer_doc_semantic_type_does_not_treat_github_description_as_issue():
    assert infer_doc_semantic_type(
        "github",
        "Repository maintenance",
        "Description: Adjusted labels and ownership notes for team visibility.",
    ) == "code_change"


@pytest.mark.parametrize(
    ("title", "content", "expected"),
    [
        ("Security gating policy", "Employees must follow this compliance rule.", "policy_rule"),
        ("Customer onboarding playbook", "Checklist for when to use the escalation workflow.", "playbook"),
        ("Billing help center", "Q: How do refunds work?", "faq"),
        ("Search architecture RFC", "Technical spec and ADR for retrieval design.", "technical_doc"),
        ("Weekly sync", "Decision: proceed with phased rollout.", "meeting_notes"),
        ("Quarterly update", "Plain status update without matching keywords.", "generic_doc"),
    ],
)
def test_infer_doc_semantic_type_uses_document_patterns(title, content, expected):
    assert infer_doc_semantic_type("notion", title, content) == expected


def test_normalize_document_adds_doc_semantic_type():
    document = prep.normalize_document(
        {
            "doc_id": "doc-1",
            "source_type": "notion",
            "title": "Engineering handbook",
            "content": "All launches must complete security-gating review.",
        },
        is_gold=True,
    )

    assert document["doc_semantic_type"] == "policy_rule"


def test_chunk_document_inherits_doc_semantic_type():
    document = prep.normalize_document(
        {
            "doc_id": "doc-2",
            "source_type": "slack",
            "title": "Release thread",
            "content": "Discuss launch status and blockers.",
        },
        is_gold=False,
    )

    chunks = prep.chunk_document(document, prep.ChunkConfig(chunk_size=80, chunk_overlap=0))

    assert chunks
    assert {chunk["doc_semantic_type"] for chunk in chunks} == {"chat_thread"}


def test_chunk_document_old_shape_defaults_doc_semantic_type_to_generic_doc():
    document = {
        "doc_id": "doc-old-shape",
        "source_type": "notion",
        "title": "Legacy export",
        "text": "Legacy exports do not include semantic type metadata.",
        "is_gold": False,
    }

    chunks = prep.chunk_document(document, prep.ChunkConfig(chunk_size=80, chunk_overlap=0))

    assert chunks
    assert {chunk["doc_semantic_type"] for chunk in chunks} == {"generic_doc"}


def test_parent_child_chunks_inherit_doc_semantic_type():
    document = prep.normalize_document(
        {
            "doc_id": "doc-3",
            "source_type": "github",
            "title": "PR 42 merge diff",
            "content": "Review comments say LGTM on the commit diff.",
        },
        is_gold=False,
    )

    parent_chunks, child_chunks = prep.chunk_document_parent_child(
        document,
        prep.ParentChildConfig(
            parent_chunk_size=90,
            parent_chunk_overlap=0,
            child_chunk_size=60,
            child_chunk_overlap=0,
        ),
    )

    assert parent_chunks
    assert child_chunks
    assert {chunk["doc_semantic_type"] for chunk in parent_chunks} == {"code_change"}
    assert {chunk["doc_semantic_type"] for chunk in child_chunks} == {"code_change"}
