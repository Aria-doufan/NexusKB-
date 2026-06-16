import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_default_event_message_returns_user_visible_progress_text():
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow

    assert RagEvidenceWorkflow._default_event_message("retrieval_started", {}) == "正在检索知识库……"
    assert RagEvidenceWorkflow._default_event_message("evaluation_finished", {}) == "正在评估证据质量……"
    assert RagEvidenceWorkflow._default_event_message("answer_started", {}) == "正在生成答案……"


def test_default_event_message_includes_retrieval_document_count():
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow

    assert (
        RagEvidenceWorkflow._default_event_message("retrieval_finished", {"selected_documents": 5})
        == "已找到 5 篇相关文档"
    )
    assert RagEvidenceWorkflow._default_event_message("retrieval_finished", {}) == "检索完成，正在整理证据"


def test_record_event_uses_default_message_when_explicit_message_is_missing():
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow
    from app.schemas.rag import RagState

    workflow = RagEvidenceWorkflow.__new__(RagEvidenceWorkflow)
    state = RagState(
        request_id="req-progress",
        debug_id="dbg-progress",
        user_id="user-progress",
        original_query="公司报销流程是什么？",
        current_query="公司报销流程是什么？",
    )

    workflow._record_event(
        state,
        "retrieval_finished",
        "retrieve",
        data={"selected_documents": 3},
    )

    assert state.sse_events[-1]["message"] == "已找到 3 篇相关文档"


def test_record_event_preserves_explicit_empty_message_for_mapped_event():
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow
    from app.schemas.rag import RagState

    workflow = RagEvidenceWorkflow.__new__(RagEvidenceWorkflow)
    state = RagState(
        request_id="req-progress",
        debug_id="dbg-progress",
        user_id="user-progress",
        original_query="公司报销流程是什么？",
        current_query="公司报销流程是什么？",
    )

    workflow._record_event(
        state,
        "retrieval_finished",
        "retrieve",
        message="",
        data={"selected_documents": 3},
    )

    assert state.sse_events[-1]["message"] == ""


def test_record_event_unmapped_event_without_message_writes_none():
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow
    from app.schemas.rag import RagState

    workflow = RagEvidenceWorkflow.__new__(RagEvidenceWorkflow)
    state = RagState(
        request_id="req-progress",
        debug_id="dbg-progress",
        user_id="user-progress",
        original_query="公司报销流程是什么？",
        current_query="公司报销流程是什么？",
    )

    workflow._record_event(
        state,
        "custom_unmapped_event",
        "custom_stage",
    )

    assert state.sse_events[-1]["message"] is None
