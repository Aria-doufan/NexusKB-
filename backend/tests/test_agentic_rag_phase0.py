import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_rag_state_serializes_router_decision_and_defaults():
    from app.schemas.rag import RagState

    state = RagState(
        request_id="req-1",
        debug_id="dbg-1",
        session_id="sess-1",
        user_id="user-1",
        original_query="Where is the PTO policy?",
        current_query="Where is the PTO policy?",
        rag_intent="constrained",
        source_hints=["confluence"],
        router_confidence=0.82,
        router_reason="Needs enterprise policy docs.",
    )

    data = state.model_dump()

    assert data["route"] == "enterprise_knowledge"
    assert data["retry_count"] == 0
    assert data["max_retries"] == 1
    assert data["rewritten_queries"] == []
    assert data["retrieval_attempts"] == []
    assert data["selected_documents"] == []
    assert data["rag_intent"] == "constrained"
    assert data["source_hints"] == ["confluence"]


def test_rag_response_exposes_public_debug_id_without_full_trace():
    from app.schemas.rag import (
        EvaluationSummary,
        RagMetrics,
        RagResponse,
        RagSource,
        RagStrategySummary,
    )

    response = RagResponse(
        request_id="req-1",
        debug_id="dbg-1",
        session_id="sess-1",
        answer="Use the HR policy page.",
        sources=[
            RagSource(
                source_id="doc-1",
                title="PTO Policy",
                source_type="confluence",
                parent_doc_id="parent-1",
                parent_chunk_id="chunk-1",
                score=0.91,
            )
        ],
        strategy=RagStrategySummary(strategy_name="default", retrieval_mode="hybrid", final_top_k=5),
        evaluation=EvaluationSummary(
            enough_evidence=True,
            covered_aspects=["policy location"],
            missing_aspects=[],
            user_visible_reason=None,
        ),
        metrics=RagMetrics(retry_count=0, retrieval_attempts=1, total_ms=12.5),
    )

    data = response.model_dump()

    assert data["debug_id"] == "dbg-1"
    assert data["sources"][0]["title"] == "PTO Policy"
    assert "retrieval_attempts" not in data
    assert "planner" not in data


def test_sse_event_formats_json_payload_with_request_metadata():
    from app.schemas.sse import SseEvent, format_sse_event

    event = SseEvent(
        event="rag_plan_created",
        request_id="req-1",
        debug_id="dbg-1",
        session_id="sess-1",
        stage="planner",
        message="Plan ready",
        data={"task_type": "fact_lookup"},
        timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=UTC),
    )

    raw = format_sse_event(event)

    assert raw.startswith("data: ")
    assert raw.endswith("\n\n")
    payload = json.loads(raw.removeprefix("data: ").strip())
    assert payload["event"] == "rag_plan_created"
    assert payload["debug_id"] == "dbg-1"
    assert payload["stage"] == "planner"
    assert payload["data"] == {"task_type": "fact_lookup"}


@pytest.mark.anyio
async def test_debug_trace_store_writes_jsonl_and_reads_by_debug_id(tmp_path):
    from app.schemas.rag import RagSource
    from app.schemas.rag_debug import RagDebugTrace
    from app.services.rag_debug_trace_store import DebugTraceStore

    trace = RagDebugTrace(
        request_id="req-1",
        debug_id="dbg-1",
        session_id="sess-1",
        user_id="user-1",
        started_at="2026-05-24T12:00:00Z",
        finished_at="2026-05-24T12:00:01Z",
        total_ms=1000.0,
        final_answer_preview="Use the HR policy page.",
        final_sources=[
            RagSource(
                source_id="doc-1",
                title="PTO Policy",
                source_type="confluence",
                parent_doc_id="parent-1",
                parent_chunk_id="chunk-1",
                score=0.91,
            )
        ],
    )
    store = DebugTraceStore(base_dir=tmp_path)

    await store.save(trace)
    loaded = await store.get("dbg-1")

    assert loaded is not None
    assert loaded.debug_id == "dbg-1"
    assert loaded.final_sources[0].title == "PTO Policy"
    trace_files = list(tmp_path.glob("*.jsonl"))
    assert len(trace_files) == 1
    assert json.loads(trace_files[0].read_text(encoding="utf-8").strip())["debug_id"] == "dbg-1"


def test_rag_state_requires_durable_request_and_debug_ids():
    from app.schemas.rag import RagState

    with pytest.raises(ValidationError):
        RagState(
            request_id="",
            debug_id="dbg-1",
            user_id="user-1",
            original_query="Question?",
            current_query="Question?",
        )
