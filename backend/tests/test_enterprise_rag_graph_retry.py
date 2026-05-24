import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SequenceEnterpriseRagService:
    def __init__(self, result_sequence):
        self.result_sequence = list(result_sequence)
        self.retrieve_calls = []
        self.generated_queries = []

    async def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        if self.result_sequence:
            return self.result_sequence.pop(0)
        return []

    async def generate_answer(self, query, documents, memory_context=None):
        self.generated_queries.append(query)
        return f"answer after retry for {query}"


class CapturingTraceStore:
    def __init__(self):
        self.saved = []

    async def save(self, trace):
        self.saved.append(trace)


def make_document(title="Policy"):
    return {
        "parent_doc_id": "parent-1",
        "parent_chunk_id": "chunk-1",
        "source_type": "confluence",
        "title": title,
        "section_heading": "Overview",
        "score": 0.8,
        "parent_text": "Relevant enterprise evidence.",
        "child_text": "Relevant evidence",
        "metadata": {"source_type": "confluence"},
    }


@pytest.mark.anyio
async def test_weak_evidence_triggers_one_query_rewrite_retry_and_records_trace():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState

    service = SequenceEnterpriseRagService([[], [make_document()]])
    trace_store = CapturingTraceStore()
    graph = EnterpriseRagGraph(service=service, trace_store=trace_store)
    state = RagState(
        request_id="req-retry",
        debug_id="dbg-retry",
        session_id="sess-1",
        user_id="user-1",
        original_query="PTO?",
        current_query="PTO?",
        rag_intent="semantic",
        source_hints=["confluence"],
        max_retries=1,
    )

    response = await graph.run(state)

    assert response.metrics.retry_count == 1
    assert response.metrics.retrieval_attempts == 2
    assert len(service.retrieve_calls) == 2
    assert service.retrieve_calls[1]["query"] != service.retrieve_calls[0]["query"]
    assert service.retrieve_calls[1]["source_hints"] == ["confluence"]
    assert state.rewritten_queries == [service.retrieve_calls[1]["query"]]
    assert response.evaluation.enough_evidence is True
    assert response.sources[0].title == "Policy"
    trace = trace_store.saved[0]
    assert len(trace.retrieval_attempts) == 2
    assert len(trace.evaluations) == 2
    assert trace.retrieval_attempts[1].attempt.reason == "Retry after rewrite_query."
    event_names = [event["event"] for event in state.sse_events]
    assert "retry_decided" in event_names
    assert "query_rewritten" in event_names
    assert event_names.count("retrieval_started") == 2
    assert event_names.count("evaluation_finished") == 2


@pytest.mark.anyio
async def test_retry_budget_prevents_second_retrieval_after_max_retries():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState

    service = SequenceEnterpriseRagService([[]])
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore())
    state = RagState(
        request_id="req-budget",
        debug_id="dbg-budget",
        user_id="user-1",
        original_query="Unknown?",
        current_query="Unknown?",
        retry_count=1,
        max_retries=1,
    )

    response = await graph.run(state)

    assert len(service.retrieve_calls) == 1
    assert response.metrics.retry_count == 1
    assert response.metrics.retrieval_attempts == 1
    assert response.evaluation.enough_evidence is False
    assert "没有找到足够信息" in response.answer


@pytest.mark.anyio
async def test_acl_hard_stop_blocks_generation_and_sources_even_with_documents():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState

    service = SequenceEnterpriseRagService([[make_document()]])
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore())
    state = RagState(
        request_id="req-acl",
        debug_id="dbg-acl",
        user_id="user-1",
        original_query="Restricted policy?",
        current_query="Restricted policy?",
        acl_filter_removed_all_candidates=True,
    )

    response = await graph.run(state)

    assert len(service.retrieve_calls) == 1
    assert service.generated_queries == []
    assert response.sources == []
    assert response.evaluation.enough_evidence is False
    assert response.evaluation.missing_aspects == ["Restricted policy?"]
