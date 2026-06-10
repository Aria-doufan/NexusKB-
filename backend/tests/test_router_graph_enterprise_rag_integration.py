import json
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEnterpriseRagGraph:
    def __init__(self):
        self.received_state = None

    async def run(self, state):
        from app.schemas.rag import EvaluationSummary, RagMetrics, RagResponse, RagSource, RagStrategySummary

        self.received_state = state
        return RagResponse(
            request_id=state.request_id,
            debug_id=state.debug_id,
            session_id=state.session_id,
            answer="graph answer",
            sources=[
                RagSource(
                    source_id="chunk-1",
                    title="Policy",
                    source_type="confluence",
                    parent_doc_id="parent-1",
                    parent_chunk_id="chunk-1",
                    score=0.8,
                )
            ],
            strategy=RagStrategySummary(strategy_name="default", retrieval_mode="hybrid", final_top_k=5),
            evaluation=EvaluationSummary(
                enough_evidence=True,
                covered_aspects=["question"],
                missing_aspects=[],
                user_visible_reason=None,
            ),
            metrics=RagMetrics(retry_count=0, retrieval_attempts=1, total_ms=10),
        )


@pytest.mark.anyio
async def test_router_enterprise_node_delegates_to_enterprise_rag_graph_with_rag_state():
    from app.agent.router_graph import RouterGraph

    router = object.__new__(RouterGraph)
    router.enterprise_rag_graph = FakeEnterpriseRagGraph()
    state = {
        "query": "Where is the PTO policy?",
        "user_id": "user-1",
        "session_id": "sess-1",
        "request_id": "req-1",
        "debug_id": "dbg-1",
        "rag_intent": "constrained",
        "source_hints": ["confluence"],
        "confidence": 0.88,
        "reason": "Needs policy source.",
        "long_term_memories": [
            {
                "id": "memory-1",
                "memory": "User is asking about the NexusKB deployment project.",
                "memory_type": "project",
                "score": 0.91,
                "created_at": "2026-05-24T00:00:00Z",
            }
        ],
    }

    result = await router.enterprise_knowledge_node(state)

    assert result["answer"] == "graph answer"
    assert result["debug_id"] == "dbg-1"
    assert result["documents"][0]["title"] == "Policy"
    assert result["steps"][0]["tool"] == "enterprise_rag_graph"
    assert result["steps"][0]["tool_output"]["debug_id"] == "dbg-1"
    rag_state = router.enterprise_rag_graph.received_state
    assert rag_state.original_query == "Where is the PTO policy?"
    assert rag_state.current_query == "Where is the PTO policy?"
    assert rag_state.source_hints == ["confluence"]
    assert rag_state.router_confidence == 0.88
    assert rag_state.memory_context.recalled[0].content == "User is asking about the NexusKB deployment project."
    assert rag_state.memory_context.recalled[0].category == "project_context"
    assert rag_state.memory_context.recalled[0].relevance_score == 0.91
    assert rag_state.memory_context.recalled[0].source == "long_term"


@pytest.mark.anyio
async def test_router_invoke_exposes_debug_id_for_agentic_rag_response():
    from app.agent.router_graph import RouterGraph

    class FakeAgenticRagGraph:
        async def invoke(self, query, user_id, session_id=None):
            return {
                "session_id": session_id,
                "request_id": "req-public",
                "debug_id": "dbg-public",
                "rag_intent": "constrained",
                "source_hints": ["confluence"],
                "confidence": 0.88,
                "reason": "Needs policy source.",
                "response": "graph answer",
                "steps": [],
                "error": None,
            }

    router = object.__new__(RouterGraph)
    router.agentic_rag_graph = FakeAgenticRagGraph()

    result = await router.invoke("Where is PTO?", "user-1", "sess-1")

    assert result["route"] == "agentic_rag"
    assert result["debug_id"] == "dbg-public"
    assert result["request_id"] == "req-public"


def test_router_response_schema_includes_debug_id():
    from app.schemas.models import RouterResponse

    response = RouterResponse(
        session_id="sess-1",
        route="enterprise_knowledge",
        response="answer",
        debug_id="dbg-public",
        request_id="req-public",
    )

    assert response.debug_id == "dbg-public"
    assert response.request_id == "req-public"


@pytest.mark.anyio
async def test_router_stream_agentic_rag_events_include_debug_id():
    from app.agent.router_graph import RouterGraph

    class FakeAgenticRagGraph:
        async def invoke(self, query, user_id, session_id=None):
            return {
                "session_id": session_id,
                "request_id": "req-chat",
                "debug_id": "dbg-chat",
                "rag_intent": "unknown",
                "source_hints": [],
                "confidence": 0.9,
                "reason": "General chat.",
                "response": "chat answer",
                "steps": [],
                "error": None,
            }

    router = object.__new__(RouterGraph)
    router.agentic_rag_graph = FakeAgenticRagGraph()

    events = []
    async for raw_event in router.stream("hello", "user-1", "sess-1"):
        events.append(json.loads(raw_event.removeprefix("data: ").strip()))

    response_event = next(event for event in events if event.get("type") == "response")
    done_event = next(event for event in events if event.get("type") == "done")
    assert response_event["debug_id"] == "dbg-chat"
    assert response_event["request_id"] == "req-chat"
    assert done_event["debug_id"] == response_event["debug_id"]
    assert done_event["request_id"] == response_event["request_id"]


@pytest.mark.anyio
async def test_router_stream_final_response_event_includes_debug_id_for_agentic_rag_route():
    from app.agent.router_graph import RouterGraph

    class FakeAgenticRagGraph:
        async def invoke(self, query, user_id, session_id=None):
            return {
                "session_id": session_id,
                "request_id": "req-stream",
                "debug_id": "dbg-stream",
                "rag_intent": "semantic_query",
                "source_hints": ["confluence"],
                "confidence": 0.9,
                "reason": "Needs enterprise docs.",
                "response": "graph answer",
                "steps": [],
                "error": None,
            }

    router = object.__new__(RouterGraph)
    router.agentic_rag_graph = FakeAgenticRagGraph()

    events = []
    async for raw_event in router.stream("Where is PTO?", "user-1", "sess-1"):
        events.append(json.loads(raw_event.removeprefix("data: ").strip()))

    route_event = next(event for event in events if event.get("type") == "route")
    response_event = next(event for event in events if event.get("type") == "response")
    assert route_event["route"] == "agentic_rag"
    assert route_event["rag_intent"] == "semantic_query"
    assert response_event["debug_id"] == "dbg-stream"
    assert response_event["request_id"] == "req-stream"


@pytest.mark.anyio
async def test_router_graph_delegates_all_requests_to_agentic_rag_graph(monkeypatch):
    from app.agent import router_graph as router_module
    from app.agent.router_graph import RouterGraph

    calls = []

    class FakeAgenticRagGraph:
        async def invoke(self, query, user_id, session_id=None):
            calls.append({"query": query, "user_id": user_id, "session_id": session_id})
            return {
                "session_id": session_id or "generated-session",
                "request_id": "req-1",
                "debug_id": "dbg-1",
                "intent": "general_chat",
                "action": "direct_answer",
                "rag_intent": "unknown",
                "source_hints": [],
                "confidence": 0.9,
                "reason": "General answer.",
                "response": "hello",
                "steps": [{"tool": "agentic_rag_graph"}],
                "error": None,
            }

    monkeypatch.setattr(router_module, "AgenticRagGraph", lambda: FakeAgenticRagGraph())

    graph = RouterGraph()
    result = await graph.invoke("hello", user_id="user-1", session_id="sess-1")

    assert calls == [{"query": "hello", "user_id": "user-1", "session_id": "sess-1"}]
    assert result["route"] == "agentic_rag"
    assert result["response"] == "hello"
    assert result["steps"] == [{"tool": "agentic_rag_graph"}]


@pytest.mark.anyio
async def test_router_graph_stream_includes_agentic_rag_events(monkeypatch):
    from app.agent import router_graph as router_module
    from app.agent.router_graph import RouterGraph

    class FakeAgenticRagGraph:
        async def invoke(self, query, user_id, session_id=None):
            return {
                "session_id": session_id or "generated-session",
                "request_id": "req-1",
                "debug_id": "dbg-1",
                "rag_intent": "constrained",
                "source_hints": ["confluence"],
                "confidence": 0.9,
                "reason": "Needs KB.",
                "response": "answer",
                "steps": [],
                "error": None,
                "sse_events": [
                    {"type": "rag_event", "event": "retrieval_started", "request_id": "req-1"},
                    {"type": "rag_event", "event": "retrieval_finished", "request_id": "req-1"},
                ],
            }

    monkeypatch.setattr(router_module, "AgenticRagGraph", lambda: FakeAgenticRagGraph())

    graph = RouterGraph()
    events = [event async for event in graph.stream("Where is PTO?", "user-1", "sess-1")]
    payloads = [json.loads(event.removeprefix("data: ").strip()) for event in events]

    assert [payload.get("event") for payload in payloads if payload.get("type") == "rag_event"] == [
        "retrieval_started",
        "retrieval_finished",
    ]
    assert payloads[-1]["type"] == "done"


def test_router_response_accepts_agentic_rag_route():
    from app.schemas.models import RouterResponse

    response = RouterResponse(
        session_id="sess-1",
        route="agentic_rag",
        request_id="req-1",
        debug_id="dbg-1",
        rag_intent="general_chat",
        source_hints=[],
        confidence=0.9,
        reason="Handled by single Agentic RAG graph.",
        response="hello",
        steps=[],
        error=None,
    )

    assert response.route == "agentic_rag"
