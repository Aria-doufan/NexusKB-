import json
import os
import sys
from pathlib import Path
from types import MethodType

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
async def test_router_invoke_exposes_debug_id_for_enterprise_response():
    from app.agent.router_graph import RouterGraph

    class FakeCompiledGraph:
        async def ainvoke(self, state):
            return {
                **state,
                "route": "enterprise_knowledge",
                "answer": "graph answer",
                "debug_id": "dbg-public",
                "request_id": "req-public",
            }

    router = object.__new__(RouterGraph)
    router.graph = FakeCompiledGraph()

    result = await router.invoke("Where is PTO?", "user-1", "sess-1")

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
async def test_router_stream_delegated_chat_events_include_debug_id(monkeypatch):
    import app.agent.router_graph as router_graph_module
    from app.agent.router_graph import RouterGraph

    router = object.__new__(RouterGraph)

    async def fake_load_context(self, state):
        return {"session_id": state["session_id"], "history": []}

    async def fake_llm_router(self, state):
        return {"route": "chat", "rag_intent": "unknown", "source_hints": [], "confidence": 0.9, "reason": "chat"}

    async def fake_chat_stream_response(query, session_id, user_id, history, long_term_memories=None):
        yield 'data: {"type": "response", "content": "chat answer", "session_id": "sess-1"}\n\n'
        yield 'data: {"type": "done", "session_id": "sess-1"}\n\n'

    router.load_context = MethodType(fake_load_context, router)
    router.llm_router = MethodType(fake_llm_router, router)
    monkeypatch.setattr(router_graph_module, "get_chat_stream_response", fake_chat_stream_response)

    events = []
    async for raw_event in router.stream("hello", "user-1", "sess-1"):
        events.append(json.loads(raw_event.removeprefix("data: ").strip()))

    response_event = next(event for event in events if event.get("type") == "response")
    done_event = next(event for event in events if event.get("type") == "done")
    assert response_event["debug_id"]
    assert response_event["request_id"]
    assert done_event["debug_id"] == response_event["debug_id"]
    assert done_event["request_id"] == response_event["request_id"]


@pytest.mark.anyio
async def test_router_stream_final_response_event_includes_debug_id_for_enterprise_route():
    from app.agent.router_graph import RouterGraph

    router = object.__new__(RouterGraph)

    async def fake_load_context(self, state):
        return {"session_id": state["session_id"], "history": []}

    async def fake_llm_router(self, state):
        return {
            "route": "enterprise_knowledge",
            "rag_intent": "semantic",
            "source_hints": ["confluence"],
            "confidence": 0.9,
            "reason": "Needs enterprise docs.",
        }

    async def fake_enterprise_knowledge_node(self, state):
        return {
            "answer": "graph answer",
            "debug_id": "dbg-stream",
            "request_id": "req-stream",
            "sse_events": [],
        }

    async def fake_persist_message(self, state):
        return {}

    router.load_context = MethodType(fake_load_context, router)
    router.llm_router = MethodType(fake_llm_router, router)
    router.enterprise_knowledge_node = MethodType(fake_enterprise_knowledge_node, router)
    router.persist_message = MethodType(fake_persist_message, router)

    events = []
    async for raw_event in router.stream("Where is PTO?", "user-1", "sess-1"):
        events.append(json.loads(raw_event.removeprefix("data: ").strip()))

    response_event = next(event for event in events if event.get("type") == "response")
    assert response_event["debug_id"] == "dbg-stream"
    assert response_event["request_id"] == "req-stream"


@pytest.mark.anyio
async def test_router_enterprise_node_exposes_web_reference_sources():
    from app.agent.router_graph import RouterGraph
    from app.schemas.rag import EvaluationSummary, RagMetrics, RagResponse, RagSource, RagStrategySummary

    class FakeWebFallbackRagGraph:
        async def run(self, state):
            return RagResponse(
                request_id=state.request_id,
                debug_id=state.debug_id,
                session_id=state.session_id,
                answer="公司知识库未找到足够资料。以下为通用参考。",
                sources=[
                    RagSource(
                        source_id="web:https://example.test/expense",
                        title="通用报销流程参考",
                        source_type="web_reference",
                        score=0.8,
                        metadata={
                            "url": "https://example.test/expense",
                            "snippet": "提交申请、主管审批、财务复核。",
                            "reference_scope": "general_public_reference",
                        },
                    )
                ],
                strategy=RagStrategySummary(strategy_name="default", retrieval_mode="hybrid", final_top_k=5),
                evaluation=EvaluationSummary(enough_evidence=False, covered_aspects=[], missing_aspects=[state.original_query]),
                metrics=RagMetrics(retry_count=0, retrieval_attempts=1, web_search_ms=12.0),
            )

    router = object.__new__(RouterGraph)
    router.enterprise_rag_graph = FakeWebFallbackRagGraph()
    state = {
        "query": "如果公司知识库没有报销流程，给我一个通用流程参考",
        "user_id": "user-1",
        "session_id": "sess-web-source",
        "request_id": "req-web-source",
        "debug_id": "dbg-web-source",
        "rag_intent": "procedure",
        "source_hints": [],
        "confidence": 0.88,
        "reason": "Needs policy source.",
        "long_term_memories": [],
    }

    result = await router.enterprise_knowledge_node(state)

    assert result["documents"][0]["source_type"] == "web_reference"
    assert result["documents"][0]["metadata"]["reference_scope"] == "general_public_reference"
    assert result["steps"][0]["tool_output"]["sources"][0]["source_type"] == "web_reference"
