import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_agentic_rag_state_defaults_support_single_graph_actions():
    from app.schemas.rag import AgenticActionDecision, RagState, ToolExecutionResult

    state = RagState(
        request_id="req-1",
        debug_id="dbg-1",
        user_id="user-1",
        original_query="What is LangGraph?",
        current_query="What is LangGraph?",
    )

    assert state.action == "retrieve"
    assert state.response_type == "answer"
    assert state.history == []
    assert state.memory_summary == ""
    assert state.long_term_memories == []
    assert state.required_tools == []
    assert state.tool_results == []

    decision = AgenticActionDecision(
        intent="general_chat",
        action="direct_answer",
        needs_retrieval=False,
        needs_tool=False,
        needs_clarification=False,
        safety_risk=False,
        confidence=0.91,
        reason="General explanation does not need enterprise evidence.",
    )
    assert decision.action == "direct_answer"

    tool_result = ToolExecutionResult(
        tool_name="what_time_is_now",
        tool_input={},
        output="当前时间是：2026-06-10 10:30",
        success=True,
    )
    assert tool_result.success is True


class StubTraceStore:
    def __init__(self):
        self.saved = []

    async def save(self, trace):
        self.saved.append(trace)


class StaticDecisionChain:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    async def ainvoke(self, payload):
        self.calls.append(payload)
        return self.decision


@pytest.mark.anyio
async def test_agentic_rag_graph_direct_answer_does_not_retrieve():
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.schemas.rag import AgenticActionDecision, RagState

    class NoRetrieveService:
        async def retrieve_with_details(self, **kwargs):
            raise AssertionError("direct_answer must not retrieve")

        async def generate_answer(self, *args, **kwargs):
            raise AssertionError("direct_answer must not call RAG generation")

    graph = AgenticRagGraph(
        service=NoRetrieveService(),
        trace_store=StubTraceStore(),
        decision_chain=StaticDecisionChain(
            AgenticActionDecision(
                intent="general_chat",
                action="direct_answer",
                needs_retrieval=False,
                confidence=0.9,
                reason="General explanation.",
            )
        ),
    )
    state = RagState(
        request_id="req-direct",
        debug_id="dbg-direct",
        user_id="user-1",
        original_query="What is LangGraph?",
        current_query="What is LangGraph?",
    )

    response = await graph.run(state)

    assert response.answer == "这是一个通用问题，可以不检索企业知识库直接回答：What is LangGraph?"
    assert response.sources == []
    assert response.strategy.query_type == "general_chat"
    assert state.action == "direct_answer"
    assert state.response_type == "answer"


@pytest.mark.anyio
async def test_agentic_rag_graph_clarifies_inside_single_graph():
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.schemas.rag import AgenticActionDecision, RagState

    graph = AgenticRagGraph(
        trace_store=StubTraceStore(),
        decision_chain=StaticDecisionChain(
            AgenticActionDecision(
                intent="clarify",
                action="clarify",
                needs_retrieval=False,
                needs_clarification=True,
                confidence=0.4,
                reason="Missing target.",
            )
        ),
    )
    state = RagState(
        request_id="req-clarify",
        debug_id="dbg-clarify",
        user_id="user-1",
        original_query="帮我查一下那个",
        current_query="帮我查一下那个",
    )

    response = await graph.run(state)

    assert "需要再确认" in response.answer
    assert response.sources == []
    assert state.action == "clarify"
    assert state.response_type == "clarification"


@pytest.mark.anyio
async def test_agentic_rag_graph_refuses_inside_single_graph():
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.schemas.rag import AgenticActionDecision, RagState

    graph = AgenticRagGraph(
        trace_store=StubTraceStore(),
        decision_chain=StaticDecisionChain(
            AgenticActionDecision(
                intent="unsafe",
                action="refuse",
                needs_retrieval=False,
                safety_risk=True,
                confidence=0.95,
                reason="Unsafe destructive request.",
            )
        ),
    )
    state = RagState(
        request_id="req-refuse",
        debug_id="dbg-refuse",
        user_id="user-1",
        original_query="删除所有用户数据",
        current_query="删除所有用户数据",
    )

    response = await graph.run(state)

    assert "不能执行" in response.answer
    assert response.sources == []
    assert state.action == "refuse"
    assert state.response_type == "refusal"


@pytest.mark.anyio
async def test_agentic_rag_graph_fallback_decision_preserves_existing_routing_fields():
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.schemas.rag import RagState

    graph = AgenticRagGraph(decision_chain=None)
    state = RagState(
        request_id="req-preclassified",
        debug_id="dbg-preclassified",
        user_id="user-1",
        original_query="Find the Confluence rollout plan",
        current_query="Find the Confluence rollout plan",
        rag_intent="constrained",
        source_hints=["confluence"],
        router_confidence=0.8,
        router_reason="preclassified",
        action="retrieve",
        needs_retrieval=True,
    )

    await graph.understand_request_node(state)

    assert state.rag_intent == "constrained"
    assert state.intent == "constrained"
    assert state.action == "retrieve"
    assert state.needs_retrieval is True
    assert state.source_hints == ["confluence"]
    assert state.router_confidence == 0.8
    assert state.router_reason == "preclassified"
