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


def test_rag_state_tracks_metadata_filter_decision_defaults():
    from app.schemas.rag import MetadataFilterDecision, RagState, RetrievalAttempt

    state = RagState(
        request_id="req-filter-defaults",
        debug_id="dbg-filter-defaults",
        user_id="user-1",
        original_query="Find the Confluence policy",
        current_query="Find the Confluence policy",
    )

    assert state.metadata_filter_decision.mode == "none"
    assert state.metadata_filter_decision.source_types == []
    assert state.metadata_filter_decision.doc_semantic_types == []
    assert state.metadata_filter_decision.confidence == 0.0
    assert state.metadata_filter_fallback_count == 0

    decision = MetadataFilterDecision(
        mode="hard",
        source_types=["confluence"],
        doc_semantic_types=["policy_rule"],
        confidence=0.91,
        reason="The query explicitly asks for a Confluence policy.",
    )
    assert decision.mode == "hard"
    assert decision.source_types == ["confluence"]
    assert decision.doc_semantic_types == ["policy_rule"]

    attempt = RetrievalAttempt(
        attempt_id=1,
        query="Find the Confluence policy",
        metadata_filter=decision,
    )
    assert attempt.metadata_filter.mode == "hard"


def test_agentic_rag_graph_is_not_enterprise_rag_graph_subclass():
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph

    assert not issubclass(AgenticRagGraph, EnterpriseRagGraph)


def test_agentic_rag_graph_accepts_rag_workflow_dependency():
    from app.rag.agentic_rag_graph import AgenticRagGraph

    class StubRagWorkflow:
        pass

    workflow = StubRagWorkflow()
    graph = AgenticRagGraph(rag_workflow=workflow)

    assert graph.rag_workflow is workflow


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


@pytest.mark.anyio
async def test_agentic_rag_graph_retrieves_inside_single_graph():
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.schemas.rag import AgenticActionDecision, RagState

    class RetrievalService:
        def __init__(self):
            self.retrieve_calls = []
            self.generated = []

        async def retrieve_with_details(self, **kwargs):
            self.retrieve_calls.append(kwargs)
            document = {
                "parent_doc_id": "parent-1",
                "parent_chunk_id": "chunk-1",
                "source_type": "confluence",
                "title": "PTO Policy",
                "section_heading": "Leave",
                "score": 0.9,
                "parent_text": "Employees can request PTO in the HR system.",
                "child_text": "PTO request process",
                "metadata": {"source_type": "confluence"},
            }
            return {
                "dense_results": [document],
                "bm25_results": [],
                "fused_results": [document],
                "reranked_results": [document],
                "selected_documents": [document],
                "metrics": {"dense_ms": 1.0, "bm25_ms": 1.0, "rrf_ms": 1.0, "rerank_ms": 1.0},
            }

        async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
            self.generated.append({"query": query, "documents": documents, "evidence_mode": evidence_mode})
            return f"generated answer for {query} with {len(documents)} docs"

    service = RetrievalService()
    graph = AgenticRagGraph(
        service=service,
        trace_store=StubTraceStore(),
        decision_chain=StaticDecisionChain(
            AgenticActionDecision(
                intent="constrained",
                action="retrieve",
                needs_retrieval=True,
                source_hints=["confluence"],
                confidence=0.88,
                reason="Needs internal policy evidence.",
            )
        ),
    )
    state = RagState(
        request_id="req-retrieve",
        debug_id="dbg-retrieve",
        user_id="user-1",
        original_query="Where is PTO policy?",
        current_query="Where is PTO policy?",
        max_retries=0,
    )

    response = await graph.run(state)

    assert response.answer == "generated answer for Where is PTO policy? with 1 docs"
    assert response.sources[0].title == "PTO Policy"
    assert service.retrieve_calls[0]["source_hints"] == ["confluence"]
    assert state.action == "retrieve"
    assert state.evaluator_result.enough_evidence is True


@pytest.mark.anyio
async def test_agentic_rag_graph_builds_insufficient_evidence_answer_when_retrieval_finds_no_documents():
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.schemas.rag import AgenticActionDecision, RagState

    class EmptyRetrievalService:
        async def retrieve_with_details(self, **kwargs):
            return {
                "dense_results": [],
                "bm25_results": [],
                "fused_results": [],
                "reranked_results": [],
                "selected_documents": [],
                "metrics": {},
            }

        async def generate_answer(self, *args, **kwargs):
            raise AssertionError("insufficient evidence must not call answer generation")

    graph = AgenticRagGraph(
        service=EmptyRetrievalService(),
        trace_store=StubTraceStore(),
        decision_chain=StaticDecisionChain(
            AgenticActionDecision(
                intent="fact_lookup",
                action="retrieve",
                needs_retrieval=True,
                confidence=0.82,
                reason="Needs internal evidence.",
            )
        ),
    )
    state = RagState(
        request_id="req-empty",
        debug_id="dbg-empty",
        user_id="user-1",
        original_query="Where is the travel policy?",
        current_query="Where is the travel policy?",
        max_retries=0,
    )

    response = await graph.run(state)

    assert response.answer == "抱歉，我没有找到足够信息来回答：Where is the travel policy?"
    assert response.sources == []
    assert state.evaluator_result.enough_evidence is False


@pytest.mark.anyio
async def test_agentic_rag_graph_plans_and_routes_metadata_filter_broadening_on_public_run_path():
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow
    from app.schemas.rag import AgenticActionDecision, RagState

    class StaticStrategyRouter:
        def select(self, state):
            from app.schemas.rag import RagStrategyConfig

            return RagStrategyConfig(final_top_k=2, top_k_dense=5, top_k_bm25=5, fusion_top_k=5)

    class EmptyThenUsefulPipeline:
        def __init__(self):
            self.filters = []

        async def run(self, **kwargs):
            from app.rag.retrieval_pipeline import RetrievalPipelineResult, RetrievalStageMetrics
            from app.schemas.rag import RagDocument, RetrievalAttempt

            metadata_filter = kwargs["metadata_filter"]
            self.filters.append(metadata_filter.mode)
            attempt = RetrievalAttempt(
                attempt_id=kwargs["attempt_id"],
                query=kwargs["query"],
                strategy_name="test",
                metadata_filter=metadata_filter,
                reason=kwargs["reason"],
            )
            if len(self.filters) == 1:
                return RetrievalPipelineResult(selected_documents=[], attempt=attempt, metrics=RetrievalStageMetrics(total_ms=1.0), raw={})
            document = RagDocument(
                source_id="p1",
                parent_doc_id="d1",
                parent_chunk_id="p1",
                source_type="confluence",
                title="PTO Policy",
                text="Employees can request PTO in the HR system.",
                metadata={"doc_semantic_type": "policy_rule"},
            )
            attempt.selected_documents = [document]
            return RetrievalPipelineResult(selected_documents=[document], attempt=attempt, metrics=RetrievalStageMetrics(total_ms=1.0), raw={})

    class AnswerService:
        async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
            return f"generated answer for {query} with {len(documents)} docs"

    pipeline = EmptyThenUsefulPipeline()
    workflow = RagEvidenceWorkflow(
        service=AnswerService(),
        trace_store=StubTraceStore(),
        strategy_router=StaticStrategyRouter(),
        retrieval_pipeline=pipeline,
    )
    graph = AgenticRagGraph(
        rag_workflow=workflow,
        decision_chain=StaticDecisionChain(
            AgenticActionDecision(
                intent="constrained",
                action="retrieve",
                needs_retrieval=True,
                source_hints=["confluence"],
                confidence=0.88,
                reason="Needs internal policy evidence.",
            )
        ),
    )
    state = RagState(
        request_id="req-filter-graph",
        debug_id="dbg-filter-graph",
        user_id="user-1",
        original_query="Find the Confluence PTO policy",
        current_query="Find the Confluence PTO policy",
        max_retries=0,
    )

    response = await graph.run(state)

    assert pipeline.filters == ["hard", "soft"]
    assert [attempt.metadata_filter.mode for attempt in state.retrieval_attempts] == ["hard", "soft"]
    assert state.metadata_filter_fallback_count == 1
    assert state.next_action == "generate"
    assert response.answer == "generated answer for Find the Confluence PTO policy with 1 docs"


@pytest.mark.anyio
async def test_agentic_rag_graph_records_retry_reason_on_followup_retrieval_attempt():
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.schemas.rag import AgenticActionDecision, RagState

    class RetryRetrievalService:
        def __init__(self):
            self.calls = 0

        async def retrieve_with_details(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "dense_results": [],
                    "bm25_results": [],
                    "fused_results": [],
                    "reranked_results": [],
                    "selected_documents": [],
                    "metrics": {},
                }
            document = {
                "parent_doc_id": "parent-1",
                "parent_chunk_id": "chunk-1",
                "source_type": "confluence",
                "title": "Travel Policy",
                "section_heading": "Booking",
                "score": 0.9,
                "parent_text": "Travel must be booked in the approved system.",
                "child_text": "Travel booking process",
                "metadata": {"source_type": "confluence"},
            }
            return {
                "dense_results": [document],
                "bm25_results": [],
                "fused_results": [document],
                "reranked_results": [document],
                "selected_documents": [document],
                "metrics": {},
            }

        async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
            return f"generated answer for {query} with {len(documents)} docs"

    graph = AgenticRagGraph(
        service=RetryRetrievalService(),
        trace_store=StubTraceStore(),
        decision_chain=StaticDecisionChain(
            AgenticActionDecision(
                intent="fact_lookup",
                action="retrieve",
                needs_retrieval=True,
                confidence=0.82,
                reason="Needs internal evidence.",
            )
        ),
    )
    state = RagState(
        request_id="req-retry",
        debug_id="dbg-retry",
        user_id="user-1",
        original_query="Where is the travel policy?",
        current_query="Where is the travel policy?",
        max_retries=1,
    )

    await graph.run(state)

    assert [attempt.reason for attempt in state.retrieval_attempts] == [
        "Initial hybrid retrieval.",
        "Retry after rewrite_query.",
    ]


def test_agentic_rag_tools_do_not_include_legacy_rag_tool():
    from app.agent.agent_tools import AGENTIC_RAG_TOOLS

    tool_names = {tool.name for tool in AGENTIC_RAG_TOOLS}

    assert "rag_summary_tools" not in tool_names
    assert "what_time_is_now" in tool_names
    assert "get_weather_tools" in tool_names


@pytest.mark.anyio
async def test_agentic_rag_graph_tool_call_uses_tool_node_not_retrieval():
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.schemas.rag import AgenticActionDecision, RagState

    class NoRetrieveService:
        async def retrieve(self, **kwargs):
            raise AssertionError("tool_call must not retrieve")

        async def generate_answer(self, *args, **kwargs):
            raise AssertionError("tool_call must not call RAG generation")

    class StubToolRunner:
        def __init__(self):
            self.calls = []

        async def run(self, query, required_tools):
            self.calls.append({"query": query, "required_tools": required_tools})
            return [
                {
                    "tool_name": "what_time_is_now",
                    "tool_input": {},
                    "output": "当前时间是：2026-06-10 10:30",
                    "success": True,
                }
            ]

    tool_runner = StubToolRunner()
    graph = AgenticRagGraph(
        service=NoRetrieveService(),
        trace_store=StubTraceStore(),
        decision_chain=StaticDecisionChain(
            AgenticActionDecision(
                intent="tool_use",
                action="tool_call",
                needs_retrieval=False,
                needs_tool=True,
                required_tools=["what_time_is_now"],
                confidence=0.96,
                reason="Current time needs safe utility tool.",
            )
        ),
        tool_runner=tool_runner,
    )
    state = RagState(
        request_id="req-tool",
        debug_id="dbg-tool",
        user_id="user-1",
        original_query="现在几点？",
        current_query="现在几点？",
    )

    response = await graph.run(state)

    assert response.answer == "当前时间是：2026-06-10 10:30"
    assert state.response_type == "tool_answer"
    assert state.tool_results[0].tool_name == "what_time_is_now"
    assert tool_runner.calls == [{"query": "现在几点？", "required_tools": ["what_time_is_now"]}]


@pytest.mark.anyio
async def test_agentic_rag_graph_invoke_loads_context_and_persists_messages(monkeypatch):
    from app.rag import agentic_rag_graph
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.schemas.rag import AgenticActionDecision

    persisted = []

    class MemoryContext:
        summary = "compressed memory"
        compressed_turns = 1
        total_turns = 2

        def to_agent_history(self):
            return [("hi", "hello")]

    class ConversationMemoryService:
        async def get_memory_context(self, session_id, user_id):
            return MemoryContext()

        async def append_interaction(self, session_id, user_id, user_message, assistant_message):
            persisted.append(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "user_message": user_message,
                    "assistant_message": assistant_message,
                }
            )

    class LongTermMemoryService:
        async def search(self, query, user_id):
            return []

    monkeypatch.setattr(agentic_rag_graph, "conversation_memory_service", ConversationMemoryService())
    monkeypatch.setattr(agentic_rag_graph, "long_term_memory_service", LongTermMemoryService())

    graph = AgenticRagGraph(
        trace_store=StubTraceStore(),
        decision_chain=StaticDecisionChain(
            AgenticActionDecision(
                intent="general_chat",
                action="direct_answer",
                needs_retrieval=False,
                confidence=0.9,
                reason="General.",
            )
        ),
    )

    result = await graph.invoke(query="What is LangGraph?", user_id="user-1", session_id="sess-1")

    assert result["session_id"] == "sess-1"
    assert result["response"] == "这是一个通用问题，可以不检索企业知识库直接回答：What is LangGraph?"
    assert result["action"] == "direct_answer"
    assert persisted == [
        {
            "session_id": "sess-1",
            "user_id": "user-1",
            "user_message": "What is LangGraph?",
            "assistant_message": "这是一个通用问题，可以不检索企业知识库直接回答：What is LangGraph?",
        }
    ]


def test_only_agentic_rag_graph_owns_compiled_langgraph_for_agent_path():
    from app.agent.router_graph import RouterGraph
    from app.rag.agentic_rag_graph import AgenticRagGraph
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow

    router = object.__new__(RouterGraph)
    agentic_graph = AgenticRagGraph()
    enterprise_wrapper = EnterpriseRagGraph()
    evidence_workflow = RagEvidenceWorkflow()

    assert hasattr(agentic_graph, "graph")
    assert not hasattr(router, "graph")
    assert not hasattr(enterprise_wrapper, "graph")
    assert not hasattr(evidence_workflow, "graph")
