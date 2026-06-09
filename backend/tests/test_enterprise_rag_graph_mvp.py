import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEnterpriseRagService:
    def __init__(self, documents):
        self.documents = documents
        self.retrieve_calls = []
        self.generated_queries = []
        self.generated_memory_contexts = []

    async def retrieve_with_details(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return {
            "dense_results": self.documents,
            "bm25_results": [],
            "fused_results": self.documents,
            "reranked_results": self.documents if kwargs.get("use_reranker") else [],
            "selected_documents": self.documents[: kwargs["final_top_k"]],
            "metrics": {"dense_ms": 1.0, "bm25_ms": 1.0, "rrf_ms": 1.0, "rerank_ms": 1.0 if kwargs.get("use_reranker") else 0.0},
        }

    async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
        self.generated_queries.append(query)
        self.generated_memory_contexts.append(memory_context)
        return f"answer for {query} using {len(documents)} docs"


class CapturingTraceStore:
    def __init__(self):
        self.saved = []

    async def save(self, trace):
        self.saved.append(trace)


class FailingTraceStore:
    async def save(self, trace):
        raise OSError("disk full")


class QueryAwareEnterpriseRagService:
    def __init__(self, documents_by_query):
        self.documents_by_query = documents_by_query
        self.retrieve_calls = []
        self.generated_queries = []
        self.generated_memory_contexts = []

    async def retrieve_with_details(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        documents = self.documents_by_query.get(kwargs["query"], [])
        return {
            "dense_results": documents,
            "bm25_results": [],
            "fused_results": documents,
            "reranked_results": documents if kwargs.get("use_reranker") else [],
            "selected_documents": documents[: kwargs["final_top_k"]],
            "metrics": {"dense_ms": 1.0, "bm25_ms": 1.0, "rrf_ms": 1.0, "rerank_ms": 1.0 if kwargs.get("use_reranker") else 0.0},
        }

    async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
        self.generated_queries.append(query)
        self.generated_memory_contexts.append(memory_context)
        return f"answer for {query} using {len(documents)} docs"


def test_enterprise_rag_graph_supports_agentic_intent_taxonomy():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph

    assert EnterpriseRagGraph._task_type_for_intent("fact_lookup", "Where is policy?") == "fact_lookup"
    assert EnterpriseRagGraph._task_type_for_intent("semantic_query", "Explain policy") == "semantic_query"
    assert EnterpriseRagGraph._task_type_for_intent("multi_hop", "Which policies apply?") == "multi_hop"
    assert EnterpriseRagGraph._task_type_for_intent("comparison", "Compare policies") == "comparison"
    assert EnterpriseRagGraph._task_type_for_intent("procedure", "How do I request leave?") == "procedure"
    assert EnterpriseRagGraph._task_type_for_intent("constrained", "Find confluence policy") == "constrained"
    assert EnterpriseRagGraph._task_type_for_intent("conflicting_info", "Compare policies") == "comparison"
    assert EnterpriseRagGraph._task_type_for_intent("project_related", "Which policies apply?") == "multi_hop"


@pytest.mark.anyio
async def test_enterprise_rag_graph_decomposes_comparison_queries(monkeypatch):
    from app.rag import enterprise_rag_graph
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.rag.decomposition import SubQuery, SubQueryPlan
    from app.schemas.rag import RagState

    async def fake_decompose_query(query, history_context=""):
        return SubQueryPlan(
            original_query=query,
            sub_queries=[
                SubQuery(id="sq1", query="trial leave process", purpose="fact"),
                SubQuery(id="sq2", query="full-time leave process", purpose="comparison_dimension"),
            ],
        )

    monkeypatch.setattr(enterprise_rag_graph, "decompose_query", fake_decompose_query)
    service = QueryAwareEnterpriseRagService(
        {
            "trial leave process": [
                {
                    "parent_doc_id": "parent-1",
                    "parent_chunk_id": "chunk-1",
                    "source_type": "policy",
                    "title": "Trial policy",
                    "section_heading": "Leave",
                    "score": 0.7,
                    "parent_text": "Trial leave process.",
                }
            ],
            "full-time leave process": [
                {
                    "parent_doc_id": "parent-2",
                    "parent_chunk_id": "chunk-2",
                    "source_type": "policy",
                    "title": "Full-time policy",
                    "section_heading": "Leave",
                    "score": 0.8,
                    "parent_text": "Full-time leave process.",
                }
            ],
        }
    )
    trace_store = CapturingTraceStore()
    graph = EnterpriseRagGraph(service=service, trace_store=trace_store)
    state = RagState(
        request_id="req-decompose",
        debug_id="dbg-decompose",
        session_id="sess-decompose",
        user_id="user-1",
        original_query="Compare trial and full-time leave process",
        current_query="Compare trial and full-time leave process",
        rag_intent="comparison",
        router_confidence=0.9,
    )

    response = await graph.run(state)

    assert response.debug_id == "dbg-decompose"
    assert response.strategy.strategy_name == "dense_bm25_rrf_reranker_decompose"
    assert [call["query"] for call in service.retrieve_calls] == [
        "trial leave process",
        "full-time leave process",
    ]
    assert [item.sub_query_id for item in state.sub_queries] == ["sq1", "sq2"]
    assert response.metrics.retrieval_attempts == 2
    assert {source.title for source in response.sources} == {"Trial policy", "Full-time policy"}
    assert trace_store.saved[0].strategy.strategy.use_decompose is True
    assert [attempt.attempt.sub_query_id for attempt in trace_store.saved[0].retrieval_attempts] == ["sq1", "sq2"]
    assert any(event["event"] == "query_decomposed" for event in state.sse_events)
    assert all(event["request_id"] == "req-decompose" for event in state.sse_events)


@pytest.mark.anyio
async def test_enterprise_rag_graph_clears_stale_sub_queries_when_decomposition_falls_back(monkeypatch):
    from app.rag import enterprise_rag_graph
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.rag.decomposition import build_fallback_plan
    from app.schemas.rag import RagState, SubQuery as RagSubQuery

    async def fake_decompose_query(query, history_context=""):
        return build_fallback_plan(query, "invalid_json")

    monkeypatch.setattr(enterprise_rag_graph, "decompose_query", fake_decompose_query)
    service = QueryAwareEnterpriseRagService(
        {
            "Compare trial and full-time leave process": [
                {
                    "parent_doc_id": "parent-1",
                    "parent_chunk_id": "chunk-1",
                    "source_type": "policy",
                    "title": "Leave policy",
                    "section_heading": "Leave",
                    "score": 0.9,
                    "parent_text": "Trial and full-time leave process comparison.",
                }
            ]
        }
    )
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore())
    state = RagState(
        request_id="req-fallback-decompose",
        debug_id="dbg-fallback-decompose",
        user_id="user-1",
        original_query="Compare trial and full-time leave process",
        current_query="Compare trial and full-time leave process",
        rag_intent="comparison",
        router_confidence=0.9,
        sub_queries=[RagSubQuery(sub_query_id="stale", query="stale aspect")],
    )

    response = await graph.run(state)

    assert state.sub_queries == []
    assert response.evaluation.enough_evidence is True
    assert response.sources[0].title == "Leave policy"
    assert service.retrieve_calls[0]["query"] == "Compare trial and full-time leave process"


@pytest.mark.anyio
async def test_enterprise_rag_graph_requires_decomposed_query_coverage(monkeypatch):
    from app.rag import enterprise_rag_graph
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.rag.decomposition import SubQuery, SubQueryPlan
    from app.schemas.rag import RagState

    async def fake_decompose_query(query, history_context=""):
        return SubQueryPlan(
            original_query=query,
            sub_queries=[
                SubQuery(id="sq1", query="trial leave process", purpose="fact"),
                SubQuery(id="sq2", query="full-time leave process", purpose="comparison_dimension"),
            ],
        )

    monkeypatch.setattr(enterprise_rag_graph, "decompose_query", fake_decompose_query)
    service = QueryAwareEnterpriseRagService(
        {
            "trial leave process": [
                {
                    "parent_doc_id": "parent-1",
                    "parent_chunk_id": "chunk-1",
                    "source_type": "policy",
                    "title": "Trial policy",
                    "section_heading": "Leave",
                    "score": 0.7,
                    "parent_text": "Trial leave process.",
                }
            ],
            "full-time leave process": [],
        }
    )
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore())
    state = RagState(
        request_id="req-partial-decompose",
        debug_id="dbg-partial-decompose",
        user_id="user-1",
        original_query="Compare trial and full-time leave process",
        current_query="Compare trial and full-time leave process",
        rag_intent="comparison",
        router_confidence=0.9,
        max_retries=0,
    )

    response = await graph.run(state)

    assert response.evaluation.enough_evidence is False
    assert response.evaluation.covered_aspects == ["trial leave process"]
    assert response.evaluation.missing_aspects == ["full-time leave process"]
    assert service.generated_queries == []
    assert "full-time leave process" in response.answer


@pytest.mark.anyio
async def test_enterprise_rag_graph_generates_answer_and_saves_trace_for_strong_evidence():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import MemoryItem, RagMemoryContext, RagState

    service = FakeEnterpriseRagService(
        documents=[
            {
                "parent_doc_id": "parent-1",
                "parent_chunk_id": "chunk-1",
                "source_type": "confluence",
                "title": "PTO Policy",
                "section_heading": "Annual leave",
                "score": 0.9,
                "parent_text": "Employees can find PTO rules in the HR page.",
                "child_text": "PTO rules",
                "metadata": {"source_type": "confluence"},
            }
        ]
    )
    trace_store = CapturingTraceStore()
    graph = EnterpriseRagGraph(service=service, trace_store=trace_store)
    state = RagState(
        request_id="req-1",
        debug_id="dbg-1",
        session_id="sess-1",
        user_id="user-1",
        original_query="Where is the PTO policy?",
        current_query="Where is the PTO policy?",
        rag_intent="constrained",
        source_hints=["confluence"],
        router_confidence=0.91,
        router_reason="Needs internal policy docs.",
        memory_context=RagMemoryContext(
            recalled=[
                MemoryItem(
                    memory_id="memory-1",
                    content="User is asking about the HR onboarding project.",
                    category="project_context",
                    relevance_score=0.9,
                    source="long_term",
                )
            ]
        ),
    )

    response = await graph.run(state)

    assert response.debug_id == "dbg-1"
    assert response.answer == "answer for Where is the PTO policy? using 1 docs"
    assert response.sources[0].title == "PTO Policy"
    assert response.evaluation.enough_evidence is True
    assert response.metrics.retrieval_attempts == 1
    assert service.retrieve_calls[0]["query"] == "Where is the PTO policy?"
    assert service.retrieve_calls[0]["source_hints"] == ["confluence"]
    assert service.retrieve_calls[0]["use_reranker"] is True
    assert service.generated_memory_contexts[0] == state.memory_context
    assert trace_store.saved[0].planner.plan.task_type == "constrained"
    assert trace_store.saved[0].strategy.strategy.strategy_name == "dense_bm25_rrf_reranker"
    assert trace_store.saved[0].retrieval_attempts[0].attempt.selected_documents[0].title == "PTO Policy"
    assert trace_store.saved[0].evaluations[0].result.enough_evidence is True
    assert trace_store.saved[0].generation.answer_preview.startswith("answer for")


@pytest.mark.anyio
async def test_enterprise_rag_graph_returns_insufficient_evidence_without_generation_when_no_documents():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState

    service = FakeEnterpriseRagService(documents=[])
    trace_store = CapturingTraceStore()
    graph = EnterpriseRagGraph(service=service, trace_store=trace_store)
    state = RagState(
        request_id="req-2",
        debug_id="dbg-2",
        session_id="sess-2",
        user_id="user-1",
        original_query="What is the launch code name?",
        current_query="What is the launch code name?",
    )

    response = await graph.run(state)

    assert "没有找到足够信息" in response.answer
    assert response.sources == []
    assert response.evaluation.enough_evidence is False
    assert response.evaluation.missing_aspects == ["What is the launch code name?"]
    assert service.generated_queries == []
    assert trace_store.saved[0].final_answer_preview == response.answer



@pytest.mark.anyio
async def test_enterprise_rag_graph_run_delegates_to_compiled_langgraph_workflow():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import EvaluationSummary, RagMetrics, RagResponse, RagState, RagStrategySummary

    class FakeCompiledGraph:
        def __init__(self):
            self.received_state = None

        async def ainvoke(self, graph_state):
            self.received_state = graph_state
            return {
                "response": RagResponse(
                    request_id=graph_state["rag_state"].request_id,
                    debug_id=graph_state["rag_state"].debug_id,
                    session_id=graph_state["rag_state"].session_id,
                    answer="graph answer",
                    sources=[],
                    strategy=RagStrategySummary(strategy_name="default", retrieval_mode="hybrid", final_top_k=5),
                    evaluation=EvaluationSummary(enough_evidence=True),
                    metrics=RagMetrics(),
                )
            }

    compiled_graph = FakeCompiledGraph()
    graph = object.__new__(EnterpriseRagGraph)
    graph.graph = compiled_graph
    state = RagState(
        request_id="req-graph-delegate",
        debug_id="dbg-graph-delegate",
        user_id="user-1",
        original_query="Where is PTO?",
        current_query="Where is PTO?",
    )

    response = await graph.run(state)

    assert response.answer == "graph answer"
    assert compiled_graph.received_state["rag_state"] is state


def test_enterprise_rag_service_formats_dict_documents_for_answer_context():
    from app.rag.enterprise_rag_service import EnterpriseRagService

    context = EnterpriseRagService._format_context(
        [
            {
                "parent_doc_id": "parent-1",
                "source_type": "confluence",
                "title": "PTO Policy",
                "section_heading": "Annual leave",
                "parent_text": "Employees can find PTO rules in the HR page.",
            }
        ]
    )

    assert "source_type: confluence" in context
    assert "title: PTO Policy" in context
    assert "parent_doc_id: parent-1" in context
    assert "Employees can find PTO rules" in context


def test_enterprise_rag_service_formats_dict_web_results_for_answer_context():
    from app.rag.enterprise_rag_service import EnterpriseRagService

    context = EnterpriseRagService._format_web_context(
        [{"title": "T", "url": "U", "snippet": "S"}]
    )

    assert "T" in context
    assert "U" in context
    assert "S" in context


@pytest.mark.anyio
async def test_enterprise_rag_graph_warns_but_still_responds_when_trace_store_fails():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState

    service = FakeEnterpriseRagService(documents=[])
    graph = EnterpriseRagGraph(service=service, trace_store=FailingTraceStore())
    state = RagState(
        request_id="req-3",
        debug_id="dbg-3",
        user_id="user-1",
        original_query="Unknown question?",
        current_query="Unknown question?",
    )

    response = await graph.run(state)

    assert response.debug_id == "dbg-3"
    assert any("trace" in warning.lower() for warning in response.warnings)


@pytest.mark.anyio
async def test_web_fallback_generation_receives_web_context():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState, WebSearchResult

    class CapturingService(QueryAwareEnterpriseRagService):
        def __init__(self):
            super().__init__({"报销流程": []})
            self.generated_web_results = None
            self.generated_evidence_mode = None

        async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
            self.generated_web_results = web_results
            self.generated_evidence_mode = evidence_mode
            return "公司知识库没有足够信息。通用参考：提交申请、主管审批、财务复核。"

    class FakeWebSearchService:
        async def search(self, query, max_results=3):
            return [WebSearchResult(title="报销流程参考", url="https://example.test/expense", snippet="提交申请、主管审批、财务复核。", score=0.8)]

    service = CapturingService()
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore(), web_search_service=FakeWebSearchService())
    state = RagState(
        request_id="req-web-generation",
        debug_id="dbg-web-generation",
        user_id="user-1",
        original_query="报销流程",
        current_query="报销流程",
        rag_intent="procedure",
        max_retries=0,
    )

    response = await graph.run(state)

    assert service.generated_web_results[0].title == "报销流程参考"
    assert service.generated_evidence_mode == "web_fallback"
    assert "通用参考" in response.answer
    assert response.sources[0].source_type == "web_reference"


@pytest.mark.anyio
async def test_public_context_query_with_internal_evidence_runs_hybrid_web_search():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState, WebSearchResult

    document = {
        "parent_doc_id": "parent-1",
        "parent_chunk_id": "chunk-1",
        "source_type": "policy",
        "title": "报销制度",
        "section_heading": "报销",
        "score": 0.9,
        "parent_text": "公司知识库说明：员工按报销制度提交票据并等待审批。",
    }

    class CapturingService(QueryAwareEnterpriseRagService):
        def __init__(self):
            super().__init__({"对比我们报销政策和行业最佳实践": [document]})
            self.generated_web_results = None
            self.generated_evidence_mode = None

        async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
            self.generated_web_results = web_results
            self.generated_evidence_mode = evidence_mode
            return "公司政策要求提交票据；公开资料建议审批链路清晰。"

    class FakeWebSearchService:
        def __init__(self):
            self.calls = []

        async def search(self, query, max_results=3):
            self.calls.append({"query": query, "max_results": max_results})
            return [
                WebSearchResult(
                    title="行业报销最佳实践",
                    url="https://example.test/best-practice",
                    snippet="公开资料建议保留票据、主管审批、财务复核。",
                    score=0.8,
                )
            ]

    service = CapturingService()
    web_service = FakeWebSearchService()
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore(), web_search_service=web_service)
    state = RagState(
        request_id="req-hybrid-public-context",
        debug_id="dbg-hybrid-public-context",
        user_id="user-1",
        original_query="对比我们报销政策和行业最佳实践",
        current_query="对比我们报销政策和行业最佳实践",
        rag_intent="comparison",
        max_retries=0,
    )

    response = await graph.run(state)

    assert web_service.calls == [{"query": "对比我们报销政策和行业最佳实践", "max_results": 3}]
    assert service.generated_evidence_mode == "hybrid"
    assert service.generated_web_results[0].title == "行业报销最佳实践"
    assert {source.source_type for source in response.sources} == {"policy", "web_reference"}


@pytest.mark.anyio
async def test_injected_web_search_dict_results_are_normalized_before_source_conversion():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState

    class CapturingService(QueryAwareEnterpriseRagService):
        def __init__(self):
            super().__init__({"如果公司知识库没有报销流程，给我一个通用流程参考": []})

        async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
            return "公司知识库没有足够信息。以下为通用参考。"

    class DictWebSearchService:
        async def search(self, query, max_results=3):
            return [
                {
                    "title": "通用报销流程参考",
                    "url": "https://example.test/expense-dict",
                    "snippet": "提交申请、主管审批、财务复核。",
                    "score": 0.8,
                }
            ]

    graph = EnterpriseRagGraph(
        service=CapturingService(),
        trace_store=CapturingTraceStore(),
        web_search_service=DictWebSearchService(),
    )
    state = RagState(
        request_id="req-dict-web-results",
        debug_id="dbg-dict-web-results",
        user_id="user-1",
        original_query="如果公司知识库没有报销流程，给我一个通用流程参考",
        current_query="如果公司知识库没有报销流程，给我一个通用流程参考",
        rag_intent="procedure",
        max_retries=0,
    )

    response = await graph.run(state)

    assert response.sources[0].source_id == "web:https://example.test/expense-dict"
    assert response.sources[0].source_type == "web_reference"
    assert state.web_results[0].url == "https://example.test/expense-dict"


@pytest.mark.anyio
async def test_internal_evidence_generation_does_not_receive_web_context():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState

    document = {
        "parent_doc_id": "parent-1",
        "parent_chunk_id": "chunk-1",
        "source_type": "policy",
        "title": "报销制度",
        "section_heading": "报销",
        "score": 0.9,
        "parent_text": "根据公司知识库，按照报销制度执行。",
    }

    class CapturingService(QueryAwareEnterpriseRagService):
        def __init__(self):
            super().__init__({"报销流程": [document]})
            self.generated_web_results = "not-called"
            self.generated_evidence_mode = "not-called"

        async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
            self.generated_web_results = web_results
            self.generated_evidence_mode = evidence_mode
            return "根据公司知识库，按照报销制度执行。"

    service = CapturingService()
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore())
    state = RagState(
        request_id="req-internal-generation",
        debug_id="dbg-internal-generation",
        user_id="user-1",
        original_query="报销流程",
        current_query="报销流程",
        rag_intent="procedure",
    )

    response = await graph.run(state)

    assert service.generated_web_results == []
    assert service.generated_evidence_mode == "internal_only"
    assert response.sources[0].title == "报销制度"
