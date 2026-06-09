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

    async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
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


def test_external_search_policy_allows_generic_procedure_fallback():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import EvaluationResult, RagState

    graph = EnterpriseRagGraph(service=SequenceEnterpriseRagService([[]]), trace_store=CapturingTraceStore())
    state = RagState(
        request_id="req-web-policy",
        debug_id="dbg-web-policy",
        user_id="user-1",
        original_query="如果公司知识库没有报销流程，给我一个通用流程参考",
        current_query="如果公司知识库没有报销流程，给我一个通用流程参考",
        rag_intent="procedure",
        max_retries=0,
        evaluator_result=EvaluationResult(
            enough_evidence=False,
            missing_aspects=["如果公司知识库没有报销流程，给我一个通用流程参考"],
            suggested_action="insufficient_evidence",
        ),
    )

    graph.decide_external_search(state)

    assert state.external_search_decision.allowed is True
    assert state.external_search_decision.mode == "fallback"
    assert state.evidence_mode == "web_fallback"


def test_external_search_policy_blocks_company_specific_facts():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import EvaluationResult, RagState

    graph = EnterpriseRagGraph(service=SequenceEnterpriseRagService([[]]), trace_store=CapturingTraceStore())
    state = RagState(
        request_id="req-web-block",
        debug_id="dbg-web-block",
        user_id="user-1",
        original_query="我们公司今年报销上限是多少？",
        current_query="我们公司今年报销上限是多少？",
        rag_intent="fact_lookup",
        max_retries=0,
        evaluator_result=EvaluationResult(
            enough_evidence=False,
            missing_aspects=["我们公司今年报销上限是多少？"],
            suggested_action="insufficient_evidence",
        ),
    )

    graph.decide_external_search(state)

    assert state.external_search_decision.allowed is False
    assert state.external_search_decision.mode == "none"
    assert state.evidence_mode == "internal_only"


def test_external_search_policy_blocks_english_salary_fact_with_generic_reference_request():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import EvaluationResult, RagState

    graph = EnterpriseRagGraph(service=SequenceEnterpriseRagService([[]]), trace_store=CapturingTraceStore())
    query = "If no internal docs exist for how much salary our company pays, give a general reference"
    state = RagState(
        request_id="req-web-block-english-salary-fallback",
        debug_id="dbg-web-block-english-salary-fallback",
        user_id="user-1",
        original_query=query,
        current_query=query,
        rag_intent="fact_lookup",
        max_retries=0,
        evaluator_result=EvaluationResult(
            enough_evidence=False,
            missing_aspects=[query],
            suggested_action="insufficient_evidence",
        ),
    )

    graph.decide_external_search(state)

    assert state.external_search_decision.allowed is False
    assert state.external_search_decision.mode == "none"
    assert state.evidence_mode == "internal_only"
    assert "company-specific" in state.external_search_decision.reason


def test_internal_api_endpoint_question_is_not_blocked_as_private_company_fact():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import EvaluationResult, RagState

    graph = EnterpriseRagGraph(service=SequenceEnterpriseRagService([[]]), trace_store=CapturingTraceStore())
    query = "What is an internal API endpoint?"
    state = RagState(
        request_id="req-generic-internal-api-endpoint",
        debug_id="dbg-generic-internal-api-endpoint",
        user_id="user-1",
        original_query=query,
        current_query=query,
        rag_intent="semantic_query",
        max_retries=0,
        evaluator_result=EvaluationResult(
            enough_evidence=False,
            missing_aspects=[query],
            suggested_action="insufficient_evidence",
        ),
    )

    graph.decide_external_search(state)

    assert state.external_search_decision.reason != (
        "The question asks for company-specific information that public web results cannot replace."
    )
    assert not (
        state.external_search_decision.mode == "none"
        and state.evidence_mode == "internal_only"
        and "company-specific" in state.external_search_decision.reason
    )


class FakeWebSearchService:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def search(self, query, max_results=3):
        self.calls.append({"query": query, "max_results": max_results})
        return self.results


@pytest.mark.anyio
async def test_yes_no_company_kb_procedure_reference_question_does_not_run_web_fallback():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState, WebSearchResult

    service = SequenceEnterpriseRagService([[]])
    web_service = FakeWebSearchService([
        WebSearchResult(
            title="通用报销流程参考",
            url="https://example.test/expense",
            snippet="提交申请、主管审批、财务复核。",
            score=0.8,
        )
    ])
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore(), web_search_service=web_service)
    state = RagState(
        request_id="req-web-block-kb-yes-no-procedure-reference",
        debug_id="dbg-web-block-kb-yes-no-procedure-reference",
        user_id="user-1",
        original_query="公司知识库有没有报销流程参考？",
        current_query="公司知识库有没有报销流程参考？",
        rag_intent="procedure",
        max_retries=0,
    )

    response = await graph.run(state)

    assert web_service.calls == []
    assert state.evidence_mode == "internal_only"
    assert response.sources == []
    assert "没有找到足够信息" in response.answer


@pytest.mark.anyio
async def test_web_fallback_runs_after_internal_evidence_is_exhausted():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState, WebSearchResult

    service = SequenceEnterpriseRagService([[]])
    web_service = FakeWebSearchService([
        WebSearchResult(
            title="通用报销流程参考",
            url="https://example.test/expense",
            snippet="提交申请、主管审批、财务复核。",
            score=0.8,
        )
    ])
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore(), web_search_service=web_service)
    state = RagState(
        request_id="req-web-flow",
        debug_id="dbg-web-flow",
        user_id="user-1",
        original_query="如果公司知识库没有报销流程，给我一个通用流程参考",
        current_query="如果公司知识库没有报销流程，给我一个通用流程参考",
        rag_intent="procedure",
        max_retries=0,
    )

    response = await graph.run(state)

    assert web_service.calls == [{"query": "如果公司知识库没有报销流程，给我一个通用流程参考", "max_results": 3}]
    assert state.web_search_attempted is True
    assert state.web_results[0].title == "通用报销流程参考"
    assert state.evidence_mode == "web_fallback"
    assert response.sources[0].source_type == "web_reference"
    assert any(event["event"] == "web_search_finished" for event in state.sse_events)


@pytest.mark.anyio
async def test_company_specific_fact_does_not_run_web_fallback():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState, WebSearchResult

    service = SequenceEnterpriseRagService([[]])
    web_service = FakeWebSearchService([
        WebSearchResult(title="Generic answer", url="https://example.test", snippet="Generic", score=0.8)
    ])
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore(), web_search_service=web_service)
    state = RagState(
        request_id="req-web-block-flow",
        debug_id="dbg-web-block-flow",
        user_id="user-1",
        original_query="我们公司今年报销上限是多少？",
        current_query="我们公司今年报销上限是多少？",
        rag_intent="fact_lookup",
        max_retries=0,
    )

    response = await graph.run(state)

    assert web_service.calls == []
    assert state.web_results == []
    assert state.evidence_mode == "internal_only"
    assert response.sources == []
    assert "没有找到足够信息" in response.answer


@pytest.mark.anyio
async def test_english_company_specific_fact_does_not_run_web_fallback():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState, WebSearchResult

    service = SequenceEnterpriseRagService([[]])
    web_service = FakeWebSearchService([
        WebSearchResult(title="Generic reimbursement limits", url="https://example.test", snippet="Generic", score=0.8)
    ])
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore(), web_search_service=web_service)
    state = RagState(
        request_id="req-web-block-english-company-fact",
        debug_id="dbg-web-block-english-company-fact",
        user_id="user-1",
        original_query="What is our company reimbursement limit?",
        current_query="What is our company reimbursement limit?",
        rag_intent="fact_lookup",
        max_retries=0,
    )

    response = await graph.run(state)

    assert web_service.calls == []
    assert state.web_results == []
    assert state.evidence_mode == "internal_only"
    assert response.sources == []
    assert "没有找到足够信息" in response.answer


@pytest.mark.anyio
async def test_internal_project_private_fact_does_not_run_web_fallback():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState, WebSearchResult

    service = SequenceEnterpriseRagService([[]])
    web_service = FakeWebSearchService([
        WebSearchResult(title="Public deployment docs", url="https://example.test", snippet="Generic", score=0.8)
    ])
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore(), web_search_service=web_service)
    state = RagState(
        request_id="req-web-block-internal-project-fact",
        debug_id="dbg-web-block-internal-project-fact",
        user_id="user-1",
        original_query="NexusKB deployment endpoint是什么？",
        current_query="NexusKB deployment endpoint是什么？",
        rag_intent="semantic_query",
        max_retries=0,
    )

    response = await graph.run(state)

    assert web_service.calls == []
    assert state.web_results == []
    assert state.evidence_mode == "internal_only"
    assert response.sources == []
    assert "没有找到足够信息" in response.answer


@pytest.mark.anyio
async def test_company_specific_procedure_does_not_run_web_fallback_without_generic_request():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState, WebSearchResult

    service = SequenceEnterpriseRagService([[]])
    web_service = FakeWebSearchService([
        WebSearchResult(
            title="Generic PTO process",
            url="https://example.test/pto",
            snippet="Ask your manager and HR.",
            score=0.8,
        )
    ])
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore(), web_search_service=web_service)
    state = RagState(
        request_id="req-web-block-company-procedure",
        debug_id="dbg-web-block-company-procedure",
        user_id="user-1",
        original_query="How do I request PTO at our company?",
        current_query="How do I request PTO at our company?",
        rag_intent="procedure",
        max_retries=0,
    )

    response = await graph.run(state)

    assert web_service.calls == []
    assert state.evidence_mode == "internal_only"
    assert response.sources == []
    assert "没有找到足够信息" in response.answer


@pytest.mark.anyio
async def test_company_specific_procedure_allows_explicit_generic_reference_request():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState, WebSearchResult

    service = SequenceEnterpriseRagService([[]])
    web_service = FakeWebSearchService([
        WebSearchResult(
            title="通用休假申请流程",
            url="https://example.test/pto-generic",
            snippet="提交申请，主管审批，HR 备案。",
            score=0.8,
        )
    ])
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore(), web_search_service=web_service)
    state = RagState(
        request_id="req-web-allow-generic-procedure",
        debug_id="dbg-web-allow-generic-procedure",
        user_id="user-1",
        original_query="如果公司知识库没有休假流程，给我一个通用参考",
        current_query="如果公司知识库没有休假流程，给我一个通用参考",
        rag_intent="procedure",
        max_retries=0,
    )

    response = await graph.run(state)

    assert web_service.calls == [{"query": "如果公司知识库没有休假流程，给我一个通用参考", "max_results": 3}]
    assert state.evidence_mode == "web_fallback"
    assert response.sources[0].source_type == "web_reference"


@pytest.mark.anyio
async def test_company_specific_fact_with_generic_reference_request_still_blocks_web_fallback():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import RagState, WebSearchResult

    service = SequenceEnterpriseRagService([[]])
    web_service = FakeWebSearchService([
        WebSearchResult(
            title="通用报销上限参考",
            url="https://example.test/reimbursement-limit",
            snippet="公开资料中的一般报销参考。",
            score=0.8,
        )
    ])
    graph = EnterpriseRagGraph(service=service, trace_store=CapturingTraceStore(), web_search_service=web_service)
    state = RagState(
        request_id="req-web-block-company-fact-generic-reference",
        debug_id="dbg-web-block-company-fact-generic-reference",
        user_id="user-1",
        original_query="如果知识库没有我们公司今年报销上限是多少，给我一个通用参考",
        current_query="如果知识库没有我们公司今年报销上限是多少，给我一个通用参考",
        rag_intent="fact_lookup",
        max_retries=0,
    )

    response = await graph.run(state)

    assert web_service.calls == []
    assert state.evidence_mode == "internal_only"
    assert response.sources == []


def test_generic_api_endpoint_question_is_not_blocked_as_private_company_fact():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import EvaluationResult, RagState

    graph = EnterpriseRagGraph(service=SequenceEnterpriseRagService([[]]), trace_store=CapturingTraceStore())
    state = RagState(
        request_id="req-generic-api-endpoint",
        debug_id="dbg-generic-api-endpoint",
        user_id="user-1",
        original_query="What is an API endpoint?",
        current_query="What is an API endpoint?",
        rag_intent="semantic_query",
        max_retries=0,
        evaluator_result=EvaluationResult(
            enough_evidence=False,
            missing_aspects=["What is an API endpoint?"],
            suggested_action="insufficient_evidence",
        ),
    )

    graph.decide_external_search(state)

    assert state.external_search_decision.reason != (
        "The question asks for company-specific information that public web results cannot replace."
    )
    assert not (
        state.external_search_decision.mode == "none"
        and state.evidence_mode == "internal_only"
        and "company-specific" in state.external_search_decision.reason
    )


def test_generic_english_company_best_practices_are_not_private_company_facts():
    from app.rag.enterprise_rag_graph import EnterpriseRagGraph
    from app.schemas.rag import EvaluationResult, RagState

    graph = EnterpriseRagGraph(service=SequenceEnterpriseRagService([[]]), trace_store=CapturingTraceStore())
    state = RagState(
        request_id="req-generic-company-best-practices",
        debug_id="dbg-generic-company-best-practices",
        user_id="user-1",
        original_query="Which company expense reimbursement best practices are common?",
        current_query="Which company expense reimbursement best practices are common?",
        rag_intent="semantic_query",
        max_retries=0,
        evaluator_result=EvaluationResult(
            enough_evidence=False,
            missing_aspects=["Which company expense reimbursement best practices are common?"],
            suggested_action="insufficient_evidence",
        ),
    )

    graph.decide_external_search(state)

    assert state.external_search_decision.allowed is True
    assert state.external_search_decision.mode == "fallback"
    assert state.evidence_mode == "web_fallback"
