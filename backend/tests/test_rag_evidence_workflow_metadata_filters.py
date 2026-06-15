import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class StaticStrategyRouter:
    def select(self, state):
        from app.schemas.rag import RagStrategyConfig

        return RagStrategyConfig(final_top_k=2, top_k_dense=5, top_k_bm25=5, fusion_top_k=5, max_retries=1)


class StubTraceStore:
    def __init__(self):
        self.saved = []

    async def save(self, trace):
        self.saved.append(trace)


class AnswerService:
    def __init__(self):
        self.generated = []

    async def generate_answer(self, query, documents, memory_context=None, web_results=None, evidence_mode="internal_only"):
        self.generated.append({"query": query, "documents": documents, "evidence_mode": evidence_mode})
        return f"generated answer for {query} with {len(documents)} docs"


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
            text="PTO policy text",
            metadata={"doc_semantic_type": "policy_rule"},
        )
        attempt.selected_documents = [document]
        return RetrievalPipelineResult(selected_documents=[document], attempt=attempt, metrics=RetrievalStageMetrics(total_ms=1.0), raw={})


@pytest.mark.anyio
async def test_workflow_run_broadens_hard_metadata_filter_to_soft_before_generating_answer():
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow
    from app.schemas.rag import RagState

    pipeline = EmptyThenUsefulPipeline()
    service = AnswerService()
    workflow = RagEvidenceWorkflow(
        service=service,
        strategy_router=StaticStrategyRouter(),
        retrieval_pipeline=pipeline,
        trace_store=StubTraceStore(),
        web_search_service=None,
    )
    state = RagState(
        request_id="req-filter-loop",
        debug_id="dbg-filter-loop",
        user_id="user-1",
        original_query="Find the Confluence PTO policy",
        current_query="Find the Confluence PTO policy",
        rag_intent="constrained",
        max_retries=0,
    )

    response = await workflow.run(state)

    assert pipeline.filters == ["hard", "soft"]
    assert [attempt.metadata_filter.mode for attempt in state.retrieval_attempts] == ["hard", "soft"]
    assert state.metadata_filter_fallback_count == 1
    assert state.next_action == "generate"
    assert response.answer == "generated answer for Find the Confluence PTO policy with 1 docs"


def test_decide_next_action_uses_metadata_fallback_budget_even_when_query_retries_are_disabled():
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow
    from app.schemas.rag import EvaluationResult, MetadataFilterDecision, RagState

    workflow = RagEvidenceWorkflow(
        service=AnswerService(),
        strategy_router=StaticStrategyRouter(),
        retrieval_pipeline=EmptyThenUsefulPipeline(),
        trace_store=StubTraceStore(),
        web_search_service=None,
    )
    state = RagState(
        request_id="req-filter-decision",
        debug_id="dbg-filter-decision",
        user_id="user-1",
        original_query="Find the Confluence PTO policy",
        current_query="Find the Confluence PTO policy",
        rag_intent="constrained",
        max_retries=0,
        metadata_filter_decision=MetadataFilterDecision(
            mode="hard",
            source_types=["confluence"],
            doc_semantic_types=["policy_rule"],
            confidence=0.9,
            reason="Explicit source and document type constraints.",
        ),
        evaluator_result=EvaluationResult(
            enough_evidence=False,
            suggested_action="rewrite_query",
            missing_aspects=["Find the Confluence PTO policy"],
        ),
    )

    workflow.decide_next_action(state)

    assert state.next_action == "broaden_metadata_filter"
    assert state.metadata_filter_fallback_count == 0
    assert state.retry_count == 0
