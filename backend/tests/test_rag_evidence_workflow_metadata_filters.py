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


class EmptyThenUsefulPipeline:
    def __init__(self):
        self.filters = []

    async def run(self, **kwargs):
        from app.rag.retrieval_pipeline import RetrievalPipelineResult, RetrievalStageMetrics
        from app.schemas.rag import RetrievalAttempt, RagDocument

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
async def test_workflow_plans_hard_filter_then_falls_back_to_soft_when_empty():
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow
    from app.schemas.rag import RagState

    pipeline = EmptyThenUsefulPipeline()
    workflow = RagEvidenceWorkflow(strategy_router=StaticStrategyRouter(), retrieval_pipeline=pipeline, trace_store=None, web_search_service=None)
    state = RagState(
        request_id="req-filter-loop",
        debug_id="dbg-filter-loop",
        user_id="user-1",
        original_query="Find the Confluence PTO policy",
        current_query="Find the Confluence PTO policy",
        rag_intent="constrained",
        max_retries=1,
    )
    trace = workflow.initialize_trace(state, workflow._now())

    workflow.planner(state, trace)
    workflow.metadata_filter_plan(state, trace)
    workflow.strategy_select(state, trace)
    await workflow.retrieve(state, trace, reason="Initial metadata-filtered retrieval.")
    workflow.evaluate_context(state, trace)
    workflow.decide_next_action(state)
    workflow.broaden_metadata_filter(state)
    await workflow.retrieve(state, trace, reason="Retry after broadening metadata filter.")

    assert pipeline.filters == ["hard", "soft"]
    assert state.metadata_filter_fallback_count == 1
    assert state.metadata_filter_decision.mode == "soft"
    assert state.selected_documents
