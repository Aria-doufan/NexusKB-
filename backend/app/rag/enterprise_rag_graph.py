from app.rag.rag_evidence_workflow import RagEvidenceWorkflow
from app.schemas.rag import RagResponse, RagState


class EnterpriseRagGraph:
    _task_type_for_intent = staticmethod(RagEvidenceWorkflow._task_type_for_intent)

    def __init__(self, service=None, trace_store=None, strategy_router=None, retrieval_pipeline=None, web_search_service=None):
        self.workflow = RagEvidenceWorkflow(
            service=service,
            trace_store=trace_store,
            strategy_router=strategy_router,
            retrieval_pipeline=retrieval_pipeline,
            web_search_service=web_search_service,
        )

    def __getattr__(self, name: str):
        return getattr(self.workflow, name)

    async def run(self, state: RagState) -> RagResponse:
        return await self.workflow.run(state)
