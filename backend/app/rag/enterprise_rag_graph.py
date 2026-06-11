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

    @property
    def service(self):
        return self.workflow.service

    @property
    def strategy_router(self):
        return self.workflow.strategy_router

    @property
    def retrieval_pipeline(self):
        return self.workflow.retrieval_pipeline

    @property
    def trace_store(self):
        return self.workflow.trace_store

    @property
    def web_search_service(self):
        return self.workflow.web_search_service

    async def run(self, state: RagState) -> RagResponse:
        return await self.workflow.run(state)

    def initialize_trace(self, *args, **kwargs):
        return self.workflow.initialize_trace(*args, **kwargs)

    def planner(self, *args, **kwargs):
        return self.workflow.planner(*args, **kwargs)

    def strategy_select(self, *args, **kwargs):
        return self.workflow.strategy_select(*args, **kwargs)

    async def retrieve(self, *args, **kwargs):
        return await self.workflow.retrieve(*args, **kwargs)

    async def retrieve_decomposed(self, *args, **kwargs):
        return await self.workflow.retrieve_decomposed(*args, **kwargs)

    def evaluate_context(self, *args, **kwargs):
        return self.workflow.evaluate_context(*args, **kwargs)

    def decide_next_action(self, *args, **kwargs):
        return self.workflow.decide_next_action(*args, **kwargs)

    def rewrite_query(self, *args, **kwargs):
        return self.workflow.rewrite_query(*args, **kwargs)

    def expand_top_k(self, *args, **kwargs):
        return self.workflow.expand_top_k(*args, **kwargs)

    def decide_external_search(self, *args, **kwargs):
        return self.workflow.decide_external_search(*args, **kwargs)

    def decide_external_search_node(self, *args, **kwargs):
        return self.workflow.decide_external_search_node(*args, **kwargs)

    async def web_search_node(self, *args, **kwargs):
        return await self.workflow.web_search_node(*args, **kwargs)

    def merge_evidence_node(self, *args, **kwargs):
        return self.workflow.merge_evidence_node(*args, **kwargs)

    async def generate_answer(self, *args, **kwargs):
        return await self.workflow.generate_answer(*args, **kwargs)

    def build_insufficient_evidence(self, *args, **kwargs):
        return self.workflow.build_insufficient_evidence(*args, **kwargs)

    async def finalize_trace(self, *args, **kwargs):
        return await self.workflow.finalize_trace(*args, **kwargs)

    def _record_event(self, *args, **kwargs):
        return self.workflow._record_event(*args, **kwargs)

    def _now(self, *args, **kwargs):
        return self.workflow._now(*args, **kwargs)
