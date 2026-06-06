from datetime import UTC, datetime
from time import perf_counter
from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from app.rag.decomposition import decompose_query, merge_decomposed_scores
from app.schemas.rag import (
    EvaluationResult,
    RagDocument,
    RagMetrics,
    RagPlan,
    RagResponse,
    RagSource,
    RagState,
    RagStrategyConfig,
    RagStrategySummary,
    SubQuery as RagSubQuery,
)
from app.schemas.rag_debug import (
    EvaluationTrace,
    GenerationTrace,
    PlannerTrace,
    RagDebugTrace,
    RagStrategyTrace,
    RetrievalAttemptTrace,
    RouteDecisionTrace,
)
from app.services.rag_debug_trace_store import debug_trace_store


class EnterpriseRagGraphState(TypedDict):
    rag_state: RagState
    trace: NotRequired[RagDebugTrace]
    started: NotRequired[float]
    started_at: NotRequired[str]
    retry_reason: NotRequired[str | None]
    response: NotRequired[RagResponse]


class EnterpriseRagGraph:
    def __init__(self, service=None, trace_store=None, strategy_router=None, retrieval_pipeline=None):
        if service is None:
            from app.rag.enterprise_rag_service import enterprise_rag_service

            service = enterprise_rag_service
        if strategy_router is None:
            from app.rag.strategy_router import strategy_router as default_strategy_router

            strategy_router = default_strategy_router
        if retrieval_pipeline is None:
            from app.rag.retrieval_pipeline import RetrievalPipeline

            retrieval_pipeline = RetrievalPipeline(service)
        self.service = service
        self.strategy_router = strategy_router
        self.retrieval_pipeline = retrieval_pipeline
        self.trace_store = trace_store or debug_trace_store
        self.graph = self._build_graph()

    async def run(self, state: RagState) -> RagResponse:
        result = await self.graph.ainvoke({"rag_state": state})
        return result["response"]

    def _build_graph(self):
        graph = StateGraph(EnterpriseRagGraphState)
        graph.add_node("initialize", self.initialize_node)
        graph.add_node("planner", self.planner_node)
        graph.add_node("strategy_select", self.strategy_select_node)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("evaluate_context", self.evaluate_context_node)
        graph.add_node("decide_next_action", self.decide_next_action_node)
        graph.add_node("apply_retry", self.apply_retry_node)
        graph.add_node("generate_answer", self.generate_answer_node)
        graph.add_node("build_insufficient_evidence", self.build_insufficient_evidence_node)
        graph.add_node("finalize_trace", self.finalize_trace_node)

        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "planner")
        graph.add_edge("planner", "strategy_select")
        graph.add_edge("strategy_select", "retrieve")
        graph.add_edge("retrieve", "evaluate_context")
        graph.add_edge("evaluate_context", "decide_next_action")
        graph.add_conditional_edges(
            "decide_next_action",
            self.route_after_decision,
            {
                "retry": "apply_retry",
                "generate": "generate_answer",
                "insufficient_evidence": "build_insufficient_evidence",
            },
        )
        graph.add_edge("apply_retry", "retrieve")
        graph.add_edge("generate_answer", "finalize_trace")
        graph.add_edge("build_insufficient_evidence", "finalize_trace")
        graph.add_edge("finalize_trace", END)
        return graph.compile()

    def initialize_node(self, graph_state: EnterpriseRagGraphState) -> dict[str, Any]:
        started = perf_counter()
        started_at = self._now()
        return {
            "started": started,
            "started_at": started_at,
            "trace": self.initialize_trace(graph_state["rag_state"], started_at),
        }

    def planner_node(self, graph_state: EnterpriseRagGraphState) -> dict[str, Any]:
        self.planner(graph_state["rag_state"], graph_state["trace"])
        return {}

    def strategy_select_node(self, graph_state: EnterpriseRagGraphState) -> dict[str, Any]:
        self.strategy_select(graph_state["rag_state"], graph_state["trace"])
        return {}

    async def retrieve_node(self, graph_state: EnterpriseRagGraphState) -> dict[str, Any]:
        await self.retrieve(
            graph_state["rag_state"],
            graph_state["trace"],
            reason=graph_state.get("retry_reason") or "Initial hybrid retrieval.",
        )
        return {"retry_reason": None}

    def evaluate_context_node(self, graph_state: EnterpriseRagGraphState) -> dict[str, Any]:
        self.evaluate_context(graph_state["rag_state"], graph_state["trace"])
        return {}

    def decide_next_action_node(self, graph_state: EnterpriseRagGraphState) -> dict[str, Any]:
        self.decide_next_action(graph_state["rag_state"])
        return {}

    def apply_retry_node(self, graph_state: EnterpriseRagGraphState) -> dict[str, Any]:
        state = graph_state["rag_state"]
        retry_action = state.next_action
        if retry_action == "rewrite_query":
            self.rewrite_query(state)
        elif retry_action == "expand_top_k":
            self.expand_top_k(state)
        else:
            return {"retry_reason": None}
        return {"retry_reason": f"Retry after {retry_action}."}

    async def generate_answer_node(self, graph_state: EnterpriseRagGraphState) -> dict[str, Any]:
        await self.generate_answer(graph_state["rag_state"], graph_state["trace"])
        return {}

    def build_insufficient_evidence_node(self, graph_state: EnterpriseRagGraphState) -> dict[str, Any]:
        self.build_insufficient_evidence(graph_state["rag_state"])
        return {}

    async def finalize_trace_node(self, graph_state: EnterpriseRagGraphState) -> dict[str, Any]:
        response = await self.finalize_trace(
            graph_state["rag_state"],
            graph_state["trace"],
            graph_state["started"],
            graph_state["started_at"],
        )
        return {"response": response}

    def route_after_decision(self, graph_state: EnterpriseRagGraphState) -> str:
        state = graph_state["rag_state"]
        if state.next_action in {"rewrite_query", "expand_top_k"}:
            if state.retry_count < state.max_retries:
                return "retry"
            return "insufficient_evidence"
        if state.next_action == "generate":
            return "generate"
        return "insufficient_evidence"

    def initialize_trace(self, state: RagState, started_at: str) -> RagDebugTrace:
        return RagDebugTrace(
            request_id=state.request_id,
            debug_id=state.debug_id,
            session_id=state.session_id,
            user_id=state.user_id,
            route_decision=RouteDecisionTrace(
                route=state.route,
                rag_intent=state.rag_intent,
                source_hints=state.source_hints,
                confidence=state.router_confidence,
                reason=state.router_reason,
            ),
            started_at=started_at,
        )

    def planner(self, state: RagState, trace: RagDebugTrace) -> None:
        task_type = self._task_type_for_intent(state.rag_intent, state.original_query)
        state.plan = RagPlan(
            task_type=task_type,
            needs_rewrite=False,
            needs_decompose=task_type in {"multi_hop", "comparison"},
            expected_evidence_count=1,
            required_aspects=[state.original_query],
            constraints={"source_hints": state.source_hints},
            reason=f"Planned from router intent: {state.rag_intent}",
        )
        trace.planner = PlannerTrace(plan=state.plan)
        self._record_event(state, "rag_plan_created", "planner", data=state.plan.model_dump())

    def strategy_select(self, state: RagState, trace: RagDebugTrace) -> None:
        strategy = self.strategy_router.select(state)
        state.strategy = strategy
        trace.strategy = RagStrategyTrace(
            strategy=strategy,
            reason="Selected by StrategyRouter from rag_intent and router confidence.",
        )
        self._record_event(state, "strategy_selected", "strategy_select", data=strategy.model_dump())

    async def retrieve(self, state: RagState, trace: RagDebugTrace, reason: str = "Initial hybrid retrieval.") -> None:
        strategy = state.strategy or RagStrategyConfig()
        if strategy.use_decompose:
            await self.retrieve_decomposed(state, trace, reason)
            return

        attempt_id = len(state.retrieval_attempts) + 1
        self._record_event(
            state,
            "retrieval_started",
            "retrieve",
            data={"attempt_id": attempt_id, "query": state.current_query},
        )
        result = await self.retrieval_pipeline.run(
            query=state.current_query,
            strategy=strategy,
            source_hints=state.source_hints,
            rag_intent=state.rag_intent,
            router_confidence=state.router_confidence,
            attempt_id=attempt_id,
            reason=reason,
        )
        state.selected_documents = result.selected_documents
        state.retrieval_attempts.append(result.attempt)
        trace.retrieval_attempts.append(RetrievalAttemptTrace(attempt=result.attempt))
        self._record_event(
            state,
            "retrieval_finished",
            "retrieve",
            data=self._retrieval_event_data(result, sub_query_id=None),
        )

    async def retrieve_decomposed(self, state: RagState, trace: RagDebugTrace, reason: str) -> None:
        strategy = state.strategy or RagStrategyConfig()
        sub_query_plan = await decompose_query(state.current_query)
        if sub_query_plan.fallback_reason or not sub_query_plan.sub_queries:
            state.sub_queries = []
            if state.plan:
                state.plan.required_aspects = [state.original_query]
                state.plan.expected_evidence_count = 1
            self._record_event(
                state,
                "query_decomposition_failed",
                "retrieve",
                data={"reason": sub_query_plan.fallback_reason or "empty_sub_query_plan"},
            )
            strategy.use_decompose = False
            await self.retrieve(state, trace, reason)
            strategy.use_decompose = True
            return

        state.sub_queries = [
            RagSubQuery(sub_query_id=sub_query.id, query=sub_query.query, reason=sub_query.purpose)
            for sub_query in sub_query_plan.sub_queries
        ]
        if state.plan:
            state.plan.required_aspects = [sub_query.query for sub_query in sub_query_plan.sub_queries]
            state.plan.expected_evidence_count = len(sub_query_plan.sub_queries)
        self._record_event(
            state,
            "query_decomposed",
            "retrieve",
            data={
                "sub_queries": [sub_query.model_dump() for sub_query in state.sub_queries],
                "original_query": state.current_query,
            },
        )

        documents_by_key: dict[str, RagDocument] = {}
        sub_query_rankings: dict[str, dict[str, float]] = {}
        sub_query_text_by_id = {sub_query.id: sub_query.query for sub_query in sub_query_plan.sub_queries}
        for sub_query in sub_query_plan.sub_queries:
            attempt_id = len(state.retrieval_attempts) + 1
            self._record_event(
                state,
                "retrieval_started",
                "retrieve",
                data={"attempt_id": attempt_id, "sub_query_id": sub_query.id, "query": sub_query.query},
            )
            result = await self.retrieval_pipeline.run(
                query=sub_query.query,
                strategy=strategy,
                source_hints=state.source_hints,
                rag_intent=state.rag_intent,
                router_confidence=state.router_confidence,
                attempt_id=attempt_id,
                sub_query_id=sub_query.id,
                reason=reason,
            )
            ranking: dict[str, float] = {}
            for rank, document in enumerate(result.selected_documents, start=1):
                key = document.parent_chunk_id or document.source_id
                documents_by_key.setdefault(key, document)
                ranking[key] = document.score or 1.0 / rank
            sub_query_rankings[sub_query.id] = ranking
            state.retrieval_attempts.append(result.attempt)
            trace.retrieval_attempts.append(RetrievalAttemptTrace(attempt=result.attempt))
            self._record_event(
                state,
                "retrieval_finished",
                "retrieve",
                data=self._retrieval_event_data(result, sub_query_id=sub_query.id),
            )

        merged_scores = merge_decomposed_scores(sub_query_rankings, total_sub_queries=len(sub_query_plan.sub_queries))
        selected_documents = []
        for parent_chunk_id, candidate in sorted(
            merged_scores.items(), key=lambda item: item[1].final_score, reverse=True
        ):
            document = documents_by_key[parent_chunk_id]
            document.score = candidate.final_score
            document.metadata = {
                **document.metadata,
                "matched_sub_query_ids": candidate.matched_sub_query_ids,
                "matched_sub_queries": [
                    sub_query_text_by_id[sub_query_id]
                    for sub_query_id in candidate.matched_sub_query_ids
                    if sub_query_id in sub_query_text_by_id
                ],
            }
            selected_documents.append(document)
        state.selected_documents = selected_documents[: strategy.final_top_k]

    def evaluate_context(self, state: RagState, trace: RagDebugTrace) -> None:
        required_aspects = state.plan.required_aspects if state.plan else [state.original_query]
        if state.sub_queries:
            matched_sub_query_ids = {
                sub_query_id
                for document in state.selected_documents
                for sub_query_id in document.metadata.get("matched_sub_query_ids", [])
            }
            covered_aspects = [
                sub_query.query for sub_query in state.sub_queries if sub_query.sub_query_id in matched_sub_query_ids
            ]
            missing_aspects = [
                sub_query.query for sub_query in state.sub_queries if sub_query.sub_query_id not in matched_sub_query_ids
            ]
            score = len(covered_aspects) / max(1, len(state.sub_queries))
            enough = bool(state.selected_documents) and not missing_aspects and not state.acl_filter_removed_all_candidates
        else:
            enough = bool(state.selected_documents) and not state.acl_filter_removed_all_candidates
            score = min(1.0, len(state.selected_documents) / max(1, len(required_aspects))) if enough else 0.0
            covered_aspects = required_aspects if enough else []
            missing_aspects = [] if enough else required_aspects
        result = EvaluationResult(
            enough_evidence=enough,
            context_score=score,
            coverage_score=score,
            citation_readiness_score=score,
            covered_aspects=covered_aspects,
            missing_aspects=missing_aspects,
            partial_answer_allowed=False,
            suggested_action="generate" if enough else "rewrite_query",
            user_visible_reason="" if enough else "没有找到足够信息来回答该问题。",
            reason="Rule-based MVP evaluation.",
        )
        state.evaluator_result = result
        trace.evaluations.append(EvaluationTrace(result=result))
        self._record_event(
            state,
            "evaluation_finished",
            "evaluate_context",
            data={
                "enough_evidence": result.enough_evidence,
                "suggested_action": result.suggested_action,
                "missing_aspects": result.missing_aspects,
            },
        )

    def decide_next_action(self, state: RagState) -> None:
        if state.security_flags or state.acl_filter_removed_all_candidates:
            state.next_action = "insufficient_evidence"
            return
        if state.evaluator_result and state.evaluator_result.enough_evidence:
            state.next_action = "generate"
            return
        if state.retry_count >= state.max_retries:
            state.next_action = "insufficient_evidence"
            return
        suggested_action = state.evaluator_result.suggested_action if state.evaluator_result else "insufficient_evidence"
        state.next_action = suggested_action if suggested_action in {"rewrite_query", "expand_top_k"} else "insufficient_evidence"
        if state.next_action in {"rewrite_query", "expand_top_k"}:
            self._record_event(
                state,
                "retry_decided",
                "decide_next_action",
                data={"retry_count": state.retry_count, "next_action": state.next_action},
            )

    def rewrite_query(self, state: RagState) -> None:
        state.retry_count += 1
        hints = " ".join(state.source_hints)
        rewritten = f"{state.original_query} {hints} enterprise knowledge evidence".strip()
        state.current_query = rewritten
        state.rewritten_queries.append(rewritten)
        self._record_event(
            state,
            "query_rewritten",
            "rewrite_query",
            data={"retry_count": state.retry_count, "query": rewritten},
        )

    def expand_top_k(self, state: RagState) -> None:
        state.retry_count += 1
        strategy = state.strategy or RagStrategyConfig()
        strategy.final_top_k = min(strategy.final_top_k * 2, 20)
        strategy.fusion_top_k = min(strategy.fusion_top_k * 2, 120)
        state.strategy = strategy
        self._record_event(
            state,
            "topk_expanded",
            "expand_top_k",
            data={"retry_count": state.retry_count, "final_top_k": strategy.final_top_k},
        )

    async def generate_answer(self, state: RagState, trace: RagDebugTrace) -> None:
        started = perf_counter()
        self._record_event(state, "answer_started", "generate_answer")
        answer = await self.service.generate_answer(
            state.current_query,
            state.selected_documents,
            memory_context=state.memory_context,
        )
        state.answer = answer
        state.sources = [self._to_rag_source(document) for document in state.selected_documents]
        trace.generation = GenerationTrace(answer_preview=answer[:300], elapsed_ms=(perf_counter() - started) * 1000)

    def build_insufficient_evidence(self, state: RagState) -> None:
        missing = state.evaluator_result.missing_aspects if state.evaluator_result else [state.original_query]
        detail = "、".join(missing) if missing else state.original_query
        state.answer = f"抱歉，我没有找到足够信息来回答：{detail}"
        state.sources = []

    async def finalize_trace(
        self,
        state: RagState,
        trace: RagDebugTrace,
        started: float,
        started_at: str,
    ) -> RagResponse:
        total_ms = (perf_counter() - started) * 1000
        trace.finished_at = self._now()
        trace.total_ms = total_ms
        trace.final_answer_preview = (state.answer or "")[:300]
        trace.final_sources = state.sources
        trace.warnings = list(state.warnings)
        trace.error = state.error

        try:
            await self.trace_store.save(trace)
        except Exception as exc:
            state.warnings.append(f"Trace write failed: {exc}")

        strategy = state.strategy or RagStrategyConfig()
        evaluation = state.evaluator_result
        return RagResponse(
            request_id=state.request_id,
            debug_id=state.debug_id,
            session_id=state.session_id,
            answer=state.answer or "",
            sources=state.sources,
            strategy=RagStrategySummary(
                strategy_name=strategy.strategy_name,
                query_type=state.rag_intent,
                retrieval_mode=strategy.retrieval_mode,
                final_top_k=strategy.final_top_k,
                use_reranker=strategy.use_reranker,
                use_query_rewrite=strategy.use_query_rewrite,
                use_decompose=strategy.use_decompose,
                retry_count=state.retry_count,
            ),
            evaluation=None
            if evaluation is None
            else {
                "enough_evidence": evaluation.enough_evidence,
                "covered_aspects": evaluation.covered_aspects,
                "missing_aspects": evaluation.missing_aspects,
                "user_visible_reason": evaluation.user_visible_reason or None,
            },
            metrics=self._aggregate_metrics(state, trace, total_ms),
            warnings=state.warnings,
        )

    def _retrieval_event_data(self, result, sub_query_id: str | None) -> dict[str, Any]:
        data = {
            "attempt_id": result.attempt.attempt_id,
            "selected_documents": len(result.selected_documents),
            "dense_candidates": len(result.attempt.dense_results),
            "bm25_candidates": len(result.attempt.bm25_results),
            "fused_candidates": len(result.attempt.fused_results),
            "reranked_candidates": len(result.attempt.reranked_results),
            "dense_ms": result.metrics.dense_ms,
            "bm25_ms": result.metrics.bm25_ms,
            "rrf_ms": result.metrics.rrf_ms,
            "rerank_ms": result.metrics.rerank_ms,
            "total_ms": result.metrics.total_ms,
        }
        if sub_query_id:
            data["sub_query_id"] = sub_query_id
        return data

    def _aggregate_metrics(self, state: RagState, trace: RagDebugTrace, total_ms: float) -> RagMetrics:
        dense_ms = sum((attempt.dense_ms or 0.0) for attempt in state.retrieval_attempts)
        bm25_ms = sum((attempt.bm25_ms or 0.0) for attempt in state.retrieval_attempts)
        rrf_ms = sum((attempt.rrf_ms or 0.0) for attempt in state.retrieval_attempts)
        rerank_ms = sum((attempt.rerank_ms or 0.0) for attempt in state.retrieval_attempts)
        retrieval_ms = sum(attempt.elapsed_ms for attempt in state.retrieval_attempts)
        return RagMetrics(
            retry_count=state.retry_count,
            retrieval_attempts=len(state.retrieval_attempts),
            dense_ms=dense_ms,
            bm25_ms=bm25_ms,
            rrf_ms=rrf_ms,
            rerank_ms=rerank_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=trace.generation.elapsed_ms if trace.generation else None,
            total_ms=total_ms,
        )

    def _record_event(
        self,
        state: RagState,
        event: str,
        stage: str,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        state.sse_events.append(
            {
                "type": event,
                "event": event,
                "request_id": state.request_id,
                "debug_id": state.debug_id,
                "session_id": state.session_id,
                "stage": stage,
                "message": message,
                "data": data or {},
                "timestamp": self._now(),
            }
        )

    @staticmethod
    def _task_type_for_intent(rag_intent: str, query: str) -> str:
        if rag_intent in {"fact_lookup", "semantic_query", "multi_hop", "comparison", "procedure", "constrained", "follow_up"}:
            return rag_intent
        if rag_intent in {"semantic", "high_level"}:
            return "semantic_query"
        if rag_intent in {"intra_document_reasoning", "project_related", "completeness"}:
            return "multi_hop"
        if rag_intent == "conflicting_info" or "compare" in query.lower():
            return "comparison"
        return "fact_lookup"


    @staticmethod
    def _to_rag_document(document: Any) -> RagDocument:
        data = document if isinstance(document, dict) else document.to_dict()
        parent_chunk_id = data.get("parent_chunk_id", "")
        parent_doc_id = data.get("parent_doc_id", "")
        metadata = dict(data.get("metadata") or {})
        text = data.get("parent_text") or data.get("text") or ""
        return RagDocument(
            source_id=parent_chunk_id or parent_doc_id or data.get("title", "source"),
            parent_doc_id=parent_doc_id,
            parent_chunk_id=parent_chunk_id,
            source_type=data.get("source_type", metadata.get("source_type", "")),
            title=data.get("title", metadata.get("title", "")),
            section_heading=data.get("section_heading", metadata.get("section_heading", "")),
            score=float(data.get("score", 0.0) or 0.0),
            text=text,
            child_text=data.get("child_text", ""),
            metadata=metadata,
        )

    @staticmethod
    def _to_rag_source(document: RagDocument) -> RagSource:
        return RagSource(
            source_id=document.source_id,
            title=document.title,
            source_type=document.source_type,
            parent_doc_id=document.parent_doc_id,
            parent_chunk_id=document.parent_chunk_id,
            section_heading=document.section_heading,
            score=document.score,
            metadata=document.metadata,
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
