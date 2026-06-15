from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from app.rag.decomposition import decompose_query, merge_decomposed_scores
from app.rag.web_search import normalize_web_search_results
from app.schemas.rag import (
    EvaluationResult,
    ExternalSearchDecision,
    RagDocument,
    RagMetrics,
    RagPlan,
    RagResponse,
    RagSource,
    RagState,
    RagStrategyConfig,
    RagStrategySummary,
    SubQuery as RagSubQuery,
    WebSearchResult,
)
from app.schemas.rag_debug import (
    EvaluationTrace,
    ExternalSearchDecisionTrace,
    GenerationTrace,
    PlannerTrace,
    RagDebugTrace,
    RagStrategyTrace,
    RetrievalAttemptTrace,
    RouteDecisionTrace,
    WebSearchTrace,
)
from app.services.rag_debug_trace_store import debug_trace_store


class RagEvidenceWorkflow:
    def __init__(self, service=None, trace_store=None, strategy_router=None, retrieval_pipeline=None, web_search_service=None):
        if service is None:
            from app.rag.enterprise_rag_service import enterprise_rag_service

            service = enterprise_rag_service
        if strategy_router is None:
            from app.rag.strategy_router import strategy_router as default_strategy_router

            strategy_router = default_strategy_router
        if retrieval_pipeline is None:
            from app.rag.retrieval_backends.factory import build_enterprise_retrieval_backend
            from app.rag.retrieval_pipeline import RetrievalPipeline

            retrieval_backend = build_enterprise_retrieval_backend(service=service)
            retrieval_pipeline = RetrievalPipeline(retrieval_backend)
        if web_search_service is None:
            from app.rag.web_search import web_search_service as default_web_search_service

            web_search_service = default_web_search_service

        self.service = service
        self.strategy_router = strategy_router
        self.retrieval_pipeline = retrieval_pipeline
        self.trace_store = trace_store or debug_trace_store
        self.web_search_service = web_search_service

    async def run(self, state: RagState) -> RagResponse:
        started = perf_counter()
        started_at = self._now()
        trace = self.initialize_trace(state, started_at)

        self.planner(state, trace)
        self.strategy_select(state, trace)

        retry_reason = "Initial hybrid retrieval."
        while True:
            await self.retrieve(state, trace, reason=retry_reason)
            retry_reason = "Initial hybrid retrieval."
            self.evaluate_context(state, trace)
            self.decide_next_action(state)

            if state.next_action == "rewrite_query" and state.retry_count < state.max_retries:
                self.rewrite_query(state)
                retry_reason = "Retry after rewrite_query."
                continue
            if state.next_action == "expand_top_k" and state.retry_count < state.max_retries:
                self.expand_top_k(state)
                retry_reason = "Retry after expand_top_k."
                continue
            break

        if state.next_action == "generate":
            await self.generate_answer(state, trace)
        elif state.next_action == "external_search":
            self.decide_external_search_node(state, trace)
            if state.external_search_decision.allowed:
                await self.web_search_node(state, trace)
            if state.web_results:
                self.merge_evidence_node(state)
                await self.generate_answer(state, trace)
            elif state.selected_documents and state.evidence_mode == "hybrid":
                state.evidence_mode = "internal_only"
                await self.generate_answer(state, trace)
            else:
                self.build_insufficient_evidence(state)
        else:
            self.build_insufficient_evidence(state)

        return await self.finalize_trace(state, trace, started, started_at)

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
            metadata_filter=state.metadata_filter_decision,
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
                metadata_filter=state.metadata_filter_decision,
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
            state.next_action = "external_search" if self._needs_public_context(state.original_query) else "generate"
            return
        if state.retry_count >= state.max_retries:
            state.next_action = "external_search"
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

    def decide_external_search(self, state: RagState) -> None:
        if state.security_flags or state.acl_filter_removed_all_candidates:
            state.external_search_decision = ExternalSearchDecision(
                mode="none",
                allowed=False,
                reason="Security or ACL guard blocked external fallback.",
            )
            state.evidence_mode = "internal_only"
            return

        query = state.original_query.strip()
        asks_generic_fallback = self._asks_for_generic_fallback(query)
        needs_public_context = self._needs_public_context(query)
        has_internal_evidence = bool(state.evaluator_result and state.evaluator_result.enough_evidence)

        if has_internal_evidence:
            if needs_public_context:
                state.external_search_decision = ExternalSearchDecision(
                    mode="hybrid",
                    allowed=True,
                    reason="Internal evidence is sufficient and the question also benefits from public context.",
                    user_visible_label="公开资料参考",
                )
                state.evidence_mode = "hybrid"
                return
            state.external_search_decision = ExternalSearchDecision(
                mode="none",
                allowed=False,
                reason="Internal evidence is sufficient.",
            )
            state.evidence_mode = "internal_only"
            return

        if self._is_company_specific_fact(query):
            state.external_search_decision = ExternalSearchDecision(
                mode="none",
                allowed=False,
                reason="The question asks for company-specific information that public web results cannot replace.",
            )
            state.evidence_mode = "internal_only"
            return

        if self._asks_for_company_specific_procedure(query) and not asks_generic_fallback:
            state.external_search_decision = ExternalSearchDecision(
                mode="none",
                allowed=False,
                reason="The question asks for company-specific information that public web results cannot replace.",
            )
            state.evidence_mode = "internal_only"
            return

        if asks_generic_fallback:
            state.external_search_decision = ExternalSearchDecision(
                mode="fallback",
                allowed=True,
                reason="Internal evidence is insufficient and the question can be answered with general public reference.",
                user_visible_label="通用参考",
            )
            state.evidence_mode = "web_fallback"
            return

        if needs_public_context:
            state.external_search_decision = ExternalSearchDecision(
                mode="fallback",
                allowed=True,
                reason="Internal evidence is insufficient and the question needs public reference context.",
                user_visible_label="公开资料参考",
            )
            state.evidence_mode = "web_fallback"
            return

        if state.rag_intent == "procedure":
            state.external_search_decision = ExternalSearchDecision(
                mode="fallback",
                allowed=True,
                reason="Internal evidence is insufficient and the procedure question can use general public reference.",
                user_visible_label="通用参考",
            )
            state.evidence_mode = "web_fallback"
            return

        state.external_search_decision = ExternalSearchDecision(
            mode="none",
            allowed=False,
            reason="External fallback is not enabled for this RAG intent.",
        )
        state.evidence_mode = "internal_only"

    def decide_external_search_node(self, state: RagState, trace: RagDebugTrace) -> None:
        self.decide_external_search(state)
        trace.external_search_decision = ExternalSearchDecisionTrace(decision=state.external_search_decision)
        self._record_event(
            state,
            "external_search_decided",
            "decide_external_search",
            data=state.external_search_decision.model_dump(),
        )

    async def web_search_node(self, state: RagState, trace: RagDebugTrace) -> None:
        state.web_search_attempted = True
        started = perf_counter()
        query = state.original_query
        self._record_event(
            state,
            "web_search_started",
            "web_search",
            data={"query": query, "max_results": 3, "mode": state.external_search_decision.mode},
        )
        raw_results = await self.web_search_service.search(query, max_results=3)
        state.web_search_ms = (perf_counter() - started) * 1000
        state.web_results = normalize_web_search_results(raw_results, max_results=3)
        trace.web_search = WebSearchTrace(query=query, results=state.web_results, elapsed_ms=state.web_search_ms)
        if not state.web_results:
            state.external_search_decision.allowed = False
            state.external_search_decision.reason = "External search returned no usable results."
            trace.external_search_decision = ExternalSearchDecisionTrace(decision=state.external_search_decision)
        self._record_event(
            state,
            "web_search_finished",
            "web_search",
            data={"results": len(state.web_results), "elapsed_ms": state.web_search_ms},
        )

    def merge_evidence_node(self, state: RagState) -> None:
        state.sources = [self._to_rag_source(document) for document in state.selected_documents]
        state.sources.extend(RagSource.from_web_result(result) for result in state.web_results)
        self._record_event(
            state,
            "evidence_merged",
            "merge_evidence",
            data={
                "internal_sources": len(state.selected_documents),
                "web_sources": len(state.web_results),
                "evidence_mode": state.evidence_mode,
            },
        )

    async def generate_answer(self, state: RagState, trace: RagDebugTrace) -> None:
        started = perf_counter()
        self._record_event(state, "answer_started", "generate_answer")
        answer = await self.service.generate_answer(
            state.current_query,
            state.selected_documents,
            memory_context=state.memory_context,
            web_results=state.web_results,
            evidence_mode=state.evidence_mode,
        )
        state.answer = answer
        if not state.sources:
            state.sources = [self._to_rag_source(document) for document in state.selected_documents]
            state.sources.extend(RagSource.from_web_result(result) for result in state.web_results)
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
            web_search_ms=state.web_search_ms,
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
    def _final_top_k_for_intent(rag_intent: str) -> int:
        if rag_intent in {"multi_hop", "comparison", "completeness", "conflicting_info"}:
            return 10
        if rag_intent in {"constrained", "project_related"}:
            return 8
        return 5

    @staticmethod
    def _search_k_for_intent(rag_intent: str) -> int:
        if rag_intent in {"multi_hop", "comparison", "completeness", "conflicting_info"}:
            return 80
        if rag_intent in {"constrained", "project_related"}:
            return 60
        return 40

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
    def _to_web_document(result: Any) -> WebSearchResult:
        if isinstance(result, WebSearchResult):
            return result
        data = result if isinstance(result, dict) else {}
        return WebSearchResult(
            title=data.get("title", getattr(result, "title", "")),
            url=data.get("url", getattr(result, "url", "")),
            snippet=data.get("snippet", getattr(result, "snippet", "")),
            score=float(data.get("score", getattr(result, "score", 0.0)) or 0.0),
            provider=data.get("provider", getattr(result, "provider", "web")),
            metadata=dict(data.get("metadata", getattr(result, "metadata", {})) or {}),
        )

    @staticmethod
    def _has_internal_company_marker(query: str) -> bool:
        normalized_query = query.lower()
        internal_company_markers = [
            "我们公司",
            "我司",
            "本公司",
            "公司内部",
            "公司知识库",
            "正式制度",
            "our company",
            "my company",
            "at our company",
            "internal policy",
            "internal docs",
            "internal documentation",
            "internal knowledge base",
            "our internal",
            "nexuskb",
        ]
        return any(marker in normalized_query for marker in internal_company_markers)

    @classmethod
    def _is_company_specific_fact(cls, query: str) -> bool:
        normalized_query = query.lower()
        private_fact_markers = [
            "员工工资",
            "薪资",
            "salary",
            "salary band",
            "salary bands",
            "绩效",
            "报销上限",
            "reimbursement limit",
            "deployment endpoint",
            "endpoint",
            "审批人",
            "负责人",
        ]
        specific_markers = [
            "是多少",
            "是什么",
            "具体",
            "上限",
            "limit",
            "金额",
            "谁",
            "哪位",
            "工号",
            "名单",
            "电话",
            "邮箱",
            "what is",
            "how much",
            "salary band",
            "salary bands",
            "who",
        ]
        has_internal_company_marker = cls._has_internal_company_marker(query)
        has_private_fact_marker = any(marker in normalized_query for marker in private_fact_markers)
        has_specific_marker = any(marker in normalized_query for marker in specific_markers)
        return has_internal_company_marker and has_private_fact_marker and (
            has_specific_marker or cls._asks_for_generic_fallback(query)
        )

    @classmethod
    def _asks_for_company_specific_procedure(cls, query: str) -> bool:
        normalized_query = query.lower()
        procedure_markers = [
            "流程",
            "程序",
            "步骤",
            "申请",
            "报销",
            "休假",
            "pto",
            "procedure",
            "process",
            "how do i",
            "how to",
            "request",
        ]
        return cls._has_internal_company_marker(query) and any(marker in normalized_query for marker in procedure_markers)

    @staticmethod
    def _asks_for_generic_fallback(query: str) -> bool:
        normalized_query = query.lower()
        explicit_missing_markers = [
            "如果公司知识库没有",
            "如果知识库没有",
            "知识库没有",
            "公司知识库没有",
            "找不到",
            "缺少",
            "未找到",
            "if the knowledge base does not have",
            "if our knowledge base does not have",
            "if no internal",
            "if internal docs do not have",
            "if internal documentation does not have",
            "not found",
            "no internal",
        ]
        generic_markers = [
            "通用",
            "一般",
            "常见",
            "generic",
            "general",
            "common",
            "best practices",
        ]
        explicit_generic_reference_markers = [
            "通用参考",
            "通用流程参考",
            "一般参考",
            "常见参考",
            "generic reference",
            "general reference",
            "common reference",
        ]
        has_explicit_missing = any(marker in normalized_query for marker in explicit_missing_markers)
        has_generic_reference = any(marker in normalized_query for marker in explicit_generic_reference_markers) or any(
            marker in normalized_query for marker in generic_markers
        )
        return has_explicit_missing and has_generic_reference

    @staticmethod
    def _needs_public_context(query: str) -> bool:
        normalized_query = query.lower()
        public_markers = [
            "业界",
            "行业",
            "公开",
            "最佳实践",
            "趋势",
            "对比",
            "差距",
            "通用",
            "常见",
            "industry",
            "public",
            "best practice",
            "best practices",
            "trend",
            "benchmark",
            "common",
            "generic",
            "general",
        ]
        return any(marker in normalized_query for marker in public_markers)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
