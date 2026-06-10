import json
from contextvars import ContextVar
from time import perf_counter
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.rag.enterprise_rag_graph import EnterpriseRagGraph
from app.schemas.rag import AgenticActionDecision, RagState, RagStrategyConfig, RetrievalAttempt
from app.schemas.rag_debug import RagDebugTrace, RetrievalAttemptTrace


_active_trace_var: ContextVar[RagDebugTrace | None] = ContextVar("agentic_rag_active_trace", default=None)


class AgenticRagGraph(EnterpriseRagGraph):
    def __init__(self, service=None, trace_store=None, decision_chain=None):
        super().__init__(service=service, trace_store=trace_store)
        self.decision_chain = decision_chain
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RagState)
        graph.add_node("initialize", self.initialize_node)
        graph.add_node("understand_request", self.understand_request_node)
        graph.add_node("safety_check", self.safety_check_node)
        graph.add_node("direct_answer", self.direct_answer_node)
        graph.add_node("clarify", self.clarify_node)
        graph.add_node("refuse", self.refuse_node)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("evaluate_context", self.evaluate_context_node)
        graph.add_node("decide_next_action", self.decide_next_action_node)
        graph.add_node("apply_retry", self.apply_retry_node)
        graph.add_node("external_search", self.external_search_node)
        graph.add_node("generate_answer", self.generate_answer_node)
        graph.add_node("finalize_trace", self.finalize_trace_node)

        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "understand_request")
        graph.add_edge("understand_request", "safety_check")
        graph.add_conditional_edges(
            "safety_check",
            self.route_after_safety_check,
            {
                "direct_answer": "direct_answer",
                "clarify": "clarify",
                "refuse": "refuse",
                "retrieve": "retrieve",
            },
        )
        graph.add_edge("direct_answer", "finalize_trace")
        graph.add_edge("clarify", "finalize_trace")
        graph.add_edge("refuse", "finalize_trace")
        graph.add_edge("retrieve", "evaluate_context")
        graph.add_edge("evaluate_context", "decide_next_action")
        graph.add_conditional_edges(
            "decide_next_action",
            self.route_after_decide_next_action,
            {
                "apply_retry": "apply_retry",
                "external_search": "external_search",
                "generate_answer": "generate_answer",
                "finalize_trace": "finalize_trace",
            },
        )
        graph.add_edge("apply_retry", "retrieve")
        graph.add_edge("external_search", "generate_answer")
        graph.add_edge("generate_answer", "finalize_trace")
        graph.add_edge("finalize_trace", END)
        return graph.compile()

    async def run(self, state: RagState):
        started = perf_counter()
        started_at = self._now()
        trace = self.initialize_trace(state, started_at)
        trace_token = _active_trace_var.set(trace)
        try:
            result = await self.graph.ainvoke(state)
            if isinstance(result, dict):
                updated_state = RagState.model_validate(result)
                for field_name in RagState.model_fields:
                    setattr(state, field_name, getattr(updated_state, field_name))
            elif isinstance(result, RagState) and result is not state:
                for field_name in RagState.model_fields:
                    setattr(state, field_name, getattr(result, field_name))
            response = await self.finalize_trace(state, trace, started, started_at)
            object.__setattr__(response.strategy, "query_type", state.rag_intent)
            return response
        finally:
            _active_trace_var.reset(trace_token)

    def initialize_node(self, state: RagState) -> RagState:
        self._record_event(state, "agentic_graph_initialized", "initialize")
        return state

    async def understand_request_node(self, state: RagState) -> RagState:
        if self.decision_chain is None:
            decision = AgenticActionDecision(
                intent=state.rag_intent,
                action=state.action,
                needs_retrieval=state.needs_retrieval,
                needs_tool=state.needs_tool,
                needs_clarification=state.needs_clarification,
                safety_risk=state.safety_risk,
                source_hints=list(state.source_hints),
                required_tools=list(state.required_tools),
                confidence=state.router_confidence,
                reason=state.router_reason,
            )
        else:
            payload = {
                "query": state.current_query,
                "original_query": state.original_query,
                "history": self._format_history_preview(state),
                "memory_summary": state.memory_summary,
                "long_term_memories": state.long_term_memories,
            }
            decision = self._parse_decision(await self.decision_chain.ainvoke(payload))

        state.intent = decision.intent
        state.rag_intent = decision.intent
        state.action = decision.action
        state.needs_retrieval = decision.needs_retrieval
        state.needs_tool = decision.needs_tool
        state.needs_clarification = decision.needs_clarification
        state.safety_risk = decision.safety_risk
        state.source_hints = list(decision.source_hints)
        state.required_tools = list(decision.required_tools)
        state.router_confidence = decision.confidence
        state.router_reason = decision.reason
        self._update_trace_route_decision(state)
        self._record_event(
            state,
            "agentic_action_decided",
            "understand_request",
            data=decision.model_dump(),
        )
        return state

    def safety_check_node(self, state: RagState) -> RagState:
        if state.safety_risk:
            state.action = "refuse"
        elif state.needs_clarification:
            state.action = "clarify"
        elif state.needs_retrieval:
            state.action = "retrieve"
        elif state.action not in {"direct_answer", "tool_call", "refuse", "clarify", "retrieve"}:
            state.action = "retrieve"

        if state.action == "tool_call":
            state.action = "retrieve"
        self._record_event(state, "agentic_action_routed", "safety_check", data={"action": state.action})
        return state

    def direct_answer_node(self, state: RagState) -> RagState:
        state.response_type = "answer"
        state.answer = f"这是一个通用问题，可以不检索企业知识库直接回答：{state.original_query}"
        state.sources = []
        self._record_event(state, "direct_answer_created", "direct_answer")
        return state

    def clarify_node(self, state: RagState) -> RagState:
        state.response_type = "clarification"
        state.answer = f"需要再确认一下您的问题：{state.original_query}。请补充目标、范围或上下文。"
        state.sources = []
        self._record_event(state, "clarification_created", "clarify")
        return state

    def refuse_node(self, state: RagState) -> RagState:
        state.response_type = "refusal"
        state.answer = "不能执行该请求，因为它可能带来安全或破坏性风险。"
        state.sources = []
        self._record_event(state, "refusal_created", "refuse")
        return state

    async def retrieve_node(self, state: RagState) -> RagState:
        trace = self._require_trace()
        if state.plan is None:
            self.planner(state, trace)
        if state.strategy is None:
            self.strategy_select(state, trace)
        await self.retrieve(state, trace)
        return state

    async def retrieve(self, state: RagState, trace: RagDebugTrace, reason: str = "Initial hybrid retrieval.") -> None:
        if hasattr(self.service, "retrieve") or not hasattr(self.service, "retrieve_with_details"):
            await super().retrieve(state, trace, reason)
            return

        strategy = state.strategy or RagStrategyConfig()
        if strategy.use_decompose:
            await self.retrieve_decomposed(state, trace, reason)
            return

        started = perf_counter()
        attempt_id = len(state.retrieval_attempts) + 1
        self._record_event(
            state,
            "retrieval_started",
            "retrieve",
            data={"attempt_id": attempt_id, "query": state.current_query},
        )
        details = await self.service.retrieve_with_details(
            query=state.current_query,
            k=strategy.final_top_k,
            search_k=strategy.fusion_top_k,
            source_hints=state.source_hints,
            strict_source_filter=False,
            rag_intent=state.rag_intent,
            router_confidence=state.router_confidence,
            use_reranker=strategy.use_reranker,
        )
        raw_documents = details.get("selected_documents") or details.get("reranked_results") or details.get("fused_results") or []
        selected_documents = [self._to_rag_document(document) for document in raw_documents]
        state.selected_documents = selected_documents
        attempt = RetrievalAttempt(
            attempt_id=attempt_id,
            query=state.current_query,
            strategy_name=strategy.strategy_name,
            selected_documents=selected_documents,
            elapsed_ms=(perf_counter() - started) * 1000,
            reason=reason,
        )
        state.retrieval_attempts.append(attempt)
        trace.retrieval_attempts.append(RetrievalAttemptTrace(attempt=attempt))
        self._record_event(
            state,
            "retrieval_finished",
            "retrieve",
            data={"attempt_id": attempt.attempt_id, "selected_documents": len(selected_documents)},
        )

    def evaluate_context_node(self, state: RagState) -> RagState:
        self.evaluate_context(state, self._require_trace())
        return state

    def decide_next_action_node(self, state: RagState) -> RagState:
        self.decide_next_action(state)
        return state

    def apply_retry_node(self, state: RagState) -> RagState:
        if state.next_action == "rewrite_query":
            self.rewrite_query(state)
        elif state.next_action == "expand_top_k":
            self.expand_top_k(state)
        return state

    def external_search_node(self, state: RagState) -> RagState:
        self._record_event(state, "external_search_skipped", "external_search")
        return state

    async def generate_answer_node(self, state: RagState) -> RagState:
        if state.next_action == "generate":
            await self.generate_answer(state, self._require_trace())
        else:
            self.build_insufficient_evidence(state)
        return state

    def finalize_trace_node(self, state: RagState) -> RagState:
        self._record_event(state, "agentic_graph_finished", "finalize_trace")
        return state

    def route_after_safety_check(self, state: RagState) -> str:
        return state.action if state.action in {"direct_answer", "clarify", "refuse"} else "retrieve"

    def route_after_decide_next_action(self, state: RagState) -> str:
        if state.next_action in {"rewrite_query", "expand_top_k"} and state.retry_count < state.max_retries:
            return "apply_retry"
        if state.next_action == "external_search":
            return "external_search"
        if state.next_action == "generate":
            return "generate_answer"
        return "finalize_trace"

    def _parse_decision(self, raw: Any) -> AgenticActionDecision:
        if isinstance(raw, AgenticActionDecision):
            return raw
        if isinstance(raw, str):
            return AgenticActionDecision.model_validate(json.loads(raw))
        if isinstance(raw, dict):
            return AgenticActionDecision.model_validate(raw)
        if hasattr(raw, "model_dump"):
            return AgenticActionDecision.model_validate(raw.model_dump())
        raise TypeError(f"Unsupported agentic action decision: {type(raw)!r}")

    def _format_history_preview(self, state: RagState) -> str:
        if not state.history:
            return ""
        turns = state.history[-6:]
        return "\n".join(f"{role}: {content}" for role, content in turns)

    def _update_trace_route_decision(self, state: RagState) -> None:
        trace = _active_trace_var.get()
        if trace is None:
            return
        trace.route_decision.route = state.route
        trace.route_decision.rag_intent = state.rag_intent
        trace.route_decision.source_hints = list(state.source_hints)
        trace.route_decision.confidence = state.router_confidence
        trace.route_decision.reason = state.router_reason

    def _require_trace(self) -> RagDebugTrace:
        trace = _active_trace_var.get()
        if trace is None:
            raise RuntimeError("AgenticRagGraph trace is not initialized.")
        return trace
