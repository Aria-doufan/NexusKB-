import json
import uuid
from contextvars import ContextVar
from time import perf_counter
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.agent_tools import AGENTIC_RAG_TOOLS
from app.core.logger_handler import logger
from app.rag.rag_evidence_workflow import RagEvidenceWorkflow
from app.schemas.rag import (
    AgenticActionDecision,
    MemoryItem,
    RagMemoryContext,
    RagResponse,
    RagState,
    ToolExecutionResult,
)
from app.schemas.rag_debug import RagDebugTrace
from app.services import session_manager as sm
from app.services.conversation_memory import conversation_memory_service
from app.services.long_term_memory import long_term_memory_service


_active_trace_var: ContextVar[RagDebugTrace | None] = ContextVar("agentic_rag_active_trace", default=None)


class AgenticToolRunner:
    def __init__(self, tools=None):
        self.tools = {tool.name: tool for tool in (tools or AGENTIC_RAG_TOOLS)}

    async def run(self, query: str, required_tools: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for tool_name in required_tools:
            tool = self.tools.get(tool_name)
            if tool is None:
                results.append(
                    {
                        "tool_name": tool_name,
                        "tool_input": {},
                        "output": "",
                        "success": False,
                        "error": f"Tool not registered: {tool_name}",
                    }
                )
                continue
            try:
                output = await tool.ainvoke({})
            except Exception as exc:
                results.append(
                    {
                        "tool_name": tool_name,
                        "tool_input": {},
                        "output": "",
                        "success": False,
                        "error": str(exc),
                    }
                )
                continue
            results.append(
                {
                    "tool_name": tool_name,
                    "tool_input": {},
                    "output": str(output),
                    "success": True,
                    "error": None,
                }
            )
        return results


class AgenticRagGraph:
    def __init__(
        self,
        service=None,
        trace_store=None,
        decision_chain=None,
        tool_runner=None,
        rag_workflow=None,
    ):
        self.rag_workflow = rag_workflow or RagEvidenceWorkflow(service=service, trace_store=trace_store)
        self.decision_chain = decision_chain
        self.tool_runner = tool_runner or AgenticToolRunner()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RagState)
        graph.add_node("initialize", self.initialize_node)
        graph.add_node("understand_request", self.understand_request_node)
        graph.add_node("safety_check", self.safety_check_node)
        graph.add_node("direct_answer", self.direct_answer_node)
        graph.add_node("clarify", self.clarify_node)
        graph.add_node("refuse", self.refuse_node)
        graph.add_node("tool_call", self.tool_call_node)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("evaluate_context", self.evaluate_context_node)
        graph.add_node("decide_next_action", self.decide_next_action_node)
        graph.add_node("apply_retry", self.apply_retry_node)
        graph.add_node("broaden_metadata_filter", self.broaden_metadata_filter_node)
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
                "tool_call": "tool_call",
                "retrieve": "retrieve",
            },
        )
        graph.add_edge("direct_answer", "finalize_trace")
        graph.add_edge("clarify", "finalize_trace")
        graph.add_edge("refuse", "finalize_trace")
        graph.add_edge("tool_call", "finalize_trace")
        graph.add_edge("retrieve", "evaluate_context")
        graph.add_edge("evaluate_context", "decide_next_action")
        graph.add_conditional_edges(
            "decide_next_action",
            self.route_after_decide_next_action,
            {
                "apply_retry": "apply_retry",
                "broaden_metadata_filter": "broaden_metadata_filter",
                "external_search": "external_search",
                "generate_answer": "generate_answer",
                "finalize_trace": "finalize_trace",
            },
        )
        graph.add_edge("apply_retry", "retrieve")
        graph.add_edge("broaden_metadata_filter", "retrieve")
        graph.add_edge("external_search", "generate_answer")
        graph.add_edge("generate_answer", "finalize_trace")
        graph.add_edge("finalize_trace", END)
        return graph.compile()

    async def invoke(self, query: str, user_id: str, session_id: str | None = None) -> dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        state = RagState(
            request_id=str(uuid.uuid4()),
            debug_id=f"rag_dbg_{uuid.uuid4().hex}",
            session_id=session_id,
            user_id=user_id,
            original_query=query,
            current_query=query,
        )
        await self.load_context(state)
        response = await self.run(state)
        if response.answer and not state.error:
            await self.persist_message(state)
        return {
            "session_id": session_id,
            "request_id": response.request_id,
            "debug_id": response.debug_id,
            "intent": state.intent,
            "action": state.action,
            "rag_intent": state.rag_intent,
            "source_hints": state.source_hints,
            "confidence": state.router_confidence,
            "reason": state.router_reason,
            "response": response.answer,
            "sources": [source.model_dump() for source in response.sources],
            "steps": self._response_steps(state, response),
            "error": state.error,
            "sse_events": list(state.sse_events),
        }

    async def load_context(self, state: RagState) -> None:
        if not state.session_id:
            return
        try:
            memory_context = await conversation_memory_service.get_memory_context(state.session_id, state.user_id)
            state.history = memory_context.to_agent_history()
            state.memory_summary = memory_context.summary
            state.memory_compressed_turns = memory_context.compressed_turns
            state.memory_total_turns = memory_context.total_turns
        except Exception as exc:
            logger.warning(f"【AgenticRAG】加载压缩记忆失败，回退完整历史: {exc}")
            try:
                manager = sm.session_manager
                if manager is not None:
                    state.history = await manager.get_history(state.session_id, state.user_id)
                    state.memory_total_turns = len(state.history)
            except Exception as history_exc:
                logger.warning(f"【AgenticRAG】加载完整历史失败: {history_exc}")

        try:
            memories = await long_term_memory_service.search(state.original_query, state.user_id)
            state.long_term_memories = [memory.to_dict() if hasattr(memory, "to_dict") else dict(memory) for memory in memories]
            state.memory_context = self._to_rag_memory_context(state.long_term_memories)
        except Exception as exc:
            logger.warning(f"【AgenticRAG】加载长期记忆失败，继续无长期记忆上下文: {exc}")

    async def persist_message(self, state: RagState) -> None:
        if not state.session_id or not state.answer:
            return
        try:
            await conversation_memory_service.append_interaction(
                state.session_id,
                state.user_id,
                state.original_query,
                state.answer,
            )
        except AttributeError:
            manager = sm.session_manager
            if manager is not None:
                await manager.add_message(state.session_id, state.user_id, state.original_query, state.answer)
        except Exception as exc:
            logger.warning(f"【AgenticRAG】持久化消息失败: {exc}")

    @staticmethod
    def _response_steps(state: RagState, response: RagResponse) -> list[dict[str, Any]]:
        return [
            {
                "tool": "agentic_rag_graph",
                "tool_input": {
                    "query": state.original_query,
                    "intent": state.intent,
                    "action": state.action,
                    "source_hints": state.source_hints,
                },
                "tool_output": {
                    "debug_id": response.debug_id,
                    "sources": [source.model_dump() for source in response.sources],
                    "strategy": response.strategy.model_dump(),
                    "evaluation": response.evaluation.model_dump() if response.evaluation else None,
                    "metrics": response.metrics.model_dump(),
                    "response_type": state.response_type,
                    "tool_results": [result.model_dump() for result in state.tool_results],
                },
            }
        ]

    @staticmethod
    def _to_rag_memory_context(memories: list[dict[str, Any]]) -> RagMemoryContext:
        recalled = []
        for memory in memories:
            memory_type = memory.get("memory_type", "other")
            category = "user_preference" if memory_type in {"preference", "profile"} else "project_context"
            recalled.append(
                MemoryItem(
                    memory_id=str(memory.get("id") or memory.get("memory_id") or uuid.uuid4()),
                    content=str(memory.get("memory") or memory.get("content") or ""),
                    category=category,
                    relevance_score=float(memory.get("score", 0.0) or 0.0),
                    source="long_term",
                    created_at=memory.get("created_at"),
                )
            )
        return RagMemoryContext(recalled=recalled)

    async def run(self, state: RagState):
        started = perf_counter()
        started_at = self.rag_workflow._now()
        trace = self.rag_workflow.initialize_trace(state, started_at)
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
            response = await self.rag_workflow.finalize_trace(state, trace, started, started_at)
            object.__setattr__(response.strategy, "query_type", state.rag_intent)
            return response
        finally:
            _active_trace_var.reset(trace_token)

    def initialize_node(self, state: RagState) -> RagState:
        self.rag_workflow._record_event(state, "agentic_graph_initialized", "initialize")
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
        self.rag_workflow._record_event(
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

        self.rag_workflow._record_event(state, "agentic_action_routed", "safety_check", data={"action": state.action})
        return state

    def direct_answer_node(self, state: RagState) -> RagState:
        state.response_type = "answer"
        state.answer = f"这是一个通用问题，可以不检索企业知识库直接回答：{state.original_query}"
        state.sources = []
        self.rag_workflow._record_event(state, "direct_answer_created", "direct_answer")
        return state

    def clarify_node(self, state: RagState) -> RagState:
        state.response_type = "clarification"
        state.answer = f"需要再确认一下您的问题：{state.original_query}。请补充目标、范围或上下文。"
        state.sources = []
        self.rag_workflow._record_event(state, "clarification_created", "clarify")
        return state

    def refuse_node(self, state: RagState) -> RagState:
        state.response_type = "refusal"
        state.answer = "不能执行该请求，因为它可能带来安全或破坏性风险。"
        state.sources = []
        self.rag_workflow._record_event(state, "refusal_created", "refuse")
        return state

    async def tool_call_node(self, state: RagState) -> RagState:
        raw_results = await self.tool_runner.run(state.original_query, state.required_tools)
        state.tool_results = [ToolExecutionResult.model_validate(result) for result in raw_results]
        successful_outputs = [result.output for result in state.tool_results if result.success and result.output]
        state.response_type = "tool_answer"
        state.answer = "\n".join(successful_outputs) if successful_outputs else "工具调用没有返回可用结果。"
        state.sources = []
        self.rag_workflow._record_event(
            state,
            "tool_call_finished",
            "tool_call",
            data={"tools": [result.tool_name for result in state.tool_results]},
        )
        return state

    async def retrieve_node(self, state: RagState) -> RagState:
        trace = self._require_trace()
        if state.plan is None:
            self.rag_workflow.planner(state, trace)
        if self._metadata_filter_plan_needed(state):
            self.rag_workflow.metadata_filter_plan(state, trace)
        if state.strategy is None:
            self.rag_workflow.strategy_select(state, trace)
        reason = "Initial hybrid retrieval."
        if state.next_action == "broaden_metadata_filter" and state.metadata_filter_fallback_count > 0:
            reason = "Retry after broadening hard metadata filter to soft metadata boost."
        elif state.next_action in {"rewrite_query", "expand_top_k"} and state.retry_count > 0:
            reason = f"Retry after {state.next_action}."
        await self.rag_workflow.retrieve(state, trace, reason)
        return state

    def evaluate_context_node(self, state: RagState) -> RagState:
        self.rag_workflow.evaluate_context(state, self._require_trace())
        return state

    def decide_next_action_node(self, state: RagState) -> RagState:
        self.rag_workflow.decide_next_action(state)
        return state

    def apply_retry_node(self, state: RagState) -> RagState:
        if state.next_action == "rewrite_query":
            self.rag_workflow.rewrite_query(state)
        elif state.next_action == "expand_top_k":
            self.rag_workflow.expand_top_k(state)
        return state

    def broaden_metadata_filter_node(self, state: RagState) -> RagState:
        self.rag_workflow.broaden_metadata_filter(state)
        return state

    def external_search_node(self, state: RagState) -> RagState:
        self.rag_workflow._record_event(state, "external_search_skipped", "external_search")
        return state

    async def generate_answer_node(self, state: RagState) -> RagState:
        if state.next_action == "generate":
            await self.rag_workflow.generate_answer(state, self._require_trace())
        else:
            self.rag_workflow.build_insufficient_evidence(state)
        return state

    def finalize_trace_node(self, state: RagState) -> RagState:
        self.rag_workflow._record_event(state, "agentic_graph_finished", "finalize_trace")
        return state

    def route_after_safety_check(self, state: RagState) -> str:
        return state.action if state.action in {"direct_answer", "clarify", "refuse", "tool_call"} else "retrieve"

    def route_after_decide_next_action(self, state: RagState) -> str:
        if state.next_action in {"rewrite_query", "expand_top_k"} and state.retry_count < state.max_retries:
            return "apply_retry"
        if state.next_action == "broaden_metadata_filter":
            return "broaden_metadata_filter"
        if state.next_action == "external_search":
            return "external_search"
        if state.next_action == "generate":
            return "generate_answer"
        return "generate_answer"

    @staticmethod
    def _metadata_filter_plan_needed(state: RagState) -> bool:
        decision = state.metadata_filter_decision
        return (
            decision.mode == "none"
            and not decision.has_filters
            and decision.confidence == 0.0
            and state.metadata_filter_fallback_count == 0
        )

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
