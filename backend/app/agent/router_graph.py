import json
import os
import re
import uuid
from typing import Any, AsyncGenerator, Literal, NotRequired, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from app.agent.agent import get_agent_response, get_agent_stream_response, get_chat_response, get_chat_stream_response
from app.core.logger_handler import logger
from app.core.perf import log_perf, perf_counter
from app.rag.enterprise_rag_service import enterprise_rag_service
from app.services import session_manager as sm
from app.services.conversation_memory import conversation_memory_service


Route = Literal["enterprise_knowledge", "tool_action", "chat", "unsafe_or_system", "clarify"]
RagIntent = Literal[
    "basic",
    "semantic",
    "intra_document_reasoning",
    "project_related",
    "constrained",
    "conflicting_info",
    "completeness",
    "high_level",
    "info_not_found",
    "unknown",
]
SourceHint = Literal[
    "confluence",
    "jira",
    "slack",
    "github",
    "google_drive",
    "linear",
    "gmail",
    "hubspot",
    "fireflies",
]

ALLOWED_ROUTES: set[str] = {"enterprise_knowledge", "tool_action", "chat", "unsafe_or_system", "clarify"}
LEGACY_ROUTE_ALIASES: dict[str, str] = {
    "rag_query": "enterprise_knowledge",
    "agent_tool_call": "tool_action",
    "system": "unsafe_or_system",
}
ALLOWED_RAG_INTENTS: set[str] = {
    "basic",
    "semantic",
    "intra_document_reasoning",
    "project_related",
    "constrained",
    "conflicting_info",
    "completeness",
    "high_level",
    "info_not_found",
    "unknown",
}
ALLOWED_SOURCE_HINTS: set[str] = {
    "confluence",
    "jira",
    "slack",
    "github",
    "google_drive",
    "linear",
    "gmail",
    "hubspot",
    "fireflies",
}
LOW_CONFIDENCE_THRESHOLD = 0.45


class RouteDecision(BaseModel):
    route: str = Field(description="Top-level route for the request.")
    rag_intent: str = Field(default="unknown", description="RAG sub intent.")
    source_hints: list[str] = Field(default_factory=list, description="Likely source types.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0 to 1.")
    reason: str = Field(default="", description="Brief reason for the route decision.")


class GraphState(TypedDict):
    query: str
    user_id: str
    session_id: str
    history: NotRequired[list[tuple[str, str]]]
    memory_summary: NotRequired[str]
    memory_compressed_turns: NotRequired[int]
    memory_total_turns: NotRequired[int]
    route: NotRequired[str]
    rag_intent: NotRequired[str]
    source_hints: NotRequired[list[str]]
    confidence: NotRequired[float]
    reason: NotRequired[str]
    answer: NotRequired[str]
    documents: NotRequired[list[Any]]
    steps: NotRequired[list[dict[str, Any]]]
    error: NotRequired[str | None]


ROUTER_SYSTEM_PROMPT = """你是一个请求路由器，只负责判断用户请求应该进入哪个系统链路。

你不能回答用户问题，只能返回一个 JSON 对象。

顶层 route 只能选择一个：
- enterprise_knowledge：需要企业知识库、上传文档、内部资料、制度、历史记录、项目细节来回答。
- tool_action：需要调用企业工具、内部/外部 API、查询系统状态、工单状态、用户信息、重排序等受控工具。
- chat：普通对话、开放问答、解释概念、写作改写总结，或只需要安全小工具（例如当前时间、天气）的低风险问题。
- unsafe_or_system：删除、清空、重置、越权、危险系统操作，或需要保守确认的系统类请求。
- clarify：用户意图不明确，或缺少必要条件，需要反问。

特别注意：
- 用户问当前时间、日期、天气时，route 选择 chat，因为 chat 链路会显式处理 safe utility。
- 用户问企业内部项目、制度、上传限制、历史记录时，route 选择 enterprise_knowledge。
- 简单闲聊、通用概念解释、写作、改写、总结、翻译等不依赖企业内部资料的问题，route 选择 chat，不要查企业知识库。
- 用户要求查询系统状态、工单状态、调用内部 API 时，route 选择 tool_action。
- 用户要求删除、清空、重置、越权操作时，route 选择 unsafe_or_system。

当 route 为 enterprise_knowledge 时，rag_intent 只能选择一个：
basic, semantic, intra_document_reasoning, project_related, constrained,
conflicting_info, completeness, high_level, info_not_found, unknown。

当 route 不是 enterprise_knowledge 时，rag_intent 必须是 unknown。

source_hints 只能从以下来源中选择，不能编造：
confluence, jira, slack, github, google_drive, linear, gmail, hubspot, fireflies。

只返回 JSON，不要返回 Markdown，不要返回解释文本。
"""

ROUTER_HUMAN_PROMPT = """用户问题：
{query}

最近会话历史：
{history_preview}

请返回如下 JSON：
{{
  "route": "enterprise_knowledge | tool_action | chat | unsafe_or_system | clarify",
  "rag_intent": "basic | semantic | intra_document_reasoning | project_related | constrained | conflicting_info | completeness | high_level | info_not_found | unknown",
  "source_hints": ["confluence"],
  "confidence": 0.0,
  "reason": "一句话说明"
}}
"""


class RouterGraph:
    """LangGraph-based router that delegates to existing RAG and Agent chains."""

    def __init__(self):
        self.router_model = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            streaming=False,
            temperature=0,
        )
        self.router_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", ROUTER_SYSTEM_PROMPT),
                    ("human", ROUTER_HUMAN_PROMPT),
                ]
            )
            | self.router_model
            | StrOutputParser()
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("load_context", self.load_context)
        graph.add_node("llm_router", self.llm_router)
        graph.add_node("validate_decision", self.validate_decision)
        graph.add_node("enterprise_knowledge_node", self.enterprise_knowledge_node)
        graph.add_node("tool_action_node", self.tool_action_node)
        graph.add_node("chat_node", self.chat_node)
        graph.add_node("unsafe_or_system_node", self.unsafe_or_system_node)
        graph.add_node("clarify_node", self.clarify_node)
        graph.add_node("persist_message", self.persist_message)
        graph.add_node("format_response", self.format_response)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "llm_router")
        graph.add_edge("llm_router", "validate_decision")
        graph.add_conditional_edges(
            "validate_decision",
            self.select_route,
            {
                "enterprise_knowledge": "enterprise_knowledge_node",
                "tool_action": "tool_action_node",
                "chat": "chat_node",
                "unsafe_or_system": "unsafe_or_system_node",
                "clarify": "clarify_node",
            },
        )
        graph.add_edge("enterprise_knowledge_node", "persist_message")
        graph.add_edge("tool_action_node", "persist_message")
        graph.add_edge("chat_node", "persist_message")
        graph.add_edge("unsafe_or_system_node", "persist_message")
        graph.add_edge("clarify_node", "persist_message")
        graph.add_edge("persist_message", "format_response")
        graph.add_edge("format_response", END)
        return graph.compile()

    async def invoke(self, query: str, user_id: str, session_id: str | None = None) -> dict[str, Any]:
        invoke_start = perf_counter()
        state: GraphState = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id or str(uuid.uuid4()),
            "history": [],
            "memory_summary": "",
            "memory_compressed_turns": 0,
            "memory_total_turns": 0,
            "route": "chat",
            "rag_intent": "unknown",
            "source_hints": [],
            "confidence": 0.0,
            "reason": "",
            "answer": "",
            "documents": [],
            "steps": [],
            "error": None,
        }
        result = await self.graph.ainvoke(state)
        log_perf(
            "router.invoke_total",
            invoke_start,
            session_id=result.get("session_id"),
            user_id=user_id,
            route=result.get("route", "chat"),
        )
        return {
            "session_id": result.get("session_id"),
            "route": result.get("route", "chat"),
            "rag_intent": result.get("rag_intent", "unknown"),
            "source_hints": result.get("source_hints", []),
            "confidence": result.get("confidence", 0.0),
            "reason": result.get("reason", ""),
            "response": result.get("answer", ""),
            "steps": result.get("steps", []),
            "error": result.get("error"),
        }

    async def stream(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        stream_start = perf_counter()
        session_id = session_id or str(uuid.uuid4())
        state: GraphState = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "history": [],
            "memory_summary": "",
            "memory_compressed_turns": 0,
            "memory_total_turns": 0,
            "route": "chat",
            "rag_intent": "unknown",
            "source_hints": [],
            "confidence": 0.0,
            "reason": "",
            "answer": "",
            "documents": [],
            "steps": [],
            "error": None,
        }

        try:
            step_start = perf_counter()
            state.update(await self.load_context(state))
            log_perf("router.load_context", step_start, session_id=session_id, user_id=user_id)

            step_start = perf_counter()
            state.update(await self.llm_router(state))
            log_perf("router.llm_decision", step_start, session_id=session_id, user_id=user_id)

            step_start = perf_counter()
            state.update(await self.validate_decision(state))
            log_perf(
                "router.validate_decision",
                step_start,
                session_id=session_id,
                user_id=user_id,
                route=state.get("route"),
            )

            route = self.select_route(state)
            log_perf(
                "router.to_route_event",
                stream_start,
                session_id=session_id,
                user_id=user_id,
                route=route,
            )
            yield self._sse_event(
                {
                    "type": "route",
                    "session_id": state["session_id"],
                    "route": route,
                    "rag_intent": state.get("rag_intent", "unknown"),
                    "source_hints": state.get("source_hints", []),
                    "confidence": state.get("confidence", 0.0),
                    "reason": state.get("reason", ""),
                }
            )

            if route == "chat":
                try:
                    async for event in get_chat_stream_response(
                        query,
                        state["session_id"],
                        user_id,
                        history=state.get("history", []),
                    ):
                        yield event
                finally:
                    log_perf("router.stream_total", stream_start, session_id=session_id, user_id=user_id, route=route)
                return

            if route == "tool_action":
                try:
                    async for event in get_agent_stream_response(
                        query,
                        state["session_id"],
                        user_id,
                        tool_profile="full",
                        history=state.get("history", []),
                    ):
                        yield event
                finally:
                    log_perf("router.stream_total", stream_start, session_id=session_id, user_id=user_id, route=route)
                return

            if route == "enterprise_knowledge":
                state.update(await self.enterprise_knowledge_node(state))
            elif route == "unsafe_or_system":
                state.update(await self.unsafe_or_system_node(state))
            elif route == "clarify":
                state.update(await self.clarify_node(state))
            else:
                state.update(await self.chat_node(state))

            answer = state.get("answer") or "抱歉，我无法理解您的请求。"
            if answer and not state.get("error"):
                await self.persist_message(state)

            yield self._sse_event(
                {
                    "type": "response",
                    "content": answer,
                    "session_id": state["session_id"],
                }
            )
            yield self._sse_event({"type": "done", "session_id": state["session_id"]})
            log_perf("router.stream_total", stream_start, session_id=session_id, user_id=user_id, route=route)
        except Exception as exc:
            logger.error(f"【RouterGraph】流式路由执行失败: {exc}", exc_info=True)
            yield self._sse_event(
                {
                    "type": "error",
                    "content": f"错误: {str(exc)}",
                    "session_id": session_id,
                }
            )
            yield self._sse_event({"type": "done", "session_id": session_id})

    async def load_context(self, state: GraphState) -> dict[str, Any]:
        session_id = state.get("session_id") or str(uuid.uuid4())
        user_id = state["user_id"]
        history: list[tuple[str, str]] = []
        memory_summary = ""
        memory_compressed_turns = 0
        memory_total_turns = 0

        try:
            memory_context = await conversation_memory_service.get_memory_context(session_id, user_id)
            history = memory_context.to_agent_history()
            memory_summary = memory_context.summary
            memory_compressed_turns = memory_context.compressed_turns
            memory_total_turns = memory_context.total_turns
        except Exception as exc:
            logger.warning(f"【RouterGraph】加载压缩记忆失败，回退完整历史: {exc}")
            try:
                if sm.session_manager is not None:
                    history = await sm.session_manager.get_history(session_id, user_id)
                    memory_total_turns = len(history)
            except Exception as history_exc:
                logger.warning(f"【RouterGraph】加载完整历史失败: {history_exc}")

        return {
            "session_id": session_id,
            "history": history,
            "memory_summary": memory_summary,
            "memory_compressed_turns": memory_compressed_turns,
            "memory_total_turns": memory_total_turns,
        }

    async def llm_router(self, state: GraphState) -> dict[str, Any]:
        try:
            raw_output = await self.router_chain.ainvoke(
                {
                    "query": state["query"],
                    "history_preview": self._format_history_preview(state.get("history", [])),
                }
            )
            decision_data = self._parse_json(raw_output)
            decision = RouteDecision.model_validate(decision_data)
            logger.info(
                "【RouterGraph】LLM路由决策 route=%s rag_intent=%s confidence=%.2f reason=%s",
                decision.route,
                decision.rag_intent,
                decision.confidence,
                decision.reason,
            )
            return decision.model_dump()
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning(f"【RouterGraph】LLM路由输出解析失败，回退到chat: {exc}")
            return {
                "route": "chat",
                "rag_intent": "unknown",
                "source_hints": [],
                "confidence": 0.0,
                "reason": "Router LLM 输出无法解析，回退到普通对话。",
            }
        except Exception as exc:
            logger.error(f"【RouterGraph】LLM路由调用失败，回退到chat: {exc}", exc_info=True)
            return {
                "route": "chat",
                "rag_intent": "unknown",
                "source_hints": [],
                "confidence": 0.0,
                "reason": "Router LLM 调用失败，回退到普通对话。",
            }

    async def validate_decision(self, state: GraphState) -> dict[str, Any]:
        route = self._normalize_route(state.get("route", "chat"))
        rag_intent = state.get("rag_intent", "unknown")
        confidence = self._normalize_confidence(state.get("confidence", 0.0))
        source_hints = [
            source for source in state.get("source_hints", []) if source in ALLOWED_SOURCE_HINTS
        ]

        if route not in ALLOWED_ROUTES:
            route = "chat"
        if rag_intent not in ALLOWED_RAG_INTENTS:
            rag_intent = "unknown"
        if route != "enterprise_knowledge":
            rag_intent = "unknown"
            source_hints = []
        if confidence < LOW_CONFIDENCE_THRESHOLD and route not in {"chat", "clarify"}:
            route = "clarify"
            rag_intent = "unknown"
            source_hints = []

        return {
            "route": route,
            "rag_intent": rag_intent,
            "source_hints": source_hints,
            "confidence": confidence,
            "reason": state.get("reason", ""),
        }

    def select_route(self, state: GraphState) -> str:
        route = self._normalize_route(state.get("route", "chat"))
        return route if route in ALLOWED_ROUTES else "chat"

    async def enterprise_knowledge_node(self, state: GraphState) -> dict[str, Any]:
        try:
            result = await enterprise_rag_service.get_documents_and_summary(
                query=state["query"],
                rag_intent=state.get("rag_intent", "unknown"),
                source_hints=state.get("source_hints", []),
                router_confidence=state.get("confidence", 0.0),
            )
            answer = result.get("summary", "抱歉，处理企业知识库请求时出现了错误。")
            documents = result.get("documents", [])
            strategy = result.get("strategy", {})
            steps = [
                {
                    "tool": "enterprise_knowledge",
                    "tool_input": {
                        "query": state["query"],
                        "rag_intent": state.get("rag_intent", "unknown"),
                        "source_hints": state.get("source_hints", []),
                        "router_confidence": state.get("confidence", 0.0),
                    },
                    "tool_output": {
                        "retrieved_documents": len(documents),
                        "strategy": strategy,
                    },
                }
            ]
            return {"answer": answer, "documents": documents, "steps": steps}
        except Exception as exc:
            logger.error(f"【RouterGraph】企业知识库节点执行失败: {exc}", exc_info=True)
            return {"answer": "抱歉，知识库检索时出现了错误。", "error": f"enterprise_knowledge_error: {exc}"}

    async def tool_action_node(self, state: GraphState) -> dict[str, Any]:
        return await self._run_agent_node(
            state,
            fallback_answer="抱歉，工具调用时出现了错误。",
            tool_profile="full",
        )

    async def chat_node(self, state: GraphState) -> dict[str, Any]:
        try:
            result = await get_chat_response(state["query"], state.get("history", []))
            return {
                "answer": result.get("response", "抱歉，处理对话时出现了错误。"),
                "steps": result.get("steps", []),
            }
        except Exception as exc:
            logger.error(f"【RouterGraph】Pure Chat节点执行失败: {exc}", exc_info=True)
            return {"answer": "抱歉，处理对话时出现了错误。", "error": f"chat_error: {exc}"}

    async def unsafe_or_system_node(self, state: GraphState) -> dict[str, Any]:
        answer = (
            "这是一个系统类请求。为避免误操作，当前 Router 只识别系统意图，"
            "不会直接执行删除、清空、重置等危险操作。请明确确认具体操作后再继续。"
        )
        return {"answer": answer, "steps": []}

    async def clarify_node(self, state: GraphState) -> dict[str, Any]:
        answer = "我需要再确认一下：你希望我基于企业知识库检索、执行工具动作，还是进行普通对话来处理这个问题？"
        return {"answer": answer, "steps": []}

    async def persist_message(self, state: GraphState) -> dict[str, Any]:
        answer = state.get("answer", "")
        if not answer or state.get("error"):
            return {}

        try:
            if sm.session_manager is not None:
                await sm.session_manager.add_message(
                    state["session_id"],
                    state["user_id"],
                    state["query"],
                    answer,
                )
                await conversation_memory_service.update_memory(
                    state["session_id"],
                    state["user_id"],
                )
        except Exception as exc:
            logger.warning(f"【RouterGraph】写入会话历史失败: {exc}")
        return {}

    async def format_response(self, state: GraphState) -> dict[str, Any]:
        return state

    async def _run_agent_node(
        self,
        state: GraphState,
        fallback_answer: str,
        tool_profile: str,
    ) -> dict[str, Any]:
        try:
            result = await get_agent_response(
                state["query"],
                state.get("history", []),
                tool_profile=tool_profile,
            )
            return {
                "answer": result.get("response", fallback_answer),
                "steps": result.get("steps", []),
            }
        except Exception as exc:
            logger.error(f"【RouterGraph】Agent节点执行失败: {exc}", exc_info=True)
            return {"answer": fallback_answer, "error": f"agent_error: {exc}"}

    @staticmethod
    def _format_history_preview(history: list[tuple[str, str]], max_turns: int = 3) -> str:
        if not history:
            return "无"

        preview_lines: list[str] = []
        for user_msg, assistant_msg in history[-max_turns:]:
            preview_lines.append(f"用户：{user_msg[:200]}")
            preview_lines.append(f"助手：{assistant_msg[:200]}")
        return "\n".join(preview_lines)

    @staticmethod
    def _parse_json(raw_output: str) -> dict[str, Any]:
        cleaned = raw_output.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        return json.loads(cleaned)

    @staticmethod
    def _normalize_route(route: Any) -> str:
        route_name = str(route or "chat").strip()
        return LEGACY_ROUTE_ALIASES.get(route_name, route_name)

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def _sse_event(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


router_graph = RouterGraph()
