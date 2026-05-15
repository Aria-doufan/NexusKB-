from typing import List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from app.core.logger_handler import logger
from app.core.perf import log_perf, perf_counter
from app.db.db_config import AsyncSessionLocal
from app.models.chat_history import ChatSession, ChatMessage, ChatSessionMemory
from app.utils.factory import chat_model


RECENT_WINDOW_TURNS = 6
SUMMARY_TRIGGER_TURNS = 10


class MemoryContext(BaseModel):
    summary: str = ""
    recent_history: List[Tuple[str, str]] = Field(default_factory=list)
    compressed_turns: int = 0
    total_turns: int = 0

    def to_agent_history(self) -> List[Tuple[str, str]]:
        history: List[Tuple[str, str]] = []
        if self.summary:
            history.append(
                (
                    "以下是本会话较早历史的摘要，请作为上下文参考。",
                    self.summary,
                )
            )
        history.extend(self.recent_history)
        return history


class ConversationMemoryService:
    """Two-layer memory: rolling session summary plus recent raw turns."""

    def __init__(
        self,
        recent_window_turns: int = RECENT_WINDOW_TURNS,
        summary_trigger_turns: int = SUMMARY_TRIGGER_TURNS,
    ):
        self.recent_window_turns = recent_window_turns
        self.summary_trigger_turns = summary_trigger_turns
        self.summary_chain = self._build_summary_chain()

    def _build_summary_chain(self):
        prompt = PromptTemplate.from_template(
            """请将以下历史对话压缩为长期会话记忆。

你需要保留：
1. 用户明确表达过的目标、偏好、约束
2. 已确认的项目背景和技术选择
3. 已完成和未完成事项
4. 重要结论、接口、文件路径、数据来源
5. 后续回答必须记住的上下文

你需要删除：
1. 寒暄
2. 重复确认
3. 临时错误信息
4. 无关细节

已有会话摘要：
{existing_summary}

需要新增压缩的历史：
{new_history}

请输出更新后的会话摘要，保持简洁、结构清晰。"""
        )
        return prompt | chat_model | StrOutputParser()

    async def get_memory_context(self, session_id: str, user_id: str) -> MemoryContext:
        op_start = perf_counter()
        history = await self._get_history(session_id, user_id)
        memory = await self._get_memory(session_id, user_id)
        total_turns = len(history)

        if total_turns <= self.summary_trigger_turns:
            context = MemoryContext(
                summary=memory.summary if memory else "",
                recent_history=history,
                compressed_turns=memory.summarized_turn_count if memory else 0,
                total_turns=total_turns,
            )
            log_perf(
                "memory.context_get",
                op_start,
                session_id=session_id,
                user_id=user_id,
                total_turns=total_turns,
                compressed_turns=context.compressed_turns,
            )
            return context

        memory = await self.update_memory(session_id, user_id, history=history)
        recent_history = history[-self.recent_window_turns :]
        context = MemoryContext(
            summary=memory.summary,
            recent_history=recent_history,
            compressed_turns=memory.summarized_turn_count,
            total_turns=total_turns,
        )
        log_perf(
            "memory.context_get",
            op_start,
            session_id=session_id,
            user_id=user_id,
            total_turns=total_turns,
            compressed_turns=context.compressed_turns,
        )
        return context

    async def get_history_for_agent(self, session_id: str, user_id: str) -> List[Tuple[str, str]]:
        memory_context = await self.get_memory_context(session_id, user_id)
        return memory_context.to_agent_history()

    async def update_memory(
        self,
        session_id: str,
        user_id: str,
        history: List[Tuple[str, str]] | None = None,
    ) -> ChatSessionMemory:
        op_start = perf_counter()
        history = history if history is not None else await self._get_history(session_id, user_id)
        total_turns = len(history)
        memory = await self._get_or_create_memory(session_id, user_id)

        if total_turns <= self.summary_trigger_turns:
            log_perf("memory.update", op_start, session_id=session_id, user_id=user_id, total_turns=total_turns, summarized=False)
            return memory

        target_summarized_turns = max(total_turns - self.recent_window_turns, 0)
        if target_summarized_turns <= memory.summarized_turn_count:
            log_perf("memory.update", op_start, session_id=session_id, user_id=user_id, total_turns=total_turns, summarized=False)
            return memory

        new_history = history[memory.summarized_turn_count : target_summarized_turns]
        if not new_history:
            log_perf("memory.update", op_start, session_id=session_id, user_id=user_id, total_turns=total_turns, summarized=False)
            return memory

        step_start = perf_counter()
        new_summary = await self._summarize(memory.summary, new_history)
        log_perf("memory.summary_chain", step_start, session_id=session_id, user_id=user_id, turns=len(new_history))
        memory = await self._save_memory(
            session_id=session_id,
            user_id=user_id,
            summary=new_summary,
            summarized_turn_count=target_summarized_turns,
        )
        log_perf("memory.update", op_start, session_id=session_id, user_id=user_id, total_turns=total_turns, summarized=True)
        return memory

    async def _get_history(self, session_id: str, user_id: str) -> List[Tuple[str, str]]:
        op_start = perf_counter()
        async with AsyncSessionLocal() as db:
            chat_session = await db.run_sync(
                lambda session: session.query(ChatSession)
                .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
                .first()
            )
            if not chat_session:
                log_perf("mysql.memory_history_get", op_start, session_id=session_id, user_id=user_id, history_turns=0)
                return []

            messages = await db.run_sync(
                lambda session: session.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at, ChatMessage.id)
                .all()
            )

        history: List[Tuple[str, str]] = []
        i = 0
        while i < len(messages):
            if (
                messages[i].role == "user"
                and i + 1 < len(messages)
                and messages[i + 1].role == "assistant"
            ):
                history.append((messages[i].content, messages[i + 1].content))
                i += 2
            else:
                i += 1
        log_perf("mysql.memory_history_get", op_start, session_id=session_id, user_id=user_id, history_turns=len(history))
        return history

    async def _get_memory(self, session_id: str, user_id: str) -> ChatSessionMemory | None:
        op_start = perf_counter()
        async with AsyncSessionLocal() as db:
            memory = await db.run_sync(
                lambda session: session.query(ChatSessionMemory)
                .filter(
                    ChatSessionMemory.session_id == session_id,
                    ChatSessionMemory.user_id == user_id,
                )
                .first()
            )
            log_perf("mysql.memory_get", op_start, session_id=session_id, user_id=user_id, found=memory is not None)
            return memory

    async def _get_or_create_memory(self, session_id: str, user_id: str) -> ChatSessionMemory:
        memory = await self._get_memory(session_id, user_id)
        if memory:
            return memory

        async with AsyncSessionLocal() as db:
            op_start = perf_counter()
            memory = ChatSessionMemory(
                session_id=session_id,
                user_id=user_id,
                summary="",
                summarized_turn_count=0,
            )
            db.add(memory)
            await db.commit()
            await db.refresh(memory)
            log_perf("mysql.memory_create", op_start, session_id=session_id, user_id=user_id)
            logger.info(f"【会话记忆】创建会话记忆: session_id={session_id}, user_id={user_id}")
            return memory

    async def _save_memory(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        summarized_turn_count: int,
    ) -> ChatSessionMemory:
        op_start = perf_counter()
        async with AsyncSessionLocal() as db:
            memory = await db.run_sync(
                lambda session: session.query(ChatSessionMemory)
                .filter(
                    ChatSessionMemory.session_id == session_id,
                    ChatSessionMemory.user_id == user_id,
                )
                .first()
            )
            if memory is None:
                memory = ChatSessionMemory(session_id=session_id, user_id=user_id)
                db.add(memory)

            memory.summary = summary
            memory.summarized_turn_count = summarized_turn_count
            await db.commit()
            await db.refresh(memory)
            log_perf(
                "mysql.memory_save",
                op_start,
                session_id=session_id,
                user_id=user_id,
                summary_chars=len(summary or ""),
                summarized_turn_count=summarized_turn_count,
            )
            logger.info(
                "【会话记忆】更新摘要 session_id=%s summarized_turn_count=%s",
                session_id,
                summarized_turn_count,
            )
            return memory

    async def _summarize(self, existing_summary: str, new_history: List[Tuple[str, str]]) -> str:
        formatted_history = self._format_history(new_history)
        try:
            return await self.summary_chain.ainvoke(
                {
                    "existing_summary": existing_summary or "无",
                    "new_history": formatted_history,
                }
            )
        except Exception as exc:
            logger.error(f"【会话记忆】生成摘要失败，使用兜底拼接: {exc}", exc_info=True)
            return self._fallback_summary(existing_summary, new_history)

    @staticmethod
    def _format_history(history: List[Tuple[str, str]]) -> str:
        blocks: list[str] = []
        for index, (user_message, assistant_message) in enumerate(history, start=1):
            blocks.append(
                f"【第{index}轮】\n用户：{user_message}\n助手：{assistant_message}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _fallback_summary(existing_summary: str, new_history: List[Tuple[str, str]]) -> str:
        lines: list[str] = []
        if existing_summary:
            lines.append(existing_summary.strip())
        lines.append("新增历史摘要：")
        for user_message, assistant_message in new_history:
            lines.append(f"- 用户询问：{user_message[:120]}；助手回答：{assistant_message[:160]}")
        return "\n".join(lines).strip()


conversation_memory_service = ConversationMemoryService()
