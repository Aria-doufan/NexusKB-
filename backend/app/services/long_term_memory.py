import asyncio
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sqlalchemy import func

from app.core.logger_handler import logger
from app.core.perf import log_perf, perf_counter
from app.db.db_config import AsyncSessionLocal
from app.models.chat_history import LongTermMemory
from app.utils.config import chroma_config
from app.utils.factory import chat_model, embed_model
from app.utils.path_tool import get_abstract_path


MEMORY_COLLECTION_NAME = "long_term_memories"
DEFAULT_LIMIT = 8
MIN_RELEVANCE_SCORE = float(os.getenv("LONG_TERM_MEMORY_MIN_RELEVANCE", "0.35"))
DUPLICATE_RELEVANCE_SCORE = float(os.getenv("LONG_TERM_MEMORY_DUPLICATE_RELEVANCE", "0.92"))
ALLOWED_MEMORY_TYPES = {
    "preference",
    "profile",
    "project",
    "decision",
    "task",
    "assistant_output",
    "other",
}
NO_MEMORY_PATTERNS = (
    "不要记",
    "别记",
    "不要保存",
    "别保存",
    "不要写入记忆",
    "do not remember",
    "don't remember",
    "do not save",
)


@dataclass(slots=True)
class LongTermMemoryItem:
    id: str
    user_id: str
    session_id: str
    memory: str
    memory_type: str
    source: str
    source_message_ids: list[int]
    metadata: dict[str, Any]
    score: float
    status: str

    @classmethod
    def from_model(cls, memory: LongTermMemory) -> "LongTermMemoryItem":
        return cls(
            id=memory.id,
            user_id=memory.user_id,
            session_id=memory.session_id,
            memory=memory.memory,
            memory_type=memory.memory_type,
            source=memory.source,
            source_message_ids=list(memory.source_message_ids or []),
            metadata=dict(memory.metadata_ or {}),
            score=float(memory.score or 0.0),
            status=memory.status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "memory": self.memory,
            "memory_type": self.memory_type,
            "source": self.source,
            "source_message_ids": self.source_message_ids,
            "metadata": self.metadata,
            "score": self.score,
            "status": self.status,
        }


class LongTermMemoryService:
    """ADD-only fact-level long-term memory service."""

    def __init__(self) -> None:
        self.extract_chain = self._build_extract_chain()
        self.vector_store = self._build_vector_store()

    def _build_extract_chain(self):
        prompt = PromptTemplate.from_template(
            """你是长期记忆抽取器。请从一轮用户-助手对话中抽取后续可能复用的事实级记忆。

只返回 JSON，不要返回 Markdown，不要解释。

抽取规则：
1. 只抽取后续可能复用的信息，例如用户长期偏好、项目背景、持续任务、重要决策、接口约定、文件路径。
2. 不抽取寒暄、一次性问题、临时语气、无意义确认。
3. 不把“用户问了什么”当记忆，要抽取问题中隐含的事实。
4. 如果用户明确说不要记、不要保存、别记，返回空数组。
5. 不要推断未经用户确认的敏感个人属性。
6. 默认用用户原语言记录，保留专有名词、文件名、接口名、日期、数量。

memory_type 只能是：
preference, profile, project, decision, task, assistant_output, other

用户消息：
{user_message}

助手回复：
{assistant_message}

返回格式：
{{
  "memories": [
    {{
      "memory": "一条独立事实",
      "memory_type": "project",
      "reason": "为什么值得长期保存",
      "confidence": 0.86
    }}
  ]
}}
"""
        )
        return prompt | chat_model | StrOutputParser()

    @staticmethod
    def _build_vector_store() -> Chroma | None:
        try:
            persist_dir = get_abstract_path(chroma_config["persist_directory"])
            return Chroma(
                collection_name=MEMORY_COLLECTION_NAME,
                embedding_function=embed_model,
                persist_directory=persist_dir,
            )
        except Exception as exc:
            logger.warning(f"【长期记忆】初始化向量库失败，将仅使用 MySQL: {exc}")
            return None

    async def extract_and_store(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        source_message_ids: list[int] | None = None,
        source: str = "chat",
    ) -> list[LongTermMemoryItem]:
        op_start = perf_counter()
        if self._should_skip_extraction(user_message):
            return []

        try:
            candidates = await self._extract_candidates(user_message, assistant_message)
        except Exception as exc:
            logger.warning(f"【长期记忆】抽取失败，跳过本轮长期记忆写入: {exc}")
            return []

        stored: list[LongTermMemoryItem] = []
        seen_hashes: set[str] = set()
        for candidate in candidates:
            memory_text = str(candidate.get("memory") or "").strip()
            if not memory_text:
                continue

            memory_hash = self._hash_memory(memory_text)
            if memory_hash in seen_hashes:
                continue
            seen_hashes.add(memory_hash)

            if await self._hash_exists(user_id, memory_hash):
                continue
            if await self._is_semantic_duplicate(memory_text, user_id):
                continue

            item = await self._save_memory(
                user_id=user_id,
                session_id=session_id,
                memory=memory_text,
                memory_type=self._normalize_memory_type(candidate.get("memory_type")),
                source=source,
                source_message_ids=source_message_ids or [],
                memory_hash=memory_hash,
                metadata={
                    "reason": candidate.get("reason", ""),
                    "confidence": self._normalize_score(candidate.get("confidence")),
                },
                score=self._normalize_score(candidate.get("confidence")),
            )
            stored.append(item)
            await self._add_to_vector_store(item)

        log_perf(
            "long_term_memory.extract_and_store",
            op_start,
            session_id=session_id,
            user_id=user_id,
            extracted=len(candidates),
            stored=len(stored),
        )
        return stored

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = DEFAULT_LIMIT,
        min_relevance: float = MIN_RELEVANCE_SCORE,
    ) -> list[LongTermMemoryItem]:
        op_start = perf_counter()
        limit = max(1, min(limit, 20))
        items: list[LongTermMemoryItem] = []

        if self.vector_store is not None:
            try:
                docs_and_scores = await asyncio.to_thread(
                    self.vector_store.similarity_search_with_relevance_scores,
                    query,
                    k=limit,
                    filter=self._active_user_filter(user_id),
                )
                memory_ids = [
                    str(doc.metadata.get("memory_id"))
                    for doc, score in docs_and_scores
                    if doc.metadata.get("memory_id") and float(score) >= min_relevance
                ]
                items = await self._get_active_by_ids(memory_ids, user_id)
            except Exception as exc:
                logger.warning(f"【长期记忆】向量检索失败，回退 MySQL 模糊检索: {exc}")

        if not items:
            items = await self._fallback_search(query, user_id, limit)

        log_perf("long_term_memory.search", op_start, user_id=user_id, returned=len(items))
        return items[:limit]

    async def list_memories(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[LongTermMemoryItem]:
        limit = max(1, min(limit, 100))
        offset = max(0, offset)
        async with AsyncSessionLocal() as db:
            memories = await db.run_sync(
                lambda session: session.query(LongTermMemory)
                .filter(LongTermMemory.user_id == user_id, LongTermMemory.status == "active")
                .order_by(LongTermMemory.created_at.desc(), LongTermMemory.id.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
        return [LongTermMemoryItem.from_model(memory) for memory in memories]

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        op_start = perf_counter()
        async with AsyncSessionLocal() as db:
            memory = await db.run_sync(
                lambda session: session.query(LongTermMemory)
                .filter(
                    LongTermMemory.id == memory_id,
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.status == "active",
                )
                .first()
            )
            if memory is None or memory.status != "active":
                return False
            memory.status = "deleted"
            memory.deleted_at = func.now()
            await db.commit()

        if self.vector_store is not None:
            try:
                await asyncio.to_thread(self.vector_store.delete, ids=[memory_id])
            except Exception as exc:
                logger.warning(f"【长期记忆】删除向量记录失败 memory_id={memory_id}: {exc}")

        log_perf("long_term_memory.delete", op_start, user_id=user_id, memory_id=memory_id)
        return True

    async def _extract_candidates(self, user_message: str, assistant_message: str) -> list[dict[str, Any]]:
        raw_output = await self.extract_chain.ainvoke(
            {
                "user_message": user_message,
                "assistant_message": assistant_message[:4000],
            }
        )
        data = self._parse_json(raw_output)
        memories = data.get("memories", [])
        if not isinstance(memories, list):
            return []
        return [memory for memory in memories if isinstance(memory, dict)]

    async def _save_memory(
        self,
        user_id: str,
        session_id: str,
        memory: str,
        memory_type: str,
        source: str,
        source_message_ids: list[int],
        memory_hash: str,
        metadata: dict[str, Any],
        score: float,
    ) -> LongTermMemoryItem:
        async with AsyncSessionLocal() as db:
            model = LongTermMemory(
                id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                memory=memory,
                memory_type=memory_type,
                source=source,
                source_message_ids=source_message_ids,
                hash=memory_hash,
                metadata_=metadata,
                score=score,
                status="active",
            )
            db.add(model)
            await db.commit()
            await db.refresh(model)
            logger.info("【长期记忆】新增记忆 user_id=%s memory_id=%s", user_id, model.id)
            return LongTermMemoryItem.from_model(model)

    async def _add_to_vector_store(self, item: LongTermMemoryItem) -> None:
        if self.vector_store is None:
            return

        document = Document(
            page_content=item.memory,
            metadata={
                **item.metadata,
                "memory_id": item.id,
                "user_id": item.user_id,
                "session_id": item.session_id,
                "memory_type": item.memory_type,
                "status": item.status,
                "source": item.source,
            },
        )
        try:
            await asyncio.to_thread(self.vector_store.add_documents, [document], ids=[item.id])
        except Exception as exc:
            logger.warning(f"【长期记忆】写入向量库失败 memory_id={item.id}: {exc}")

    async def _hash_exists(self, user_id: str, memory_hash: str) -> bool:
        async with AsyncSessionLocal() as db:
            existing = await db.run_sync(
                lambda session: session.query(LongTermMemory.id)
                .filter(
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.hash == memory_hash,
                    LongTermMemory.status == "active",
                )
                .first()
            )
        return existing is not None

    async def _is_semantic_duplicate(self, memory: str, user_id: str) -> bool:
        if self.vector_store is None:
            return False
        try:
            matches = await asyncio.to_thread(
                self.vector_store.similarity_search_with_relevance_scores,
                memory,
                k=1,
                filter=self._active_user_filter(user_id),
            )
            return bool(matches and float(matches[0][1]) >= DUPLICATE_RELEVANCE_SCORE)
        except Exception as exc:
            logger.warning(f"【长期记忆】相似度去重失败，继续使用 hash 去重: {exc}")
            return False

    async def _get_active_by_ids(self, memory_ids: list[str], user_id: str) -> list[LongTermMemoryItem]:
        if not memory_ids:
            return []
        unique_ids = list(dict.fromkeys(memory_ids))
        async with AsyncSessionLocal() as db:
            rows = await db.run_sync(
                lambda session: session.query(LongTermMemory)
                .filter(
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.status == "active",
                    LongTermMemory.id.in_(unique_ids),
                )
                .all()
            )
        by_id = {row.id: LongTermMemoryItem.from_model(row) for row in rows}
        return [by_id[memory_id] for memory_id in unique_ids if memory_id in by_id]

    async def _fallback_search(self, query: str, user_id: str, limit: int) -> list[LongTermMemoryItem]:
        keywords = self._keywords(query)
        if not keywords:
            return []

        async with AsyncSessionLocal() as db:
            if keywords:
                rows = await db.run_sync(
                    lambda session: session.query(LongTermMemory)
                    .filter(LongTermMemory.user_id == user_id, LongTermMemory.status == "active")
                    .filter(*[LongTermMemory.memory.ilike(f"%{keyword}%") for keyword in keywords[:3]])
                    .order_by(LongTermMemory.created_at.desc(), LongTermMemory.id.desc())
                    .limit(limit)
                    .all()
                )
            else:
                rows = await db.run_sync(
                    lambda session: session.query(LongTermMemory)
                    .filter(LongTermMemory.user_id == user_id, LongTermMemory.status == "active")
                    .order_by(LongTermMemory.created_at.desc(), LongTermMemory.id.desc())
                    .limit(limit)
                    .all()
                )
        return [LongTermMemoryItem.from_model(row) for row in rows]

    @staticmethod
    def _active_user_filter(user_id: str) -> dict[str, Any]:
        return {"$and": [{"user_id": user_id}, {"status": "active"}]}

    @staticmethod
    def _parse_json(raw_output: str) -> dict[str, Any]:
        cleaned = raw_output.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        return json.loads(cleaned)

    @staticmethod
    def _normalize_memory_type(value: Any) -> str:
        memory_type = str(value or "other").strip()
        return memory_type if memory_type in ALLOWED_MEMORY_TYPES else "other"

    @staticmethod
    def _normalize_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _hash_memory(memory: str) -> str:
        normalized = " ".join(memory.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _should_skip_extraction(user_message: str) -> bool:
        normalized = user_message.lower()
        return any(pattern in normalized for pattern in NO_MEMORY_PATTERNS)

    @staticmethod
    def _keywords(query: str) -> list[str]:
        return [
            token
            for token in re.split(r"[\s,，。！？；;:：、/\\|()\[\]{}<>《》\"']+", query.strip())
            if len(token) >= 2
        ]


long_term_memory_service = LongTermMemoryService()
