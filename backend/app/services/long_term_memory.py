"""Vector-backed long-term memory service.

This module intentionally avoids import-time LLM, embedding, and Chroma
initialization. External clients are created lazily by the methods that need
those integrations so unit tests can import the module without credentials or
running services.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_

from app.core.logger_handler import logger
from app.db.db_config import AsyncSessionLocal
from app.models.chat_history import LongTermMemory

MEMORY_COLLECTION_NAME = "long_term_memories"
DEFAULT_LIMIT = 5
MIN_RELEVANCE_SCORE = float(os.getenv("LONG_TERM_MEMORY_MIN_RELEVANCE", "0.35"))
DUPLICATE_RELEVANCE_SCORE = float(os.getenv("LONG_TERM_MEMORY_DUPLICATE_RELEVANCE", "0.88"))
EXTRACTION_MIN_SCORE = float(os.getenv("LONG_TERM_MEMORY_EXTRACTION_MIN_SCORE", "0.55"))

ALLOWED_MEMORY_TYPES: set[str] = {
    "preference",
    "profile",
    "project",
    "fact",
    "instruction",
    "relationship",
    "constraint",
    "other",
}
NO_MEMORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*$"),
    re.compile(r"^(hi|hello|hey|你好|您好|thanks|thank you|谢谢)[!.。！\s]*$", re.IGNORECASE),
    re.compile(r"^(ok|okay|好的|收到|明白|嗯|是的|不是)[!.。！\s]*$", re.IGNORECASE),
)


@dataclass(slots=True)
class LongTermMemoryItem:
    id: str
    user_id: str
    session_id: str
    memory: str
    memory_type: str
    source: str = "chat"
    source_message_ids: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    status: str = "active"

    @classmethod
    def from_model(cls, model: LongTermMemory) -> "LongTermMemoryItem":
        return cls(
            id=str(model.id),
            user_id=str(model.user_id),
            session_id=str(model.session_id),
            memory=str(model.memory),
            memory_type=str(model.memory_type),
            source=str(model.source or "chat"),
            source_message_ids=list(model.source_message_ids or []),
            metadata=dict(model.metadata_ or {}),
            score=float(model.score or 0.0),
            status=str(model.status or "active"),
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
    def __init__(self, vector_store: Any | None = None):
        self._vector_store = vector_store
        self._extraction_chain: Any | None = None

    @property
    def vector_store(self) -> Any | None:
        return self._vector_store

    @vector_store.setter
    def vector_store(self, value: Any | None) -> None:
        self._vector_store = value

    async def extract_and_store(
        self,
        *,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str = "",
        source_message_ids: list[Any] | None = None,
        source: str = "chat",
    ) -> list[LongTermMemoryItem]:
        text = "\n".join(part for part in [user_message, assistant_message] if part)
        if self._should_skip_extraction(text):
            return []

        chain = self._get_extraction_chain()
        raw_output = await chain.ainvoke(
            {
                "user_message": user_message,
                "assistant_message": assistant_message,
            }
        )
        parsed = self._parse_json(raw_output)
        memories_data = parsed if isinstance(parsed, list) else parsed.get("memories", [])
        if not isinstance(memories_data, list):
            return []

        stored: list[LongTermMemoryItem] = []
        async with AsyncSessionLocal() as db:
            for entry in memories_data:
                if not isinstance(entry, dict):
                    continue
                memory = str(entry.get("memory") or "").strip()
                if self._should_skip_extraction(memory):
                    continue
                score = self._normalize_score(entry.get("score", 1.0))
                if score < EXTRACTION_MIN_SCORE:
                    continue
                if await self._is_semantic_duplicate(memory, user_id):
                    continue

                memory_type = self._normalize_memory_type(entry.get("memory_type"))
                memory_id = str(uuid.uuid4())
                metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
                model = LongTermMemory(
                    id=memory_id,
                    user_id=user_id,
                    session_id=session_id,
                    memory=memory,
                    memory_type=memory_type,
                    source=source,
                    source_message_ids=source_message_ids or [],
                    hash=self._hash_memory(memory),
                    metadata_=metadata,
                    score=score,
                    status="active",
                )
                db.add(model)
                item = LongTermMemoryItem.from_model(model)
                stored.append(item)
            await db.commit()

        if stored:
            await self._add_to_vector_store(stored)
        return stored

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = DEFAULT_LIMIT,
        min_relevance: float = MIN_RELEVANCE_SCORE,
    ) -> list[LongTermMemoryItem]:
        if not query or not user_id:
            return []

        vector_store = self._get_vector_store()
        if vector_store is not None:
            try:
                matches = await asyncio.to_thread(
                    vector_store.similarity_search_with_relevance_scores,
                    query,
                    k=limit,
                    filter={"user_id": user_id, "status": "active"},
                )
                memory_ids: list[str] = []
                vector_scores: dict[str, float] = {}
                for document, score in matches:
                    normalized_score = self._normalize_score(score)
                    if normalized_score < min_relevance:
                        continue
                    metadata = getattr(document, "metadata", {}) or {}
                    memory_id = metadata.get("memory_id") or metadata.get("id")
                    if memory_id:
                        memory_ids.append(str(memory_id))
                        vector_scores[str(memory_id)] = normalized_score
                if memory_ids:
                    items = await self._get_active_by_ids(memory_ids, user_id)
                    if items:
                        for item in items:
                            if item.id in vector_scores:
                                item.score = vector_scores[item.id]
                        return items[:limit]
            except Exception as exc:
                logger.warning(f"【长期记忆】向量检索失败，回退关键词检索: {exc}")

        return await self._fallback_search(query, user_id, limit)

    async def list_memories(self, user_id: str, limit: int = 50, offset: int = 0) -> list[LongTermMemoryItem]:
        async with AsyncSessionLocal() as db:
            result = await db.run_sync(
                lambda session: session.query(LongTermMemory)
                .filter(LongTermMemory.user_id == user_id, LongTermMemory.status == "active")
                .order_by(LongTermMemory.updated_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [LongTermMemoryItem.from_model(model) for model in result]

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        async with AsyncSessionLocal() as db:
            model = await db.run_sync(
                lambda session: session.query(LongTermMemory)
                .filter(
                    LongTermMemory.id == memory_id,
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.status == "active",
                )
                .first()
            )
            if model is None:
                return False
            model.status = "deleted"
            model.deleted_at = datetime.now(timezone.utc)
            await db.commit()

        vector_store = self._get_vector_store()
        if vector_store is not None:
            try:
                await asyncio.to_thread(vector_store.delete, ids=[memory_id])
            except Exception as exc:
                logger.warning(f"【长期记忆】删除向量记忆失败，已保留数据库删除状态: {exc}")
        return True

    async def _is_semantic_duplicate(self, memory: str, user_id: str) -> bool:
        vector_store = self._get_vector_store()
        if vector_store is None:
            return await self._has_active_hash_duplicate(memory, user_id)

        try:
            matches = await asyncio.to_thread(
                vector_store.similarity_search_with_relevance_scores,
                memory,
                k=1,
                filter={"user_id": user_id, "status": "active"},
            )
        except Exception as exc:
            logger.warning(f"【长期记忆】语义去重失败，回退哈希去重: {exc}")
            return await self._has_active_hash_duplicate(memory, user_id)

        duplicate_ids: list[str] = []
        for document, score in matches:
            if self._normalize_score(score) < DUPLICATE_RELEVANCE_SCORE:
                continue
            metadata = getattr(document, "metadata", {}) or {}
            memory_id = metadata.get("memory_id") or metadata.get("id")
            if memory_id:
                duplicate_ids.append(str(memory_id))

        if not duplicate_ids:
            return False

        active_matches = await self._get_active_by_ids(duplicate_ids, user_id)
        if active_matches:
            return True

        return await self._has_active_hash_duplicate(memory, user_id)

    async def _has_active_hash_duplicate(self, memory: str, user_id: str) -> bool:
        memory_hash = self._hash_memory(memory)
        async with AsyncSessionLocal() as db:
            existing = await db.run_sync(
                lambda session: session.query(LongTermMemory)
                .filter(
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.status == "active",
                    LongTermMemory.hash == memory_hash,
                )
                .first()
            )
            return existing is not None

    async def _get_active_by_ids(self, memory_ids: list[str], user_id: str) -> list[LongTermMemoryItem]:
        if not memory_ids:
            return []
        id_order = {memory_id: index for index, memory_id in enumerate(memory_ids)}
        async with AsyncSessionLocal() as db:
            result = await db.run_sync(
                lambda session: session.query(LongTermMemory)
                .filter(
                    LongTermMemory.id.in_(memory_ids),
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.status == "active",
                )
                .all()
            )
        items = [LongTermMemoryItem.from_model(model) for model in result]
        return sorted(items, key=lambda item: id_order.get(item.id, len(id_order)))

    async def _fallback_search(self, query: str, user_id: str, limit: int) -> list[LongTermMemoryItem]:
        keywords = self._keywords(query)
        async with AsyncSessionLocal() as db:
            def _query(session: Any) -> list[LongTermMemory]:
                base = session.query(LongTermMemory).filter(
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.status == "active",
                )
                if keywords:
                    base = base.filter(or_(*(LongTermMemory.memory.ilike(f"%{keyword}%") for keyword in keywords)))
                return base.order_by(LongTermMemory.updated_at.desc()).limit(limit).all()

            result = await db.run_sync(_query)
        return [LongTermMemoryItem.from_model(model) for model in result]

    def _get_vector_store(self) -> Any | None:
        if self._vector_store is not None:
            return self._vector_store
        try:
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings
        except Exception as exc:
            logger.warning(f"【长期记忆】向量库依赖不可用: {exc}")
            return None

        try:
            persist_directory = os.getenv("LONG_TERM_MEMORY_CHROMA_DIR", "./chroma_langchain_db")
            embedding_model = os.getenv("LONG_TERM_MEMORY_EMBEDDING_MODEL", "nomic-embed-text")
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            self._vector_store = Chroma(
                collection_name=MEMORY_COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )
        except Exception as exc:
            logger.warning(f"【长期记忆】向量库初始化失败: {exc}")
            self._vector_store = None
        return self._vector_store

    def _get_extraction_chain(self) -> Any:
        if self._extraction_chain is not None:
            return self._extraction_chain

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            streaming=False,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract durable user memories from the conversation. Return JSON only with a memories array. "
                    "Each item must contain memory, memory_type, score, and optional metadata. "
                    f"memory_type must be one of: {', '.join(sorted(ALLOWED_MEMORY_TYPES))}.",
                ),
                (
                    "human",
                    "User message:\n{user_message}\n\nAssistant message:\n{assistant_message}",
                ),
            ]
        )
        self._extraction_chain = prompt | model | StrOutputParser()
        return self._extraction_chain

    async def _add_to_vector_store(self, items: list[LongTermMemoryItem]) -> None:
        vector_store = self._get_vector_store()
        if vector_store is None or not items:
            return
        try:
            from langchain_core.documents import Document

            documents = [
                Document(
                    page_content=item.memory,
                    metadata={
                        **item.metadata,
                        "memory_id": item.id,
                        "user_id": item.user_id,
                        "session_id": item.session_id,
                        "memory_type": item.memory_type,
                        "status": item.status,
                    },
                )
                for item in items
            ]
            await asyncio.to_thread(vector_store.add_documents, documents=documents, ids=[item.id for item in items])
        except Exception as exc:
            logger.warning(f"【长期记忆】写入向量库失败: {exc}")

    @staticmethod
    def _hash_memory(memory: str) -> str:
        normalized = " ".join((memory or "").strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _keywords(text: str) -> list[str]:
        tokens = re.findall(r"[\w一-鿿]+", text or "", flags=re.UNICODE)
        return [token for token in tokens if len(token) >= 2][:8]

    @staticmethod
    def _normalize_memory_type(value: Any) -> str:
        memory_type = str(value or "other").strip().lower()
        return memory_type if memory_type in ALLOWED_MEMORY_TYPES else "other"

    @staticmethod
    def _normalize_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _should_skip_extraction(text: str) -> bool:
        cleaned = str(text or "").strip()
        if len(cleaned) < 4:
            return True
        return any(pattern.match(cleaned) for pattern in NO_MEMORY_PATTERNS)

    @staticmethod
    def _parse_json(raw_output: Any) -> Any:
        if isinstance(raw_output, (dict, list)):
            return raw_output
        cleaned = str(raw_output or "").strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        return json.loads(cleaned)


long_term_memory_service = LongTermMemoryService()
