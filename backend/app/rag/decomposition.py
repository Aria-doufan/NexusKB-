from __future__ import annotations

import json
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError

from app.core.logger_handler import logger

SubQueryPurpose = Literal["fact", "comparison_dimension", "procedure_step", "constraint", "verification"]

DECOMPOSE_RAG_INTENTS = {"multi_hop", "comparison"}
MAX_SUB_QUERIES = 4
MIN_SUB_QUERIES = 2

DECOMPOSITION_PROMPT = PromptTemplate.from_template(
    """你是企业知识库检索规划器。你的任务是把复杂问题拆成可独立检索的子问题。

规则：
1. 只生成 2 到 4 个子问题。
2. 每个子问题必须能直接拿去检索企业知识库。
3. 不要回答问题，不要补充原始问题没有给出的事实。
4. 保留原始问题中的实体名、制度名、项目名、时间范围、来源约束。
5. comparison 问题按比较对象或比较维度拆分。
6. multi_hop 问题按前置事实、后续事实、综合判断拆分。
7. 如果无法可靠拆分，返回 {{"sub_queries": []}}。

用户问题：
{query}

最近会话摘要：
{history_context}

只返回 JSON，格式如下：
{{
  "sub_queries": [
    {{"id": "sq1", "query": "子问题文本", "purpose": "fact", "depends_on": []}}
  ]
}}
"""
)


class SubQuery(BaseModel):
    id: str
    query: str
    purpose: SubQueryPurpose
    depends_on: list[str] = Field(default_factory=list)


class SubQueryPlan(BaseModel):
    original_query: str
    sub_queries: list[SubQuery] = Field(default_factory=list)
    fallback_reason: str | None = None

    def query_texts(self) -> list[str]:
        return [sub_query.query for sub_query in self.sub_queries]


class SubQueryRetrievalResult(BaseModel):
    sub_query_id: str
    sub_query: str
    dense_ids: list[str] = Field(default_factory=list)
    bm25_ids: list[str] = Field(default_factory=list)
    fused_ids: list[str] = Field(default_factory=list)
    elapsed_ms: float | None = None


class DecomposedCandidate(BaseModel):
    parent_chunk_id: str
    matched_sub_query_ids: list[str] = Field(default_factory=list)
    fused_score: float = 0.0
    coverage_score: float = 0.0
    final_score: float = 0.0


def merge_decomposed_scores(
    sub_query_rankings: dict[str, dict[str, float]],
    total_sub_queries: int,
    coverage_weight: float = 0.25,
) -> dict[str, DecomposedCandidate]:
    merged: dict[str, DecomposedCandidate] = {}
    safe_total = max(total_sub_queries, 1)

    for sub_query_id, ranked_scores in sub_query_rankings.items():
        for parent_chunk_id, score in ranked_scores.items():
            candidate = merged.setdefault(
                parent_chunk_id,
                DecomposedCandidate(parent_chunk_id=parent_chunk_id),
            )
            if sub_query_id not in candidate.matched_sub_query_ids:
                candidate.matched_sub_query_ids.append(sub_query_id)
            candidate.fused_score = max(candidate.fused_score, score)

    for candidate in merged.values():
        candidate.coverage_score = len(candidate.matched_sub_query_ids) / safe_total
        candidate.final_score = candidate.fused_score * (1.0 + coverage_weight * candidate.coverage_score)

    return merged


def should_decompose_intent(rag_intent: str) -> bool:
    return rag_intent in DECOMPOSE_RAG_INTENTS


def build_fallback_plan(original_query: str, reason: str) -> SubQueryPlan:
    return SubQueryPlan(original_query=original_query, sub_queries=[], fallback_reason=reason)


def parse_sub_query_plan(raw_text: str, original_query: str) -> SubQueryPlan:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return build_fallback_plan(original_query, "invalid_json")

    raw_sub_queries = payload.get("sub_queries") if isinstance(payload, dict) else None
    if not isinstance(raw_sub_queries, list):
        return build_fallback_plan(original_query, "invalid_json_shape")

    if len(raw_sub_queries) < MIN_SUB_QUERIES or len(raw_sub_queries) > MAX_SUB_QUERIES:
        return build_fallback_plan(original_query, "invalid_sub_query_count")

    try:
        sub_queries = [SubQuery.model_validate(item) for item in raw_sub_queries]
    except ValidationError:
        return build_fallback_plan(original_query, "invalid_sub_query_content")

    normalized_queries = [item.query.strip() for item in sub_queries]
    if any(not query for query in normalized_queries):
        return build_fallback_plan(original_query, "invalid_sub_query_content")
    if len(set(normalized_queries)) != len(normalized_queries):
        return build_fallback_plan(original_query, "invalid_sub_query_content")

    normalized_ids = [item.id.strip() for item in sub_queries]
    if any(not item_id for item_id in normalized_ids):
        return build_fallback_plan(original_query, "invalid_sub_query_content")
    if len(set(normalized_ids)) != len(normalized_ids):
        return build_fallback_plan(original_query, "invalid_sub_query_content")

    declared_ids = set(normalized_ids)
    normalized_sub_queries: list[SubQuery] = []
    for item, item_id, query in zip(sub_queries, normalized_ids, normalized_queries):
        depends_on = [dependency.strip() for dependency in item.depends_on]
        if any(not dependency or dependency not in declared_ids for dependency in depends_on):
            return build_fallback_plan(original_query, "invalid_sub_query_content")
        normalized_sub_queries.append(
            SubQuery(id=item_id, query=query, purpose=item.purpose, depends_on=depends_on)
        )

    return SubQueryPlan(original_query=original_query, sub_queries=normalized_sub_queries)


async def decompose_query(query: str, history_context: str = "") -> SubQueryPlan:
    from app.utils.factory import chat_model

    chain = DECOMPOSITION_PROMPT | chat_model | StrOutputParser()
    try:
        raw_text = await chain.ainvoke({"query": query, "history_context": history_context})
    except Exception as exc:
        logger.warning(f"【EnterpriseRAG】子问题拆解失败: {exc}")
        return build_fallback_plan(query, "decomposition_chain_error")

    return parse_sub_query_plan(raw_text, original_query=query)
