from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from app.schemas.rag import MetadataFilterDecision, RagCandidate, RagDocument, RagStrategyConfig, RetrievalAttempt


@dataclass(slots=True)
class RetrievalStageMetrics:
    dense_ms: float | None = None
    bm25_ms: float | None = None
    rrf_ms: float | None = None
    rerank_ms: float | None = None
    total_ms: float | None = None


@dataclass(slots=True)
class RetrievalPipelineResult:
    selected_documents: list[RagDocument]
    attempt: RetrievalAttempt
    metrics: RetrievalStageMetrics
    raw: dict[str, Any] = field(default_factory=dict)


class RetrievalPipeline:
    def __init__(self, service):
        self.service = service

    async def run(
        self,
        query: str,
        strategy: RagStrategyConfig,
        source_hints: list[str] | None,
        *,
        metadata_filter: MetadataFilterDecision | None = None,
        rag_intent: str,
        router_confidence: float,
        attempt_id: int,
        sub_query_id: str | None = None,
        reason: str = "Initial retrieval.",
    ) -> RetrievalPipelineResult:
        started = perf_counter()
        retrieval_kwargs = {
            "query": query,
            "final_top_k": strategy.final_top_k,
            "dense_top_k": strategy.top_k_dense,
            "bm25_top_k": strategy.top_k_bm25,
            "fusion_top_k": strategy.fusion_top_k,
            "source_hints": source_hints,
            "use_reranker": strategy.use_reranker,
        }
        if metadata_filter and (metadata_filter.mode != "none" or metadata_filter.has_filters):
            retrieval_kwargs["metadata_filter"] = metadata_filter
        raw_result = await self.service.retrieve_with_details(**retrieval_kwargs)
        selected_documents = [self._to_rag_document(document) for document in raw_result["selected_documents"]]
        attempt = RetrievalAttempt(
            attempt_id=attempt_id,
            query=query,
            sub_query_id=sub_query_id,
            strategy_name=strategy.strategy_name,
            metadata_filter=metadata_filter or MetadataFilterDecision(),
            dense_results=[self._to_candidate(item) for item in raw_result.get("dense_results", [])],
            bm25_results=[self._to_candidate(item) for item in raw_result.get("bm25_results", [])],
            fused_results=[self._to_candidate(item) for item in raw_result.get("fused_results", [])],
            reranked_results=[self._to_candidate(item) for item in raw_result.get("reranked_results", [])],
            selected_documents=selected_documents,
            elapsed_ms=(perf_counter() - started) * 1000,
            dense_ms=raw_result.get("metrics", {}).get("dense_ms"),
            bm25_ms=raw_result.get("metrics", {}).get("bm25_ms"),
            rrf_ms=raw_result.get("metrics", {}).get("rrf_ms"),
            rerank_ms=raw_result.get("metrics", {}).get("rerank_ms"),
            reason=reason,
        )
        metrics = RetrievalStageMetrics(
            dense_ms=attempt.dense_ms,
            bm25_ms=attempt.bm25_ms,
            rrf_ms=attempt.rrf_ms,
            rerank_ms=attempt.rerank_ms,
            total_ms=attempt.elapsed_ms,
        )
        return RetrievalPipelineResult(
            selected_documents=selected_documents,
            attempt=attempt,
            metrics=metrics,
            raw=raw_result,
        )

    def _to_candidate(self, item: Any) -> RagCandidate:
        data = item if isinstance(item, dict) else item.to_dict()
        return RagCandidate(
            candidate_id=data.get("parent_chunk_id") or data.get("chunk_id") or data.get("parent_doc_id") or data.get("title") or "candidate",
            source_type=data.get("source_type", ""),
            title=data.get("title", ""),
            text=data.get("child_text") or data.get("parent_text") or data.get("text") or "",
            score=float(data.get("score", 0.0) or 0.0),
            metadata=dict(data.get("metadata") or {}),
        )

    def _to_rag_document(self, item: Any) -> RagDocument:
        data = item if isinstance(item, dict) else item.to_dict()
        parent_chunk_id = data.get("parent_chunk_id", "")
        parent_doc_id = data.get("parent_doc_id", "")
        metadata = dict(data.get("metadata") or {})
        return RagDocument(
            source_id=parent_chunk_id or parent_doc_id or data.get("title", "source"),
            parent_doc_id=parent_doc_id,
            parent_chunk_id=parent_chunk_id,
            source_type=data.get("source_type", metadata.get("source_type", "")),
            title=data.get("title", metadata.get("title", "")),
            section_heading=data.get("section_heading", metadata.get("section_heading", "")),
            score=float(data.get("score", 0.0) or 0.0),
            text=data.get("parent_text") or data.get("text") or "",
            child_text=data.get("child_text", ""),
            metadata=metadata,
        )
