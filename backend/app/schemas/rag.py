from typing import Any, Literal

from pydantic import BaseModel, Field


class SubQuery(BaseModel):
    sub_query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    reason: str = ""


class RagPlan(BaseModel):
    task_type: Literal[
        "fact_lookup",
        "semantic_query",
        "multi_hop",
        "comparison",
        "procedure",
        "constrained",
        "follow_up",
        "unknown",
    ] = "unknown"
    needs_rewrite: bool = False
    needs_decompose: bool = False
    expected_evidence_count: int = Field(default=1, ge=0)
    required_aspects: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class MemoryItem(BaseModel):
    memory_id: str = Field(min_length=1)
    content: str
    category: Literal[
        "user_preference",
        "project_context",
        "prior_question",
        "domain_term",
        "answer_style",
        "constraint",
        "irrelevant",
    ]
    relevance_score: float = Field(ge=0.0, le=1.0)
    source: Literal["conversation", "long_term", "profile"]
    created_at: str | None = None


class RagMemoryContext(BaseModel):
    recalled: list[MemoryItem] = Field(default_factory=list)
    dropped: list[MemoryItem] = Field(default_factory=list)


class RagStrategyConfig(BaseModel):
    strategy_name: str = "default"
    retrieval_mode: str = "hybrid"
    top_k_dense: int = Field(default=40, ge=0)
    top_k_bm25: int = Field(default=40, ge=0)
    fusion_top_k: int = Field(default=40, ge=0)
    final_top_k: int = Field(default=5, ge=0)
    use_reranker: bool = False
    use_query_rewrite: bool = False
    use_decompose: bool = False
    allow_expand_top_k: bool = True
    max_retries: int = Field(default=1, ge=0)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    fallback_policy: str = "insufficient_evidence"


class RagCandidate(BaseModel):
    candidate_id: str = Field(min_length=1)
    source_type: str = ""
    title: str = ""
    text: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagDocument(BaseModel):
    source_id: str = Field(min_length=1)
    parent_doc_id: str = ""
    parent_chunk_id: str = ""
    source_type: str = ""
    title: str = ""
    section_heading: str = ""
    score: float = 0.0
    text: str = ""
    child_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebSearchResult(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    provider: str = "web"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalSearchDecision(BaseModel):
    mode: Literal["none", "fallback", "hybrid"] = "none"
    allowed: bool = False
    reason: str = ""
    user_visible_label: str = ""


class RetrievalAttempt(BaseModel):
    attempt_id: int = Field(ge=1)
    query: str = Field(min_length=1)
    sub_query_id: str | None = None
    strategy_name: str = "default"
    dense_results: list[RagCandidate] = Field(default_factory=list)
    bm25_results: list[RagCandidate] = Field(default_factory=list)
    fused_results: list[RagCandidate] = Field(default_factory=list)
    reranked_results: list[RagCandidate] = Field(default_factory=list)
    selected_documents: list[RagDocument] = Field(default_factory=list)
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    dense_ms: float | None = Field(default=None, ge=0.0)
    bm25_ms: float | None = Field(default=None, ge=0.0)
    rrf_ms: float | None = Field(default=None, ge=0.0)
    rerank_ms: float | None = Field(default=None, ge=0.0)
    reason: str = ""


class EvaluationResult(BaseModel):
    enough_evidence: bool = False
    context_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_readiness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    partial_answer_allowed: bool = False
    suggested_action: Literal[
        "generate",
        "rewrite_query",
        "decompose_query",
        "expand_top_k",
        "clarify",
        "insufficient_evidence",
    ] = "insufficient_evidence"
    user_visible_reason: str = ""
    reason: str = ""


class EvaluationConfig(BaseModel):
    min_context_score: float = Field(default=0.45, ge=0.0, le=1.0)
    min_source_count: int = Field(default=1, ge=0)
    require_citation_overlap: bool = False
    max_retries: int = Field(default=1, ge=0)
    allowed_retry_actions: list[str] = Field(default_factory=lambda: ["rewrite_query", "expand_top_k"])
    evaluator_prompt_template: str = ""
    insufficient_evidence_policy: str = "transparent"


class SecurityFlag(BaseModel):
    code: str = Field(min_length=1)
    reason: str = ""
    severity: Literal["low", "medium", "high"] = "medium"


class RagSource(BaseModel):
    source_id: str = Field(min_length=1)
    title: str = ""
    source_type: str = ""
    parent_doc_id: str = ""
    parent_chunk_id: str = ""
    section_heading: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_web_result(cls, result: WebSearchResult) -> "RagSource":
        source_reference = result.url or result.title or "result"
        title = result.title or result.url or "Web reference"

        return cls(
            source_id=f"web:{source_reference}",
            title=title,
            source_type="web_reference",
            score=result.score,
            metadata={
                **result.metadata,
                "url": result.url,
                "snippet": result.snippet,
                "provider": result.provider,
                "reference_scope": "general_public_reference",
            },
        )


class RagStrategySummary(BaseModel):
    strategy_name: str
    query_type: str = "unknown"
    retrieval_mode: str
    final_top_k: int = Field(ge=0)
    use_reranker: bool = False
    use_query_rewrite: bool = False
    use_decompose: bool = False
    retry_count: int = Field(default=0, ge=0)


class EvaluationSummary(BaseModel):
    enough_evidence: bool
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    user_visible_reason: str | None = None


class RagMetrics(BaseModel):
    retry_count: int = Field(default=0, ge=0)
    retrieval_attempts: int = Field(default=0, ge=0)
    route_ms: float | None = Field(default=None, ge=0.0)
    planning_ms: float | None = Field(default=None, ge=0.0)
    strategy_ms: float | None = Field(default=None, ge=0.0)
    dense_ms: float | None = Field(default=None, ge=0.0)
    bm25_ms: float | None = Field(default=None, ge=0.0)
    rrf_ms: float | None = Field(default=None, ge=0.0)
    rerank_ms: float | None = Field(default=None, ge=0.0)
    retrieval_ms: float | None = Field(default=None, ge=0.0)
    evaluation_ms: float | None = Field(default=None, ge=0.0)
    generation_ms: float | None = Field(default=None, ge=0.0)
    total_ms: float | None = Field(default=None, ge=0.0)
    web_search_ms: float | None = Field(default=None, ge=0.0)


class RagResponse(BaseModel):
    request_id: str = Field(min_length=1)
    debug_id: str = Field(min_length=1)
    session_id: str | None = None
    answer: str
    sources: list[RagSource]
    strategy: RagStrategySummary
    evaluation: EvaluationSummary | None = None
    metrics: RagMetrics
    warnings: list[str] = Field(default_factory=list)


class RagState(BaseModel):
    request_id: str = Field(min_length=1)
    debug_id: str = Field(min_length=1)
    session_id: str | None = None
    user_id: str = Field(min_length=1)
    original_query: str = Field(min_length=1)
    current_query: str = Field(min_length=1)
    rewritten_queries: list[str] = Field(default_factory=list)
    sub_queries: list[SubQuery] = Field(default_factory=list)
    route: str = "enterprise_knowledge"
    rag_intent: str = "unknown"
    source_hints: list[str] = Field(default_factory=list)
    router_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    router_reason: str = ""
    plan: RagPlan | None = None
    memory_context: RagMemoryContext | None = None
    strategy: RagStrategyConfig | None = None
    retrieval_attempts: list[RetrievalAttempt] = Field(default_factory=list)
    selected_documents: list[RagDocument] = Field(default_factory=list)
    external_search_decision: ExternalSearchDecision = Field(default_factory=ExternalSearchDecision)
    web_results: list[WebSearchResult] = Field(default_factory=list)
    web_search_attempted: bool = False
    web_search_ms: float | None = Field(default=None, ge=0.0)
    evidence_mode: Literal["internal_only", "web_fallback", "hybrid"] = "internal_only"
    evaluator_result: EvaluationResult | None = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=1, ge=0)
    next_action: str | None = None
    security_flags: list[SecurityFlag] = Field(default_factory=list)
    acl_filter_removed_all_candidates: bool = False
    answer: str | None = None
    sources: list[RagSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    sse_events: list[dict[str, Any]] = Field(default_factory=list)
