from pydantic import BaseModel, Field

from app.schemas.rag import EvaluationResult, RagPlan, RagSource, RagStrategyConfig, RetrievalAttempt


class RouteDecisionTrace(BaseModel):
    route: str
    rag_intent: str = "unknown"
    source_hints: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


class PlannerTrace(BaseModel):
    plan: RagPlan
    elapsed_ms: float = 0.0


class MemoryRecallTrace(BaseModel):
    recalled_count: int = 0
    dropped_count: int = 0
    recalled: list[dict] = Field(default_factory=list)
    dropped: list[dict] = Field(default_factory=list)


class RagStrategyTrace(BaseModel):
    strategy: RagStrategyConfig
    reason: str = ""


class RetrievalAttemptTrace(BaseModel):
    attempt: RetrievalAttempt


class EvaluationTrace(BaseModel):
    result: EvaluationResult


class GenerationTrace(BaseModel):
    answer_preview: str = ""
    elapsed_ms: float = 0.0


class RagDebugTrace(BaseModel):
    request_id: str = Field(min_length=1)
    debug_id: str = Field(min_length=1)
    session_id: str | None = None
    user_id: str = Field(min_length=1)
    route_decision: RouteDecisionTrace | None = None
    planner: PlannerTrace | None = None
    memory_recall: MemoryRecallTrace | None = None
    strategy: RagStrategyTrace | None = None
    retrieval_attempts: list[RetrievalAttemptTrace] = Field(default_factory=list)
    evaluations: list[EvaluationTrace] = Field(default_factory=list)
    generation: GenerationTrace | None = None
    final_answer_preview: str | None = None
    final_sources: list[RagSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    reproduced_from_debug_id: str | None = None
    started_at: str
    finished_at: str | None = None
    total_ms: float | None = None
