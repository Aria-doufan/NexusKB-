# RAGFlow Agentic RAG 架构设计

整理日期：2026-05-16

## 1. 文档定位

本文记录 RAGFlow Agentic RAG 的工程架构、核心链路、统一 Response Schema、SSE Event Schema、前端兼容策略和 Agentic 策略矩阵。执行阶段、优先级和当前状态见 [执行路线图](./RAGFLOW_AGENTIC_RAG_ROADMAP.md)。评测指标和阈值见 [评测方案](./RAGFLOW_AGENTIC_RAG_EVALUATION.md)。安全边界见 [安全边界](./RAGFLOW_AGENTIC_RAG_SECURITY.md)。

## 2. 架构原则

本项目的核心原则是：RAG Pipeline 是主角，Agentic Router 是策略控制器，评测系统是策略优劣判断标准。

```text
Agent / LangGraph = 查询理解与策略路由
RAG Pipeline = 检索、融合、重排、上下文构建、引用生成
Evaluation = 判断策略是否真正提升效果
Security = 企业知识边界与引用边界
```

Agent 不应该自由决定所有动作，而应该输出受控枚举和可验证配置。RAG 服务根据策略配置执行检索链路，并把策略、来源、指标和调试信息写入统一响应。

## 3. 在线主链路

当前在线链路已经收敛为 `RouterGraph` 兼容入口 + `AgenticRagGraph` 主状态机 + `RagEvidenceWorkflow` 证据工作流：

```text
用户问题
  -> FastAPI /api/agent/query/stream
  -> Auth / RateLimit / RequestId
  -> RouterGraph.invoke / stream 兼容包装
  -> AgenticRagGraph.load_context
       -> conversation memory recent window + rolling summary
       -> long_term_memory_service.search
  -> AgenticRagGraph StateGraph
       -> initialize
       -> understand_request: action / rag_intent / source_hints / confidence
       -> safety_check: direct_answer / retrieve / tool_call / clarify / refuse
       -> retrieve: 委托 RagEvidenceWorkflow
  -> RagEvidenceWorkflow
       -> planner
       -> strategy_select
       -> RetrievalPipeline.run
            -> EnterpriseRagService.retrieve_with_details
            -> Dense Retrieval
            -> BM25 Retrieval
            -> Source Boost
            -> RRF Fusion
            -> Reranker，可选
       -> evaluate_context
       -> rewrite_query / expand_top_k retry，可选
       -> decompose for multi_hop / comparison，可选
       -> external search fallback，可选
       -> LLM Answer Generation 或 insufficient evidence
  -> RagResponse(answer, sources, strategy, evaluation, metrics, debug_id)
  -> SSE Events
  -> Save Session / Debug Trace
```

## 4. 核心模块边界

在线主链路优先看 `mainline` 行：`AgenticRagGraph` 负责在线 LangGraph 状态机和动作路由，`RagEvidenceWorkflow` 负责证据工作流，`RetrievalPipeline` 负责检索归一化和编排。`adapter` 仅保留兼容入口，`knowledge source adapter` 接入不同知识来源，`legacy` 与 `experimental` 不应被误认为在线主线。

| 分类 | 文件 | 在线角色 |
| --- | --- | --- |
| mainline | `backend/app/rag/agentic_rag_graph.py` | 在线 LangGraph state machine；拥有 action routing 和 context loading。 |
| mainline | `backend/app/rag/rag_evidence_workflow.py` | Evidence workflow；拥有 planning、strategy、retrieval、evaluation、retry/fallback、generation、trace finalization。 |
| mainline | `backend/app/rag/retrieval_pipeline.py` | Retrieval normalization/orchestration；应演进为统一 source-aware evidence retrieval。 |
| mainline | `backend/app/rag/strategy_router.py` | Strategy matrix for `rag_intent`、confidence、reranker、decompose、retry choices。 |
| mainline | `backend/app/schemas/rag.py` | Main RAG state、strategy、source、metrics、response models。 |
| mainline | `backend/app/schemas/rag_debug.py` | Debug trace schema for observing evidence workflow。 |
| mainline | `backend/app/services/conversation_memory.py` | Conversation summary 和 recent-history context。 |
| mainline | `backend/app/services/long_term_memory.py` | Long-term memory recall for Agentic RAG context。 |
| adapter | `backend/app/agent/router_graph.py` | Compatibility adapter around `AgenticRagGraph` for existing API response fields and SSE event shape。 |
| adapter | `backend/app/rag/enterprise_rag_graph.py` | Compatibility adapter around `RagEvidenceWorkflow` for tests/evaluation scripts。 |
| knowledge source adapter | `backend/app/rag/enterprise_rag_service.py` | Enterprise corpus retrieval source adapter behind `RetrievalPipeline`。 |
| knowledge source adapter | `backend/app/rag/rag_service.py` | Uploaded-document retrieval backend / candidate supplemental source；target source-specific backend, not currently wired into online evidence path。 |
| knowledge source adapter | `backend/app/rag/vector_store.py` | Chroma vector storage for uploaded documents。 |
| legacy | `backend/app/agent/agent.py` | Legacy LangChain Agent and PureChat utilities。 |
| legacy | `backend/app/agent/agent_middleware.py` | Legacy Agent middleware tied to legacy Agent usage。 |
| legacy | `backend/app/agent/agent_tools.py` | Mixed legacy full-agent tools plus Agentic RAG tool subset；clarified in place。 |
| experimental | `backend/app/rag/graph_extraction.py` | Graph RAG extraction experiment。 |
| experimental | `backend/app/rag/graph_index_service.py` | Graph RAG indexing experiment。 |
| experimental | `backend/scripts/index_enterprise_graph.py` | Offline graph index script。 |

`backend/app/rag/retrieval_backends/*` contains enterprise retrieval backend abstraction; Chroma remains default while Elasticsearch is evaluation-ready candidate.

上传文档应被理解为统一 evidence layer 方向下的补充知识来源，而不是与 Agentic RAG 并行竞争的另一条产品线。当前状态下 `RagService` 是上传文档检索后端 / 候选补充来源，目标方向是作为 source-specific backend 接入统一 source-aware evidence retrieval；除非后续代码接线完成，不应表述为已进入在线主证据链路。

## 5. Pydantic Response Schema

建议新增或集中维护以下 schema。路径建议为：

```text
backend/app/schemas/rag.py
```

### 5.1 RagSource

```python
from typing import Any, Literal
from pydantic import BaseModel, Field

class RagSource(BaseModel):
    doc_id: str = Field(..., description="文档 ID")
    chunk_id: str = Field(..., description="检索命中的 child chunk ID")
    parent_id: str | None = Field(None, description="回填上下文对应的 parent chunk ID")
    knowledge_base_id: str | None = Field(None, description="知识库 ID")
    owner_user_id: str | None = Field(None, description="文档归属用户，默认不返回给前端，可用于 debug")
    source_type: str | None = Field(None, description="来源类型，如 upload、enterprise_bench、confluence、jira")
    title: str | None = Field(None, description="文档标题")
    section: str | None = Field(None, description="章节标题")
    page: int | None = Field(None, description="页码")
    snippet: str = Field(..., description="可展示引用片段")
    score: float | None = Field(None, description="最终相关性分数")
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

前端默认只展示 `title`、`section`、`page`、`snippet`、`score`。`owner_user_id`、内部路径、完整 metadata 默认不展示，避免泄露。

### 5.2 RagStrategy

```python
class RagStrategy(BaseModel):
    route: Literal["chat", "enterprise_knowledge", "tool_action", "clarify", "unsafe_or_system"]
    rag_intent: str = "unknown"
    strategy_name: str = "default"
    retrieval_mode: Literal[
        "none",
        "dense_only",
        "bm25_only",
        "dense_bm25_rrf",
        "dense_bm25_rrf_reranker"
    ] = "dense_bm25_rrf"
    top_k_dense: int = 20
    top_k_bm25: int = 20
    fusion_top_k: int = 20
    final_top_k: int = 5
    use_hyde: bool = False
    use_query_rewrite: bool = False
    use_decompose: bool = False
    use_reranker: bool = False
    reranker_candidate_k: int = 10
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    fallback_policy: str = "answer_with_evidence_or_clarify"
    confidence: float | None = None
    reason: str | None = None
```

### 5.3 RagMetrics

```python
class RagMetrics(BaseModel):
    route_ms: float | None = None
    rewrite_ms: float | None = None
    retrieve_ms: float | None = None
    dense_ms: float | None = None
    bm25_ms: float | None = None
    fusion_ms: float | None = None
    rerank_ms: float | None = None
    context_build_ms: float | None = None
    generate_ms: float | None = None
    total_ms: float | None = None
    dense_candidates: int = 0
    bm25_candidates: int = 0
    fused_candidates: int = 0
    reranked_candidates: int = 0
    final_sources: int = 0
```

### 5.4 RagDebugInfo

```python
class RagDebugInfo(BaseModel):
    debug_id: str
    rewritten_query: str | None = None
    hyde_document: str | None = None
    sub_queries: list[str] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    fused_ids: list[str] = Field(default_factory=list)
    reranked_ids: list[str] = Field(default_factory=list)
    selected_context_ids: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
```

生产响应可只返回 `debug_id`，完整 debug info 只在 `/api/rag/debug` 或开发环境返回。

### 5.5 RagResponse

```python
class RagResponse(BaseModel):
    request_id: str
    session_id: str | None = None
    answer: str
    sources: list[RagSource] = Field(default_factory=list)
    strategy: RagStrategy
    metrics: RagMetrics = Field(default_factory=RagMetrics)
    debug_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
```

## 6. SSE Event Schema

建议把 SSE 事件从“自由文本 data”收敛成稳定结构，前端可以渐进兼容。

### 6.1 通用事件结构

```python
class SseEvent(BaseModel):
    event: str
    request_id: str
    session_id: str | None = None
    stage: str
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
```

### 6.2 推荐事件类型

| event | stage | 说明 | data 示例 |
| --- | --- | --- | --- |
| `route_decided` | `routing` | Router 已决定 route、intent 和 strategy | `route`, `rag_intent`, `strategy_name`, `confidence` |
| `rewrite_done` | `rewriting` | query rewrite / HyDE / decompose 完成 | `rewritten_query`, `sub_queries` |
| `retrieving` | `retrieving` | 开始检索或检索完成 | `top_k_dense`, `top_k_bm25`, `dense_candidates`, `bm25_candidates` |
| `fusion_done` | `retrieving` | RRF 融合完成 | `fused_candidates` |
| `reranking` | `reranking` | reranker 开始或完成 | `candidate_k`, `reranked_candidates` |
| `context_done` | `context` | 上下文构建完成 | `final_sources`, `token_estimate` |
| `token` | `generating` | LLM 增量文本 | `delta` |
| `done` | `done` | 最终响应完成 | `answer`, `sources`, `strategy`, `metrics`, `debug_id` |
| `error` | `error` | 错误事件 | `code`, `message`, `recoverable` |

### 6.3 前端兼容策略

前端兼容分三层：第一层继续支持旧的 token 流，只要收到 `token` 或旧格式文本就更新答案；第二层识别 `stage` 事件，用于展示“正在路由、正在检索、正在重排序、正在生成”；第三层在 `done` 事件里读取 `sources`、`strategy`、`metrics`，用于展示引用、策略和耗时。

如果后端短期不能一次性改完，可以先保证 `done` 事件包含完整 `RagResponse`，阶段事件逐步补齐。

## 7. RAG Debug 接口

建议新增：

```text
POST /api/rag/debug
```

请求：

```json
{
  "query": "试用期请假制度是什么？",
  "session_id": "optional",
  "knowledge_base_id": "optional",
  "strategy_override": {
    "retrieval_mode": "dense_bm25_rrf_reranker",
    "use_hyde": false,
    "use_reranker": true
  }
}
```

响应：

```json
{
  "request_id": "...",
  "debug_id": "...",
  "query": "...",
  "rewritten_query": "...",
  "strategy": {},
  "dense_results": [],
  "bm25_results": [],
  "fused_results": [],
  "reranked_results": [],
  "selected_context": [],
  "response": {},
  "metrics": {},
  "warnings": []
}
```

这个接口的目标不是给最终用户使用，而是用于开发、评测和面试演示。它能说明一次回答为什么引用了某些文档，也能定位失败原因。

## 8. Agentic 策略矩阵

### 8.1 枚举建议

```python
class RagIntent(str, Enum):
    FACT_LOOKUP = "fact_lookup"
    SEMANTIC_QUERY = "semantic_query"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    CONSTRAINED = "constrained"
    FOLLOW_UP = "follow_up"
    NOT_ENOUGH_INFO = "not_enough_info"
    UNKNOWN = "unknown"

class RetrievalMode(str, Enum):
    DENSE_ONLY = "dense_only"
    BM25_ONLY = "bm25_only"
    DENSE_BM25_RRF = "dense_bm25_rrf"
    DENSE_BM25_RRF_RERANKER = "dense_bm25_rrf_reranker"
```

### 8.2 可配置策略矩阵

| rag_intent | retrieval_mode | dense_k | bm25_k | fusion_k | final_k | reranker | reranker_k | HyDE | rewrite | decompose | fallback |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |
| `fact_lookup` | `dense_bm25_rrf_reranker` | 20 | 20 | 20 | 5 | true | 10 | false | false | false | `answer_with_evidence_or_clarify` |
| `semantic_query` | `dense_bm25_rrf_reranker` | 30 | 10 | 25 | 5 | true | 10 | true | true | false | `rewrite_then_retry` |
| `multi_hop` | `dense_bm25_rrf_reranker` | 20 | 20 | 30 | 8 | true | 10 | false | true | true | `answer_with_partial_evidence` |
| `comparison` | `dense_bm25_rrf_reranker` | 20 | 20 | 30 | 8 | true | 10 | false | true | true | `group_by_entity_then_answer` |
| `procedure` | `dense_bm25_rrf` | 20 | 20 | 20 | 6 | false | 0 | false | true | false | `expand_parent_context` |
| `constrained` | `dense_bm25_rrf_reranker` | 20 | 30 | 25 | 5 | true | 10 | false | true | false | `metadata_filter_then_retry` |
| `follow_up` | `dense_bm25_rrf` | 20 | 20 | 20 | 5 | false | 0 | false | true | false | `history_aware_rewrite` |
| `not_enough_info` | `dense_bm25_rrf` | 10 | 10 | 10 | 3 | false | 0 | false | false | false | `clarify_or_insufficient_evidence` |
| `unknown` | `dense_bm25_rrf` | 20 | 20 | 20 | 5 | false | 0 | false | false | false | `safe_default` |

### 8.3 配置落地建议

短期可以先把策略矩阵写成 Python dict：

```python
RAG_STRATEGY_MATRIX = {
    "fact_lookup": RagStrategyConfig(
        retrieval_mode="dense_bm25_rrf_reranker",
        top_k_dense=20,
        top_k_bm25=20,
        fusion_top_k=20,
        final_top_k=5,
        use_reranker=True,
        reranker_candidate_k=10,
        fallback_policy="answer_with_evidence_or_clarify",
    ),
    ...
}
```

中期可以迁移到 YAML，便于调参和评测：

```text
backend/app/config/rag_strategy.yaml
```

### 8.4 子问题分解执行设计

子问题分解只用于 `multi_hop` 和 `comparison`，不作为企业知识库默认检索路径。默认路径仍然是原始 query 的 dense + BM25 + RRF；decompose 是复杂问题的策略分支，用于补齐多证据、多实体、多步骤问题的召回覆盖。

适用条件：

| 条件 | 处理方式 |
| --- | --- |
| `rag_intent=multi_hop` | 启用 rewrite + decompose，生成 2 到 4 个可独立检索的子问题。 |
| `rag_intent=comparison` | 启用 rewrite + decompose，按比较对象或比较维度生成子问题。 |
| `rag_intent=follow_up` | 只做 history-aware rewrite，不做 decompose，除非 rewrite 后被重新分类为 multi_hop / comparison。 |
| 有强 source constraint | 子问题必须继承原始 source_hints 和 metadata_filters，不能扩大检索范围。 |
| 子问题为空或不可信 | 回退到原始 query 的 dense + BM25 + RRF。 |

推荐执行链路：

```text
original_query
  -> optional history-aware rewrite
  -> decompose(original_or_rewritten_query)
       -> sub_query[1..N]
       -> 每个 sub_query 继承 source_hints / metadata_filters
  -> parallel retrieve per sub_query
       -> dense_search(sub_query)
       -> bm25_search(sub_query)
       -> rrf_fuse(sub_query candidates)
  -> cross_subquery_merge
       -> 按 parent_chunk_id / chunk_id 去重
       -> 保留 matched_sub_queries provenance
       -> 同一候选命中多个子问题时加权
  -> optional reranker
       -> comparison 使用 original_query + sub_queries 作为 rerank query
       -> multi_hop 使用 original_query 作为主 rerank query
  -> context build
       -> 按子问题覆盖率选择证据
       -> 至少优先覆盖不同子问题，而不是只取一个子问题的高分结果
  -> answer generation
       -> 提示模型按子问题组织证据并说明证据不足处
```

子问题生成约束：

```text
1. 生成 2 到 4 个子问题。
2. 每个子问题必须可以被知识库独立检索。
3. 不生成答案，不引入原始问题之外的事实假设。
4. 保留原始问题中的实体名、制度名、项目名、时间范围和来源约束。
5. comparison 问题优先按“对象 × 维度”拆分。
6. multi_hop 问题优先按“前置事实 -> 后续事实 -> 综合判断”拆分。
7. 如果无法可靠拆分，返回空列表并使用原始 query 检索。
```

推荐内部数据结构：

```python
class SubQuery(BaseModel):
    id: str
    query: str
    purpose: Literal["fact", "comparison_dimension", "procedure_step", "constraint", "verification"]
    depends_on: list[str] = Field(default_factory=list)

class SubQueryResult(BaseModel):
    sub_query_id: str
    sub_query: str
    dense_ids: list[str] = Field(default_factory=list)
    bm25_ids: list[str] = Field(default_factory=list)
    fused_ids: list[str] = Field(default_factory=list)
    elapsed_ms: float | None = None

class DecomposedCandidate(BaseModel):
    chunk_id: str
    parent_chunk_id: str
    matched_sub_query_ids: list[str] = Field(default_factory=list)
    fused_score: float = 0.0
    coverage_score: float = 0.0
    final_score: float = 0.0
```

跨子问题融合建议：

```text
base_score = max(candidate.rff_score_per_sub_query)
coverage_score = matched_sub_query_count / total_sub_query_count
final_score = base_score * (1.0 + 0.25 * coverage_score)
```

这个加权只用于候选合并阶段；最终相关性仍应优先交给 reranker 和上下文覆盖规则判断。不要把 decompose 做成“多个子问题结果简单拼接”，否则容易让一个高召回子问题挤掉其他必要证据。

Debug 和响应要求：

```text
RagDebugInfo.sub_queries: 保存子问题文本。
RagDebugInfo.retrieved_ids: 保存所有子问题召回候选。
RagDebugInfo.fused_ids: 保存跨子问题融合后的候选。
RagSource.metadata.matched_sub_queries: 保存每个最终引用命中的子问题 ID。
RagMetrics.rewrite_ms: 包含 rewrite + decompose 耗时；debug 接口可拆成 rewrite_ms 和 decompose_ms。
```

失败回退：

| 失败点 | 回退策略 |
| --- | --- |
| LLM 拆解失败 | 原始 query 直接检索。 |
| 子问题数量 < 2 | 原始 query 直接检索。 |
| 所有子问题召回为空 | 扩大原始 query 的 dense_k / bm25_k 后重试一次。 |
| 只有部分子问题有证据 | 允许回答已找到部分，并明确缺失证据。 |
| source constraint 下无证据 | 不放宽约束，返回证据不足或澄清问题。 |

## 9. 引用与上下文构建

上下文构建建议分四步：先根据 rerank 后的 child chunk 回填 parent chunk；再合并相邻 chunk，去除重复片段；然后按 token budget 选择最终上下文；最后从最终上下文中选择可展示引用片段。

引用片段必须来自实际传给 LLM 的上下文，不能引用未进入最终上下文的候选文档。否则会出现答案和引用不一致。

## 10. 实施顺序建议

当前已落地 `RagResponse`、`RagSource`、`RagStrategySummary`、`RagMetrics`、`AgenticRagGraph`、`RagEvidenceWorkflow`、策略矩阵、decompose 检索和 debug trace。下一步实施顺序建议调整为：第一步稳定 SSE 事件协议和前端展示；第二步完善 `/api/rag/debug` 的检索、融合、重排、证据评估可视化；第三步基于评测报告调参 strategy matrix；第四步完善外部搜索 fallback 和证据不足策略；第五步再评估 HyDE、contextual chunk 或更复杂的上下文压缩是否值得引入。