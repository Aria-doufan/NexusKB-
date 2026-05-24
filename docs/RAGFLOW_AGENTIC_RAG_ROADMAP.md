# RAGFlow Agentic RAG 执行路线图

整理日期：2026-05-16

## 1. 文档定位

本文只记录项目执行路线、阶段优先级、当前状态和下一步任务。系统架构、Response Schema、SSE 事件、Agentic 策略矩阵见 [RAGFlow Agentic RAG 架构设计](./RAGFLOW_AGENTIC_RAG_ARCHITECTURE.md)。评测指标、baseline、阈值和失败样例格式见 [RAGFlow Agentic RAG 评测方案](./RAGFLOW_AGENTIC_RAG_EVALUATION.md)。企业 RAG 最小安全边界见 [RAGFlow Agentic RAG 安全边界](./RAGFLOW_AGENTIC_RAG_SECURITY.md)。

项目统一定位为：

> RAGFlow：企业级多策略 Agentic RAG 检索增强问答系统。

项目主线是 RAG 检索质量优化。Agentic RAG 负责查询策略路由、复杂问题拆解、低置信度重试和工具编排，但不替代 RAG 主链路。

## 2. 状态标记规则

后续所有任务统一使用以下状态：

| 状态 | 含义 |
| --- | --- |
| 已完成 | 代码已有，并且文档确认过当前链路。 |
| 部分完成 | 核心代码或实验已有，但 schema、前端、测试、评测、文档或安全边界不完整。 |
| 待实现 | 当前没有稳定实现，或还没有接入主链路。 |
| 可选增强 | 不是面试项目 MVP 必需，但后续可提升展示质量。 |

## 3. 当前能力盘点

| 能力 | 状态 | 说明 | 后续动作 |
| --- | --- | --- | --- |
| FastAPI 主后端 | 已完成 | `backend/` 是当前主线。 | 继续作为 Agentic RAG 主服务。 |
| Vue 前端 | 已完成 | `front/` 已承担聊天、用户状态、会话入口。 | 补检索阶段、引用和策略展示。 |
| Django 用户服务 | 已完成 | 负责注册、登录、JWT、用户信息。 | 后续用于用户级文档隔离。 |
| LangGraph RouterGraph | 已完成 | 已统一承接 `/api/agent/query/stream` 主入口。 | 扩展输出策略枚举和调试信息。 |
| PureChat 轻链路 | 已完成 | 普通聊天不默认进入完整 Tool Agent。 | 保持与企业 RAG 分离。 |
| EnterpriseRagService | 已完成 | 企业 RAG 与原业务知识库链路已分离。 | 收敛为多策略 RAG Pipeline 主链路。 |
| Chroma 向量召回 | 已完成 | 已用于企业知识库和上传知识库检索。 | 保留为 dense retrieval baseline。 |
| BM25 召回 | 已完成 | 已接入混合召回。 | 与 dense 结果统一进入 RRF。 |
| RRF 融合 | 已完成 | 文档记录已有默认混合召回与 RRF。 | 在 schema 和评测中显式记录策略名。 |
| 条件 reranker | 已完成 | 已采用 Qwen3 reranker，且候选窗口收敛到 10。 | 补策略矩阵和响应字段。 |
| SSE 阶段事件 | 部分完成 | 已有 retrieving、reranking、summarizing 等真实事件。 | 标准化 SSE event schema，前端兼容。 |
| Working / Session Memory | 已完成 | 会话记忆已经接入 RouterGraph 与 SSE Agent 链路。 | 用于 follow-up query rewrite。 |
| PERF_METRIC 性能埋点 | 已完成 | 已覆盖 Router、RAG、MySQL、记忆等关键耗时。 | 聚合 avg、p95、策略维度指标。 |
| 引用溯源 | 部分完成 | RAG 文档已强调 sources，但统一响应结构还不稳定。 | 定义 `RagSource` schema 并接入前端。 |
| RAG Debug 接口 | 待实现 | 需要暴露召回、融合、重排、上下文和生成输入。 | 作为下一项开发任务。 |
| Contextual chunk | 待实现 | 当前主要是 parent-child 思路，context summary 还未稳定接入。 | 放入 P1。 |
| HyDE / query rewrite | 部分完成 | 旧 RAG 文档提到 HyDE，但需确认当前主链路是否稳定。 | 归入策略矩阵，可配置开关。 |
| 子问题拆解 | 待实现 | 多跳和对比问题还缺工程化 decomposition、跨子问题融合和证据覆盖控制。 | 放入 Agentic RAG P1；先做 debug 可观察，再接策略矩阵。 |
| 离线检索评测 | 已完成 | 已有 EnterpriseRAG-Bench 多策略评测。 | 扩展 citation、answer faithfulness、阈值。 |
| Citation Accuracy | 待实现 | 目前没有明确公式和通过阈值。 | 放入评测方案 P0/P1。 |
| Answer Faithfulness | 待实现 | 可先用人工抽检或 LLM-as-judge。 | 放入评测方案 P1。 |
| Prompt injection 防护 | 部分完成 | 已有 system/unsafe 路由和审计基础。 | 需要 RAG 文档指令隔离策略。 |
| 用户文档隔离 / ACL | 部分完成 | JWT 和用户服务已有，真实数据 ACL 仍待完善。 | 作为企业 RAG 最小安全边界提前做。 |

## 4. 阶段路线

### 阶段 0：文档和主线收敛

目标：先把方向、状态、架构、评测和安全拆开，避免一个大文档混用。

| 优先级 | 任务 | 状态 | 验收标准 |
| --- | --- | --- | --- |
| P0 | 拆分总路线图 | 已完成 | ROADMAP、ARCHITECTURE、EVALUATION、SECURITY 四类文档存在。 |
| P0 | 标注当前实现状态 | 已完成 | 文档中明确已完成、部分完成、待实现、可选增强。 |
| P0 | 明确下一项开发任务 | 已完成 | 下一项为 RAG Debug 接口 + 统一 Response Schema。 |

### 阶段 1：统一响应结构与 RAG Debug

目标：让 RAG 链路可观察、可调试、可被前端稳定消费。

| 优先级 | 任务 | 状态 | 验收标准 |
| --- | --- | --- | --- |
| P0 | 定义 Pydantic Response Schema | 待实现 | `RagResponse`、`RagSource`、`RagStrategy`、`RagMetrics` 可在后端复用。 |
| P0 | 定义 SSE Event Schema | 待实现 | `route_decided`、`retrieving`、`reranking`、`generating`、`done`、`error` 字段固定。 |
| P0 | 新增 RAG Debug 接口 | 待实现 | 返回 query、strategy、retrieved、fused、reranked、context、sources、metrics。 |
| P0 | 前端兼容最终响应 | 待实现 | 旧流式文本不破坏，新字段可渐进展示。 |
| P1 | 保存 debug_id | 待实现 | 每次 RAG 请求可用 debug_id 追踪失败样例。 |

推荐接口：

```text
POST /api/rag/debug
POST /api/agent/query/stream
```

阶段 1 完成后，应能判断一次 RAG 失败到底发生在召回、融合、重排、上下文压缩、引用选择还是生成阶段。

### 阶段 2：高质量 RAG 主链路

目标：把已有 hybrid retrieval、RRF、条件 reranker、parent-child chunk 和引用溯源统一到一条可配置主链路。

| 优先级 | 任务 | 状态 | 验收标准 |
| --- | --- | --- | --- |
| P0 | 固化 dense + BM25 + RRF 主召回 | 部分完成 | 响应 strategy 中能明确记录 `dense_bm25_rrf`。 |
| P0 | 固化条件 reranker | 已完成 | 复杂、精确、低置信度问题可启用 reranker，默认候选窗口为 10。 |
| P0 | 标准化 sources | 待实现 | 每个 source 包含 doc_id、chunk_id、parent_id、title、section、page、snippet、score。 |
| P0 | parent chunk 回填 | 部分完成 | child 检索结果可回填 parent 内容进入上下文。 |
| P1 | token budget 控制 | 待实现 | 上下文组装不超过配置上限，保留引用片段。 |
| P1 | contextual chunk | 待实现 | chunk 可注入章节摘要、文档位置和语义上下文。 |

阶段 2 完成后，项目可以稳定讲清楚“为什么不是简单 TopK 向量检索”。

### 阶段 3：Agentic 策略矩阵

目标：让 RouterGraph 不只做 route 分类，还能输出可执行的 RAG 策略。

| 优先级 | 任务 | 状态 | 验收标准 |
| --- | --- | --- | --- |
| P0 | 定义 `RagIntent` 枚举 | 待实现 | 与文档策略矩阵一致，非法值可兜底。 |
| P0 | 定义 `RagStrategyConfig` | 待实现 | topK、reranker、HyDE、decompose、fallback 可配置。 |
| P0 | 接入策略矩阵 | 待实现 | Router 输出 query type 后可查表得到策略配置。 |
| P1 | history-aware rewrite | 部分完成 | 多轮追问能结合会话记忆补全查询。 |
| P1 | HyDE 策略开关 | 部分完成 | 可按 query type 启用，并在评测中对比。 |
| P1 | 子问题拆解 | 待实现 | `multi_hop` / `comparison` 能生成 2 到 4 个子查询；每个子查询独立 dense + BM25 + RRF；跨子查询按覆盖率合并证据；debug 输出 sub_queries 和每个子查询命中。 |
| P1 | 低置信度 fallback | 待实现 | 可 rewrite retry、扩大 topK、clarify 或回答证据不足。 |

策略矩阵的工程定义见架构文档。

### 阶段 4：评测闭环

目标：让项目能用指标证明策略改动是否有效。

| 优先级 | 任务 | 状态 | 验收标准 |
| --- | --- | --- | --- |
| P0 | 明确 baseline | 部分完成 | 至少包含 chroma_only、bm25_only、dense_bm25_rrf、dense_bm25_rrf_reranker、strategy_matrix。 |
| P0 | 明确指标公式 | 已完成 | 评测文档写清 Hit@K、Recall@K、MRR、Latency。 |
| P0 | 失败样例格式 | 待实现 | 保存 query、gold、retrieved、fused、reranked、answer、failure_type。 |
| P1 | Citation Accuracy | 待实现 | 引用命中 gold doc/chunk 的比例可统计。 |
| P1 | Answer Faithfulness | 待实现 | 初版可用人工抽检或 LLM-as-judge。 |
| P1 | 策略对比报告 | 待实现 | 生成 Markdown/CSV 报告，可用于面试展示。 |

评测细节见评测方案文档。

### 阶段 5：企业 RAG 最小安全边界

目标：即使是面试项目，也要体现企业知识库的基本安全意识。

| 优先级 | 任务 | 状态 | 验收标准 |
| --- | --- | --- | --- |
| P0 | 用户文档隔离 | 部分完成 | 检索必须带 user_id / tenant_id / knowledge_base_id 过滤。 |
| P0 | 引用泄露防护 | 待实现 | sources 只返回用户有权限查看的片段。 |
| P0 | prompt injection 防护 | 部分完成 | 文档内容作为 untrusted context，不允许覆盖系统指令。 |
| P0 | 工具风险元数据 | 已完成 | 工具已有 risk_level、data_scope、operation 等元数据。 |
| P1 | 审计事件补齐 | 部分完成 | RAG 查询、工具调用、权限拒绝、引用返回可记录 AUDIT_EVENT。 |
| P1 | 日志脱敏 | 待实现 | 普通日志不泄露 token、密钥、原始敏感片段。 |

安全细节见安全边界文档。

## 5. 当前最推荐下一步

下一项开发任务建议是：

```text
RAG Debug 接口 + 统一 RAG Response Schema + SSE Event Schema
```

原因是这项任务能同时支撑后续三件事：前端展示引用和阶段事件，评测保存失败样例，Agentic 策略矩阵输出可解释结果。没有这个基础，后续做 HyDE、decompose、contextual chunk 或 citation accuracy 都很难判断效果。

## 6. 暂不优先做

以下事情暂不作为近期重点：

| 事项 | 原因 |
| --- | --- |
| 替换 Milvus / Elasticsearch / OpenSearch | 当前 Chroma + BM25 足够支持面试项目主线。 |
| 做完整 RAGFlow 式可视化编排平台 | 范围过大，容易偏离 RAG 检索质量主线。 |
| 多 Agent 协作 | 当前 Agentic 重点是策略路由，不是多 Agent 炫技。 |
| 完整组织权限系统 | 先做用户文档隔离和引用泄露防护即可。 |
| 大规模重构后端目录 | 先用 schema 和 debug 接口稳定主链路，再考虑模块拆分。 |

## 7. 文档维护规则

每次完成一项开发任务后，同步更新本文的状态。每次新增 schema、策略矩阵、指标阈值或安全规则，同步更新对应专题文档。