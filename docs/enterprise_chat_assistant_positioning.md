# 企业聊天助手定位与能力差距分析

## 当前定位

项目定位调整为：

```text
企业聊天助手：以自然对话为基础入口，延伸企业知识、会话/长期记忆、权限安全和可控工具能力。
```

它不是单纯的知识库机器人，也不是纯 Agent 平台。更准确地说，它是面向企业员工的聊天式工作助手：

- 员工可以像普通聊天一样提问、追问、整理信息。
- 当问题涉及企业资料时，系统自动进入企业知识库检索。
- 当需要连续工作上下文时，系统使用会话记忆和用户长期记忆。
- 当涉及内部数据和工具时，系统必须遵守用户身份、文档权限和工具权限。

## 产品原则

1. 聊天是入口，不是附属能力。
2. 企业知识是增强层，不要求用户显式说“检索知识库”。
3. 记忆是基础体验，但必须可控、可查看、可删除。
4. 权限安全是企业场景底线，不能被 RAG、记忆或工具调用绕过。
5. Agent 工具调用是可选能力，应该受 Router 和权限约束。
6. Workspace Memory 暂不做，避免过早引入团队共享记忆和组织级治理复杂度。

## 目标能力框架

```text
Enterprise Conversational Assistant

1. Conversation Core
   - 基础聊天
   - 流式响应
   - 多轮上下文
   - 会话管理

2. Memory Layer
   - Working Memory：最近若干轮原文
   - Session Memory：当前会话滚动摘要
   - User Long-term Memory：用户长期偏好、工作背景、常用上下文
   - Workspace Memory：暂不需要

3. Enterprise Knowledge Layer
   - 企业知识库 RAG
   - parent-child chunk
   - Chroma 向量检索
   - BM25 关键词召回
   - RRF 混合融合
   - Qwen3 Reranker 可选重排
   - 答案来源引用

4. Router & Strategy Layer
   - 判断 chat / rag_query / agent_tool_call / clarify / system
   - 判断 rag_intent
   - 判断 source_hints
   - 按意图选择检索策略
   - 低置信度澄清

5. Permission & Safety Layer
   - 用户身份
   - 会话归属校验
   - 文档级 ACL
   - source 级 ACL
   - 工具权限
   - 危险操作保护
   - 记忆权限边界

6. Tool Actions Layer
   - 查询类工具
   - 文档处理工具
   - 任务/工单/项目类工具
   - 受控写操作

7. Evaluation & Observability
   - 检索评测
   - 回答来源质量
   - 延迟统计
   - 失败样例分析
   - 工具调用审计
```

## 当前能力对照

### 总览

| 能力 | 当前状态 | 判断 | 主要证据 | 下一步 |
| --- | --- | --- | --- | --- |
| 基础聊天 | 已有 | 需要改进 | `/api/agent/query/stream`、`get_agent_stream_response` | 保留流式体验，避免普通聊天误触工具 |
| 会话管理 | 已有 | 需要小幅改进 | `chat_sessions`、`chat_messages`、session CRUD | 增加标题生成、搜索、分页、归档 |
| Working Memory | 已有 | 基本可用 | 最近 6 轮原文 | 后续改为 token-aware window |
| Session Memory | 已有 | 基本可用 | `chat_session_memories` 滚动摘要 | 摘要注入方式从虚拟历史改成 system/context |
| User Long-term Memory | 没有 | 需要新增 | 暂无用户长期记忆表和服务 | 设计 MVP：偏好、工作背景、长期任务状态 |
| Workspace Memory | 暂不需要 | 不做 | 用户明确暂不需要 | 不进入当前路线 |
| Router Graph | 已有 | 需要增强 | `router_graph.py`、非流式 Router API | 接入 SSE，拆普通 chat 链 |
| 企业 RAG | 已有 | 需要增强 | `EnterpriseRagService`、parent-child Chroma | 接入混合检索和 reranker 到主链路 |
| Chroma 向量检索 | 已有 | 可用 | `chromadb_enterprise_parent_child` | 保留 |
| BM25 混合召回 | 评测已有，主链路没有 | 需要产品化 | `evaluate_enterprise_hybrid_retrieval.py` | 抽成 retrieval service |
| Reranker | 评测已有，主链路没有 | 需要策略化接入 | Qwen3 reranker GPU 评测完成 | 按 rag_intent 开关启用 |
| 答案来源引用 | 部分已有 | 需要改进 | `EnterpriseRetrievedDocument.to_dict()` 返回来源字段 | API 响应明确返回 sources |
| 权限系统 | 很弱 | 完全需要补 | 只有 JWT user_id、会话归属和上传清理 | 增加文档 ACL/source ACL/工具权限 |
| 工具调用 | 已有基础 | 需要治理 | Agent tool calling 已有，system 路由保守处理 | 工具分类、权限、审计 |
| 危险操作保护 | 部分已有 | 需要扩展 | `system` 路由不直接执行危险操作 | 对工具写操作加确认流 |
| 检索评测 | 已有 | 良好 | 三组评测已完成 | 做失败样例分析与策略矩阵 |
| 观测与审计 | 很弱 | 需要新增 | 有日志，缺结构化审计 | 增加 retrieval trace、tool audit、memory audit |

## 已经具备的能力

### 1. 基础聊天和流式响应

当前已有：

- `POST /api/agent/query/stream`
- SSE 流式返回
- JWT 用户识别
- Redis 限流
- MySQL 会话落库

问题：

- 当前流式接口还没有先经过 Router 分流。
- 普通聊天目前仍复用 Agent 链路，可能误触工具。

建议：

- 保留现有流式聊天作为主入口。
- 下一步做 Router SSE 化，让聊天入口也能自动选择 chat / RAG / tool。
- 将 `chat_node` 拆成纯聊天链，工具调用只通过 `agent_tool_call` 进入。

### 2. 会话记忆

当前已有两层记忆：

```text
Working Memory：最近 6 轮原文
Session Memory：当前 session 的滚动摘要
```

已完成：

- `chat_session_memories` 表
- `ConversationMemoryService`
- RouterGraph 接入压缩记忆
- SSE Agent 接入压缩记忆

问题：

- 当前是 session 级记忆，不是用户长期记忆。
- 使用轮数阈值，不是 token 阈值。
- 摘要以虚拟 `(user, assistant)` 历史形式注入，不够自然。
- 没有记忆查看、删除、修正。

建议：

- 短期内保留现有结构。
- 增加 token-aware 压缩。
- 将摘要注入改成 system/context message。
- 增加记忆管理接口。

### 3. 企业知识库 RAG

当前已有：

- `EnterpriseRagService`
- parent-child Chroma 检索
- child chunk 检索后回填 parent chunk 原文
- `rag_intent` 影响 `top_k/search_k`
- `source_hints` 记录但不硬过滤

问题：

- 主链路还是 Chroma-only。
- BM25 和 reranker 目前只在评测脚本中。
- 答案返回结构还没有把 sources 作为一等字段暴露给前端。

建议：

- 新增 `EnterpriseRetrievalService`，把 Chroma、BM25、RRF、reranker 从评测脚本抽到业务层。
- `EnterpriseRagService` 只负责“检索资料 -> 生成答案”，不要继续堆检索细节。
- API 响应返回 `answer + sources + retrieval_strategy + confidence`。

### 4. Router Graph

当前已有：

- `POST /api/agent/router/query`
- LangGraph Router Graph 第一版
- `rag_query`、`agent_tool_call`、`chat`、`system`、`clarify`
- `rag_intent`
- `source_hints`
- 低置信度兜底
- 危险 system 请求保守处理

问题：

- Router 目前主要是非流式接口。
- SSE 主入口还没有经过 Router。
- `rag_intent` 还没有完整策略矩阵。
- `source_hints` 还没有用于软加权。

建议：

- 优先做 Router SSE 化。
- 让 Router 决定检索策略，而不只是决定 route。
- 基于评测结果配置策略矩阵。

### 5. 检索评测

已完成三组评测：

| 方案 | hit@1 | hit@20 | recall@20 | mrr@20 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chroma only | 0.614 | 0.814 | 0.7858 | 0.6744 | 129.23ms |
| Chroma + BM25 + RRF | 0.640 | 0.900 | 0.8868 | 0.7190 | 370.40ms |
| Chroma + BM25 + Reranker | 0.734 | 0.900 | 0.8868 | 0.7846 | 1149.55ms |

结论：

- BM25 + RRF 应进入主链路，因为显著提升召回。
- Reranker 应作为策略能力保留，因为显著提升 Top1 和 MRR。
- Reranker 不应无脑默认开启，应由 `rag_intent`、问题复杂度或用户显式要求控制。

## 需要稍微改进的能力

### 1. Conversation Core

当前聊天可用，但需要从“Agent 聊天”改成“聊天核心 + 能力分流”。

改进项：

- 纯聊天链。
- Router 接入流式入口。
- 统一响应格式。
- 会话标题、分页、归档。

### 2. Session Memory

当前方案可用，但还不像一个企业聊天助手的记忆系统。

改进项：

- token-aware 最近窗口。
- 摘要注入方式调整。
- 记忆可查看、删除、修正。
- 摘要质量评估。

### 3. Enterprise RAG

当前能回答，但还不是企业助手级别。

改进项：

- 主链路接入 hybrid retrieval。
- 主链路接入可选 reranker。
- sources 作为响应字段。
- 检索 trace 可调试。
- 失败样例驱动策略优化。

### 4. Router Strategy

当前 Router 能分流，但策略能力还弱。

建议第一版策略矩阵：

| rag_intent | 检索策略 |
| --- | --- |
| `basic` | Chroma + BM25 + RRF |
| `semantic` | Chroma + BM25 + RRF |
| `constrained` | Chroma + BM25 + RRF + reranker |
| `project_related` | Chroma + BM25 + RRF + reranker |
| `completeness` | 扩大 candidate_k + reranker |
| `conflicting_info` | 多来源召回 + reranker + 展示差异 |
| `high_level` | Chroma + BM25，reranker 可选 |
| `info_not_found` | 低置信度时返回未找到和候选来源 |
| `unknown` | 默认 Chroma + BM25 + RRF |

## 完全没有或基本缺失的能力

### 1. User Long-term Memory

这是当前最重要的缺口之一。

不做 Workspace Memory，但需要 User Memory。

建议 MVP 存储内容：

```text
用户偏好：
- 回答语言
- 喜欢简洁/详细
- 喜欢表格/步骤/结论优先

用户工作背景：
- 常关注的项目
- 常用数据源
- 常见职责领域

长期任务状态：
- 用户反复推进的事项
- 已确认的长期目标
- 用户明确要求记住的信息
```

必须避免：

- 保存敏感文档原文。
- 保存用户无意授权的隐私信息。
- 用长期记忆绕过企业知识库权限。

建议数据表：

```text
user_memories
id
user_id
memory_type
content
source
confidence
status
created_at
updated_at
last_used_at
```

建议能力：

- 用户说“记住...”时显式写入。
- 对高置信偏好可建议写入，但不要静默无限制写入。
- 提供查看、删除、禁用接口。

### 2. Permission & Safety Layer

当前只有：

- JWT 获取 `user_id`
- 会话归属校验
- 用户上传向量清理
- system 危险操作保守返回

缺失：

- 企业文档 ACL。
- source-level 权限。
- 检索阶段权限过滤。
- 生成前 context 权限过滤。
- sources 展示权限过滤。
- 工具级权限。
- 记忆写入权限和敏感信息过滤。

建议权限链路：

```text
user_id
  -> resolve user roles/groups
  -> retrieval ACL filter
  -> context ACL filter
  -> source citation ACL filter
  -> tool permission check
  -> audit log
```

### 3. Tool Governance

当前有 Agent tool calling，但缺少企业级治理。

缺失：

- 工具分类：read-only / write / dangerous。
- 工具权限矩阵。
- 写操作确认。
- 工具调用审计。
- 工具失败恢复。

建议：

- 查询类工具可以直接执行。
- 写操作必须确认。
- 危险操作必须二次确认并记录审计。

### 4. Observability & Audit

当前有日志，但缺少结构化观测。

缺失：

- 每次 RAG 的检索 trace。
- Router 决策日志。
- reranker 开关和耗时。
- 工具调用审计。
- 记忆读写审计。
- 权限拒绝记录。

建议新增：

```text
assistant_interaction_traces
retrieval_traces
tool_call_audits
memory_audits
permission_denials
```

## 推荐下一阶段路线

### 阶段 1：把当前聊天助手主链路理顺

目标：

- 聊天入口统一经过 Router。
- Router 支持流式。
- chat 与 tool_call 分离。
- 企业 RAG 返回 sources。

优先级：

1. Router SSE 化。
2. 拆纯聊天链。
3. 统一响应 schema。
4. RAG sources 前端可见。

### 阶段 2：把检索评测成果产品化

目标：

- 把 BM25 + RRF + reranker 从评测脚本迁移到业务服务。
- 按 `rag_intent` 控制策略。

优先级：

1. 新增 `EnterpriseRetrievalService`。
2. 接入 Chroma + BM25 + RRF。
3. 接入 Qwen3 reranker 可选重排。
4. 做 `rag_intent` 策略矩阵。

### 阶段 3：User Long-term Memory MVP

目标：

- 做用户级长期记忆，不做 Workspace Memory。
- 只存偏好、工作背景、长期任务状态。

优先级：

1. 设计 `user_memories` 表。
2. 新增记忆读写服务。
3. 显式“记住/忘记/查看记忆”接口。
4. Router/Chat 加载用户记忆。
5. 敏感信息过滤。

### 阶段 4：权限安全

目标：

- 企业数据检索不能越权。
- 工具调用不能越权。
- 记忆不能越权。

优先级：

1. 文档 metadata 增加 ACL 字段。
2. 检索前 ACL filter。
3. sources 展示过滤。
4. 工具权限矩阵。
5. 权限拒绝审计。

## 当前最应该先做什么

如果按“企业聊天助手”的定位，当前最推荐的下一步不是继续加新模型，而是：

```text
Router SSE 化 + 纯聊天链拆分
```

原因：

- 聊天是产品主入口。
- 当前非流式 Router 和流式聊天是两套入口，体验和能力不统一。
- 企业知识、工具、记忆都应该挂在统一聊天入口后面。
- 普通聊天不应该默认复用可调用工具的 Agent 链路。

第二步再做：

```text
EnterpriseRetrievalService 产品化
```

把已经验证有效的 BM25 + RRF + reranker 真正接进企业 RAG 主链路。
