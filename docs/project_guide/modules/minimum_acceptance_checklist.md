# 核心模块最小验收标准

整理日期：2026-05-15

本文档定义每个核心模块的最小可执行验收标准。它和项目级 [验收标准](../../ACCEPTANCE_CRITERIA.md) 的关系是：

- `ACCEPTANCE_CRITERIA.md` 说明阶段目标和原则。
- 本文档说明每个模块怎么验、看什么返回、慢在哪里、失败时怎么兜底、用什么命令复现。

## 通用验收要求

| 项目 | 最小标准 |
| --- | --- |
| 接口返回 | 返回结构稳定，字段命名和前端消费一致；流式接口必须输出 `done` 或 `error` 后再结束 |
| 延迟观测 | 关键链路必须输出 `PERF_METRIC`，至少包含 `name` 和 `elapsed_ms` |
| 错误兜底 | 外部依赖失败时返回用户可理解信息，不暴露 API Key、Token、连接串或完整堆栈 |
| 测试命令 | 每个模块至少有一个可执行命令用于 smoke test 或离线评测 |
| 文档记录 | 新增实验或策略调整必须写入 `docs/experiments/`，总览同步到 `docs/WORKBOARD.md` |

## 1. FastAPI 主服务

核心文件：

- `backend/main.py`
- `backend/app/router/health.py`

接口返回：

| 接口 | 最小返回 |
| --- | --- |
| `GET /health/live` | `status=ok` |
| `GET /health/ready` | MySQL 和 Redis 可用时返回 `status=ok`；任一失败返回 `503` |

延迟与日志：

- 启动日志应包含数据库初始化、Redis 初始化、BM25 预热任务启动。
- HTTP 响应应包含 `X-Process-Time`。

错误兜底：

- Redis 或 MySQL 不可用时，`/health/ready` 返回 `503`。
- BM25 预热失败只记录 warning，不阻塞服务启动。

测试命令：

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

通过条件：

- 服务能启动。
- `/health/live` 通过。
- 完整环境下 `/health/ready` 通过。
- 日志中出现 `企业 RAG BM25 索引预热任务已启动`。

## 2. Agentic RAG 主入口

核心文件：

- `backend/app/agent/router_graph.py`
- `backend/app/rag/agentic_rag_graph.py`
- `backend/app/rag/rag_evidence_workflow.py`

接口返回：

| 接口 | 最小返回 |
| --- | --- |
| `POST /api/agent/router/query` | `session_id`、`route=agentic_rag`、`request_id`、`debug_id`、`rag_intent`、`source_hints`、`confidence`、`reason`、`response`、`steps` |
| `POST /api/agent/query/stream` | 先输出兼容 `route` 事件，再输出 workflow 阶段事件、`response`、`done` |

延迟与日志：

- `RagResponse.metrics` 至少能表达 `total_ms`、`retrieval_ms`、`generation_ms`、`retry_count`、`retrieval_attempts`。
- 底层企业 RAG 检索继续输出 `enterprise_rag.*` PERF_METRIC。
- Debug trace 必须包含 `request_id` 和 `debug_id`，便于定位 planner、strategy、retrieval、evaluation 和 generation。

错误兜底：

- decision chain 输出非法 action 时回退安全路径。
- 低置信度、上下文不足或证据不足时进入 `clarify` 或 evidence-insufficient 响应。
- 删除、清空、重置、越权等高风险请求进入 `refuse`。

测试命令：

```powershell
conda run -n NexusKB python -m py_compile backend\app\agent\router_graph.py backend\app\rag\agentic_rag_graph.py backend\app\rag\rag_evidence_workflow.py
```

真实接口 smoke test 需要带 JWT：

```powershell
# 使用 backend/.env 中 SECRET_KEY 和 ALGORITHM 生成测试 JWT 后请求：
POST http://127.0.0.1:8000/api/agent/query/stream
Authorization: Bearer <test_jwt>
Content-Type: application/json

{"session_id":"router-smoke","query":"What is the contractor to FTE conversion process in the enterprise knowledge base?"}
```

通过条件：

- 兼容 route 事件返回 `route=agentic_rag`。
- 企业知识问题触发 `retrieve` action，并返回 `debug_id`、`sources` 或证据不足说明。
- 普通直接回答不触发企业 RAG 检索。
- 高风险系统操作返回 `route=unsafe_or_system` 或安全拒绝。

## 3. PureChat 普通聊天

核心文件：

- `backend/app/agent/agent.py`

接口返回：

- 流式事件类型至少包含 `response` 和 `done`。
- 普通聊天不返回企业 RAG `documents`。

延迟与日志：

- `pure_chat.prepare_chain`
- `pure_chat.llm_first_token`
- `pure_chat.llm_stream_total`
- `pure_chat.persist_and_memory`
- `pure_chat.stream_total`

错误兜底：

- LLM 调用失败时输出 `error` 事件和 `done`。
- 安全小工具失败时回退纯聊天。

测试命令：

```powershell
backend\.venv\Scripts\python.exe -m py_compile backend\app\agent\agent.py
```

通过条件：

- 闲聊、写作、解释概念不查企业知识库。
- 当前时间、天气等安全工具只通过 PureChat 轻链路处理。
- 出错时用户可见响应不包含内部堆栈。

## 4. 企业 RAG 服务

核心文件：

- `backend/app/rag/enterprise_rag_service.py`
- `backend/scripts/evaluate_enterprise_hybrid_retrieval.py`

接口返回：

| 字段 | 最小标准 |
| --- | --- |
| `documents` | 返回 parent chunk 回填后的资料列表 |
| `summary` | 基于资料生成回答；资料不足时明确说明 |
| `strategy` | 包含 `retrieval`、`reranker`、`reranker_candidate_k`、`source_hint_mode`、`rag_intent` |

流式阶段事件：

- `retrieving start/done`
- 需要 reranker 时输出 `reranking start/done`
- `summarizing start/done`

延迟与日志：

- `enterprise_rag.chroma_search`
- `enterprise_rag.bm25_search`
- `enterprise_rag.rrf_fuse`
- `enterprise_rag.reranker`
- `enterprise_rag.retrieve_phase`
- `enterprise_rag.summary_chain`
- `enterprise_rag.total`

错误兜底：

- 摘要生成失败时返回检索资料列表的 fallback summary。
- BM25 预热失败时首个请求可按需构建。
- reranker 失败时保留 RRF 顺序。

测试命令：

```powershell
backend\.venv\Scripts\python.exe -m py_compile backend\app\rag\enterprise_rag_service.py

backend\.venv\Scripts\python.exe backend\scripts\evaluate_enterprise_hybrid_retrieval.py --method strategy_matrix --output-name strategy_matrix_smoke --limit 10 --reranker-device cuda --reranker-max-length 512 --reranker-candidate-k 10 --reranker-batch-size 4
```

真实 SSE 验收：

```text
复杂企业知识问题应出现：
stage=reranking status=start candidates=10
stage=reranking status=done candidates=10
response.strategy.reranker=true
response.strategy.reranker_candidate_k=10
```

通过条件：

- 默认召回为 `chroma+bm25+rrf`。
- source hints 只软加权，不硬过滤。
- `strategy_matrix_k10` 离线评测不低于既有 `candidate_k=20` 的核心指标。

## 5. Tool Agent

核心文件：

- `backend/app/agent/agent.py`
- `backend/app/agent/agent_tools.py`

接口返回：

- 流式输出 `response` 和 `done`。
- 工具调用路径应保留 `steps`，至少包含 `tool`、`tool_input`、`tool_output`。

延迟与日志：

- `tool_agent.load_history`
- `tool_agent.first_token`
- `tool_agent.stream_generation_total`
- `tool_agent.persist_and_memory`
- `tool_agent.stream_total`

错误兜底：

- 工具执行失败时返回可理解错误。
- 普通聊天只允许 `CHAT_SAFE_TOOLS`，不进入完整工具池。
- 完整工具池只通过 `tool_action` 分支进入。

测试命令：

```powershell
backend\.venv\Scripts\python.exe -m py_compile backend\app\agent\agent.py backend\app\agent\agent_tools.py
```

通过条件：

- 每个工具必须有 `risk_level`、`data_scope`、`operation`、`requires_confirmation`。
- `chat_safe` 工具池只包含天气和时间。
- `full` 工具池包含 RAG、重排序、用户信息、天气、时间。
- 高风险动作不应存在可直接执行工具；如果后续新增，必须先接入确认和审计。

## 6. 会话记忆

核心文件：

- `backend/app/services/conversation_memory.py`
- `backend/app/services/database_session_manager.py`
- `backend/app/models/chat_history.py`

接口返回：

- `GET /api/session/{session_id}` 返回当前用户有权访问的历史。
- Router/PureChat/Tool Agent 均能读取最近历史和摘要记忆。

延迟与日志：

- `memory.context_get`
- `memory.update`
- `memory.summary_chain`
- `mysql.memory_history_get`
- `mysql.memory_get`
- `mysql.memory_create`
- `mysql.memory_save`

错误兜底：

- 摘要记忆读取失败时回退完整历史。
- 记忆更新失败不应中断主回答。

测试命令：

```powershell
backend\.venv\Scripts\python.exe -m py_compile backend\app\services\conversation_memory.py backend\app\services\database_session_manager.py
```

通过条件：

- 同一 `session_id` 的多轮问题可读取上下文。
- 不同用户不能读取彼此会话。
- 记忆失败时主链路仍能返回回答。

## 6.1 长期记忆

核心文件：

- `backend/app/services/long_term_memory.py`
- `backend/app/models/chat_history.py`
- `backend/app/router/chat.py`
- `backend/scripts/evaluate_long_term_memory.py`
- `backend/scripts/memory_eval_golden_cases.jsonl`

接口返回：

| 接口 | 最小返回 |
| --- | --- |
| `GET /api/memories` | 当前用户 active 长期记忆列表 |
| `DELETE /api/memories/{memory_id}` | 软删除当前用户指定记忆 |

当前公开 API 尚未暴露 `GET /api/memories/search?q=...`；语义搜索由问答链路内部调用，独立搜索端点属于后续扩展。

延迟与日志：

- `long_term_memory.extract_and_store`
- `long_term_memory.search`
- `long_term_memory.delete`

错误兜底：

- 长期记忆检索失败时跳过长期记忆上下文，不中断主回答。
- 长期记忆抽取失败时保留会话历史，不中断主回答。
- 删除 MySQL 状态以 `deleted` 为准，向量删除失败只记录 warning。

测试命令：

```powershell
backend\.venv\Scripts\python.exe -m compileall backend\app backend\scripts\evaluate_long_term_memory.py

backend\.venv\Scripts\python.exe backend\scripts\evaluate_long_term_memory.py `
  --base-url http://127.0.0.1:8000 `
  --user-id memory-eval-user `
  --other-user-id memory-eval-other-user `
  --settle-seconds 3 `
  --output-name memory_v2_smoke
```

通过条件：

- `memory_search_hit_rate >= 0.75`。
- `answer_hit_rate >= 0.75`。
- `delete_pass_rate = 1.0`。
- `isolation_pass_rate = 1.0`。
- 详细流程参考 [长期记忆评估流程](../../experiments/memory_eval.md)。

## 7. MySQL / Redis

核心文件：

- `backend/app/db/db_config.py`
- `backend/app/db/redis_config.py`

接口返回：

- `/health/ready` 同时验证 MySQL 和 Redis。

延迟与日志：

- MySQL 写入相关指标：`mysql.session_get`、`mysql.message_add`。
- Redis 当前主要用于 JWT 黑名单和缓存，后续如加入缓存命中率需补指标。

错误兜底：

- MySQL 不可用：ready 返回 `503`，业务写入失败应记录 warning 或 error。
- Redis 不可用：ready 返回 `503`，鉴权黑名单检查可能失败，应在安全设计中收敛策略。

测试命令：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

通过条件：

- MySQL 可建表/读写。
- Redis 可 ping。
- 服务启动后数据库会话管理器初始化成功。

## 8. 前端 SSE 聊天

核心文件：

- `front/src/views/AIChat.vue`
- `front/src/config/api.js`

接口返回：

- 前端能解析 `route`、`stage`、`response`、`done`、`error`。
- 最终 `response.strategy` 保存到当前助手消息的 `ragStrategy`。

状态反馈：

- 发送中显示占位。
- RAG 阶段显示当前阶段，例如检索、重排、生成回答。
- 错误时替换为错误提示。

测试命令：

```powershell
cd front
npm run build
```

通过条件：

- 构建通过。
- 真实 SSE 请求中前端能显示阶段状态。
- `reranking` 阶段能接收 `candidates=10`，最终 `message.ragStrategy.reranker_candidate_k=10`。

## 9. 文档与实验记录

核心文件：

- `docs/WORKBOARD.md`
- `docs/experiments/enterprise_retrieval_eval.md`
- `docs/experiments/enterprise_rag_latency_optimization.md`
- `docs/PERFORMANCE_METRICS.md`
- `docs/archive/RAG_EVALUATION_PLAN.md`

验收标准：

- 每次重要尝试写入详细实验文档。
- 每次阶段状态变化同步 `WORKBOARD.md`。
- 有数据支撑的策略调整必须记录命令、配置、结果和结论。

测试命令：

```powershell
Get-ChildItem docs -Recurse -File -Filter *.md
```

通过条件：

- 能从 `WORKBOARD.md` 找到当前正在进行和下一步。
- 能从 `docs/experiments/` 追溯具体实验。
