# LangGraph Router Graph 设计与 TODO

## 背景

当前项目已有 LangChain Tool Calling Agent、RAGService、MySQL 会话历史和 SSE 流式接口。接下来新增一个轻量但可扩展的 LangGraph Router Graph，用于统一调度用户请求。

Router Graph 的目标不是重写现有 RAG 或 Agent 链路，而是在现有能力之上增加一层可解释、可扩展的请求分流。

## 当前目标

第一版先新增非流式 Router 接口，不替换现有 SSE 链路。

核心目标：

- 使用 LangGraph `StateGraph` 构建请求调度图。
- 使用 LLM 做路由决策，但输出必须约束在固定枚举范围内。
- 顶层 route 只决定系统能力模块。
- RAG 内部再通过 `rag_intent` 和 `source_hints` 表示更细的检索意图。
- 底层复用现有 `RagService`、`AgentExecutor`、会话管理模块。
- 不破坏现有 `/api/agent/query/stream` 和 `/api/rag/query`。

## 总体流程

```text
START
  -> load_context
  -> llm_router
  -> validate_decision
  -> conditional route
       rag_query       -> rag_node
       agent_tool_call -> agent_node
       chat            -> chat_node
       system          -> system_node
       clarify         -> clarify_node
  -> persist_message
  -> format_response
END
```

## 顶层路由

`route` 负责决定用户请求交给哪个系统能力处理，不做过细业务分类。

允许值：

- `rag_query`：需要企业知识库、上传文档、内部资料、项目记录来回答。
- `agent_tool_call`：需要调用工具，例如时间、天气、用户信息、重排序等。
- `chat`：普通对话、开放问答、解释概念，不强依赖知识库。
- `system`：会话、文件、向量库、状态、权限、清理等系统操作。
- `clarify`：用户意图不明确，或缺少必要条件，需要反问。

## RAG 子意图

企业数据集中的问题类型不作为顶层 route，而作为 `rag_intent`。

当 `route=rag_query` 时，Router 额外输出：

- `basic`：事实型问题，通常精确检索即可。
- `semantic`：语义表达不直接，需要 HyDE 或语义扩展。
- `intra_document_reasoning`：单文档内部多信息点推理。
- `project_related`：项目、事故、工单、客户问题相关。
- `constrained`：带时间、地点、账号、版本、区域等强约束。
- `conflicting_info`：可能存在多个来源答案不一致。
- `completeness`：需要完整流程、清单、步骤或覆盖多个来源。
- `high_level`：使命、定位、整体介绍等高层问题。
- `info_not_found`：问题看似要求知识库，但资料中可能不存在。
- `unknown`：无法判断具体 RAG 子意图。

当 `route` 不是 `rag_query` 时，`rag_intent` 统一为 `unknown`。

## 数据源提示

`source_hints` 表示 Router 认为更可能相关的数据来源。

允许值：

- `confluence`
- `jira`
- `slack`
- `github`
- `google_drive`
- `linear`
- `gmail`
- `hubspot`
- `fireflies`

第一版可以只记录和返回 `source_hints`，暂不强过滤检索结果。后续再用于 metadata filter、source 加权、TopK 配额和评测分组。

## LLM 路由输出

Router LLM 只负责分类，不回答用户问题。

目标输出：

```json
{
  "route": "rag_query",
  "rag_intent": "project_related",
  "source_hints": ["jira", "slack", "confluence"],
  "confidence": 0.91,
  "reason": "用户询问项目事故、策略例外和验证方式，需要检索企业知识库。"
}
```

代码层必须二次校验：

- `route` 不在白名单内：兜底到 `chat` 或 `clarify`。
- `rag_intent` 不在白名单内：设为 `unknown`。
- `source_hints` 不在白名单内：过滤掉。
- JSON 解析失败：兜底到 `chat`。
- `confidence` 过低：走 `clarify`。

## GraphState

建议状态字段：

```python
class GraphState(TypedDict):
    query: str
    user_id: str
    session_id: str
    history: list[tuple[str, str]]

    route: str
    rag_intent: str
    source_hints: list[str]
    confidence: float
    reason: str

    answer: str
    documents: list
    steps: list
    error: str | None
```

## 节点职责

### load_context

读取或生成 `session_id`，并通过现有会话管理模块加载历史消息。

### llm_router

调用 LLM 生成路由决策。该节点不回答用户问题，只输出结构化分类结果。

### validate_decision

校验 LLM 输出中的枚举值、置信度、数据源提示和 JSON 格式，保证后续 Graph 只接收可控 route。

### rag_node

调用现有 `RagService().rag_summary(query)`。

第一版先不改 RAG 内部检索策略，但将 `rag_intent` 和 `source_hints` 传入状态、日志和响应中，为后续策略切换预留接口。

### agent_node

调用现有 `get_agent_response(query, history)`，复用 Tool Calling Agent。

### chat_node

第一版可以复用 `get_agent_response(query, history)`。后续可拆出无工具普通聊天链，避免普通聊天误触工具。

### system_node

处理系统类请求。第一版保持保守：删除、清空、重置等危险操作只返回确认提示，不直接执行。

### clarify_node

返回简短追问，引导用户补充目标、数据范围或上下文。

### persist_message

将用户问题和最终回答写入 MySQL 会话历史。若执行失败，记录日志并避免写入错误答案。

### format_response

统一输出接口响应。

## 新增接口

第一版新增非流式接口：

```text
POST /api/agent/router/query
```

请求体复用现有 `QueryRequest`：

```json
{
  "query": "xxx",
  "session_id": "optional"
}
```

响应结构：

```json
{
  "session_id": "...",
  "route": "rag_query",
  "rag_intent": "project_related",
  "source_hints": ["jira", "slack"],
  "confidence": 0.91,
  "reason": "...",
  "response": "...",
  "steps": []
}
```

现有接口暂时不动：

```text
POST /api/agent/query/stream
POST /api/rag/query
```

## TODO

### 第一阶段：Router Graph 骨架

- [x] 新增 `backend/app/agent/router_graph.py`。
- [x] 定义 `GraphState`。
- [x] 定义固定枚举：`route`、`rag_intent`、`source_hints`。
- [x] 实现 `RouteDecision` 结构化输出模型。
- [x] 实现 `load_context` 节点。
- [x] 实现 `llm_router` 节点。
- [x] 实现 `validate_decision` 节点。
- [x] 实现 `rag_node`，复用 `RagService().rag_summary(query)`。
- [x] 实现 `agent_node`，复用 `get_agent_response(query, history)`。
- [x] 实现 `chat_node`，第一版复用现有 Agent 链路。
- [x] 实现 `system_node`，危险操作只返回确认提示。
- [x] 实现 `clarify_node`。
- [x] 实现 `persist_message` 节点。
- [x] 实现 `format_response` 节点。
- [x] 使用 LangGraph `StateGraph` 串联所有节点和条件边。

### 第二阶段：新增接口

- [x] 在 `backend/app/schemas/models.py` 中补充 Router 响应 schema。
- [x] 在 `backend/app/router/chat_service.py` 中新增 Router 调用方法。
- [x] 在 `backend/app/router/chat.py` 中新增 `POST /api/agent/router/query`。
- [x] 保持现有 `/api/agent/query/stream` 不变。
- [x] 保持现有 `/api/rag/query` 不变。

### 第三阶段：验证

- [x] 跑 Python 语法/导入检查。
- [x] 验证 `rag_query` 能进入 RAG 节点。
- [x] 验证 `agent_tool_call` 能进入 Agent 节点。
- [x] 验证 `chat` 能进入普通聊天节点。
- [x] 验证 `system` 不直接执行危险操作。
- [x] 验证低置信度或解析失败时能兜底。
- [x] 验证接口返回 `route`、`rag_intent`、`source_hints`、`confidence`、`reason`。
- [x] 验证会话历史正常写入。

### 后续扩展

- [ ] 将 Router 接入 SSE 链路。
- [ ] 根据 `rag_intent` 调整 TopK、HyDE、BM25、Reranker 策略。
- [ ] 根据 `source_hints` 做 metadata filter 或 source 加权。
- [ ] 加入 clarification loop。
- [x] 在 `load_context` 阶段接入短期窗口 + 长期摘要记忆。
- [ ] 在离线评测中按 `rag_intent` 统计 Hit@K、Recall@K、MRR。

### EnterpriseRAG 接入

- [x] 新增 `backend/app/rag/enterprise_rag_service.py`。
- [x] 连接 `backend/data/chromadb_enterprise_parent_child`。
- [x] 读取 collection `enterprise_rag_bench_parent_child`。
- [x] 懒加载 `backend/data/enterprise_rag_bench/parent_chunks_parent_child.jsonl`。
- [x] 基于 child chunk 检索后回填 parent chunk 原文。
- [x] Router `rag_node` 切换为调用 `EnterpriseRagService`。
- [x] 验证 collection 可读取，当前 child chunks 数量为 `124353`。
- [x] 使用企业样例问题验证端到端回答可返回 `10MiB` 和 `50MiB`。
- [ ] 后续评估是否将 `source_hints` 从记录升级为软加权或 metadata filter。

## 更新记录

- 2026-05-14：创建 LangGraph Router Graph 设计文档和 TODO 清单。
- 2026-05-14：完成 Router Graph 骨架、新增非流式接口、补充依赖声明，并完成语法/导入、system 路由和低置信度兜底验证。
- 2026-05-14：启动 Redis 和 FastAPI，使用临时 JWT 完成 `/api/agent/router/query` HTTP 联调；验证 `system`、`rag_query`、`agent_tool_call`、`chat` 路径和 MySQL 会话写入，测试会话已清理。
- 2026-05-14：新增 `EnterpriseRagService` 读取企业 parent-child Chroma 库，Router `rag_node` 已切换到企业 RAG；验证企业样例问题能从库中检索并回答默认上传限制 `10MiB/50MiB`。
- 2026-05-14：RouterGraph `load_context` 已接入两层会话记忆，读取 Session Memory 摘要 + Working Memory 最近 6 轮原文。

## 当前联调备注

- FastAPI 当前运行在 `http://127.0.0.1:8000`。
- Redis 当前运行在 `localhost:6379`。
- `rag_query` 当前已接入 `EnterpriseRagService`，读取 `backend/data/chromadb_enterprise_parent_child`。
- `source_hints` 目前只作为记录和响应字段，不做硬过滤。联调发现 Router 可能将 API 上传限制问题的来源误判为 `confluence`，而正确文档在 `github`；如果将 hint 直接作为 metadata filter，会过滤掉正确答案。
- RouterGraph 非流式接口已完成第一版闭环，但尚未接入 SSE 流式返回；当前 SSE 仍走原 `/api/agent/query/stream`，只是历史读取已接入两层记忆。
