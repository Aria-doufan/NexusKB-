# Chat Agent MVP 计划

## 当前阶段定位

当前阶段先做：

```text
Chat Agent MVP
```

目标不是复杂多 Agent，也不是把所有问题都推给 RAG，而是先把聊天 Agent 做稳：

- 普通聊天体验稳定。
- 多轮上下文自然。
- 流式输出稳定。
- 回答风格一致。
- 会话记忆能接住上下文。
- 意图识别服务真实场景，不炫技。
- 知识库调用克制，只在企业内部知识问题中使用。
- 工具调用可控，只调用适合当前场景的工具。

## 当前核心原则

### 1. 普通聊天体验稳定

需要做到：

- 不乱调用工具。
- 多轮上下文自然。
- 流式输出稳定。
- 回答风格一致。
- 会话记忆能帮它接住上下文。

### 2. 意图识别服务于场景

意图分类不按技术炫技，而按员工实际使用场景设计。

| 场景 | 路由 |
| --- | --- |
| 员工闲聊、解释概念、写文案、总结、改写 | `chat` |
| 问公司资料、制度、历史记录、项目细节 | `enterprise_knowledge` |
| 要查系统状态、执行动作、调用外部/内部 API | `tool_action` |
| 问得模糊、指代不清、缺少范围 | `clarify` |
| 删除、清空、越权、危险系统操作 | `unsafe_or_system` |

### 3. 知识库调用要克制

原则：

- 不是所有问题都 RAG。
- 不是所有 RAG 都 reranker。
- 先判断“这个问题是不是企业内部知识问题”。
- 普通知识解释、写作、总结不应该默认查企业知识库。

例子：

| 用户问题 | 预期 |
| --- | --- |
| 解释一下 RAG 是什么 | `chat` |
| 帮我把这段话改正式一点 | `chat` |
| 我们公司上传文件限制是多少 | `enterprise_knowledge` |
| 上次我们说下一步做什么 | `chat` + session memory |
| 查一下系统状态 | `tool_action` |
| 那个限制在哪里配置 | 有上下文则 `enterprise_knowledge`，无上下文则 `clarify` |
| 清空数据库 | `unsafe_or_system` |

## 普通聊天与工具调用的关系

普通聊天不应该复用完整 Tool Agent，否则容易误触企业工具或危险工具。

但普通聊天也不应该完全禁止工具。真实助手里，有些轻量工具是合理的，例如：

- 查询天气。
- 查询时间。
- 简单计算。
- 发送邮件草稿或发送邮件。
- 日历查询。

因此当前结论是：

```text
普通聊天可以调用工具，但只能调用少部分安全、低风险、用户意图明确的工具。
```

## 工具分类方案

建议把工具分成四类。

| 工具类别 | 说明 | 普通聊天是否可用 | 是否需要确认 |
| --- | --- | --- | --- |
| `safe_utility` | 安全小工具，例如时间、天气、简单计算、格式转换 | 可以 | 不需要 |
| `personal_action` | 用户个人工具，例如发送邮件、创建日历、生成待办 | 可以，但必须意图明确 | 写操作需要确认 |
| `enterprise_read` | 企业只读查询，例如查工单、查项目状态、查部署记录 | 不走普通 chat，由 Router 判为 `tool_action` | 通常不需要，但要权限校验 |
| `enterprise_write_or_dangerous` | 企业写操作或危险操作，例如删除、清库、改配置、发公告 | 不走普通 chat | 必须确认，必要时拒绝 |

### safe_utility

适合普通聊天直接调用。

例子：

- 获取当前时间。
- 查询天气。
- 简单数学计算。
- 文本格式转换。

特点：

- 不接触企业敏感数据。
- 不修改外部系统。
- 调用失败影响小。

### personal_action

可以由普通聊天触发，但必须谨慎。

例子：

- 发送邮件。
- 创建日历事件。
- 创建个人待办。

规则：

- 用户意图必须明确。
- 涉及发送、创建、修改时必须先给用户确认。
- 默认可以先生成草稿，不直接发送。

### enterprise_read

企业只读工具不应该由普通 chat 随意调用，而应该由 Router 明确判定为 `tool_action`。

例子：

- 查询 Jira 工单。
- 查询部署状态。
- 查询客户记录。
- 查询内部系统状态。

规则：

- 必须做用户身份和权限校验。
- 工具结果可以作为回答依据。
- 需要记录 tool trace。

### enterprise_write_or_dangerous

高风险工具。

例子：

- 删除数据。
- 清空向量库。
- 修改生产配置。
- 发布公告。
- 关闭服务。

规则：

- 普通聊天不可调用。
- 必须走 `unsafe_or_system` 或受控 `tool_action`。
- 必须二次确认。
- 没有权限时直接拒绝。
- 必须审计。

## 简化路由设计

当前阶段建议路由只保留五类：

```text
chat
enterprise_knowledge
tool_action
clarify
unsafe_or_system
```

工具选择不再只靠一个大的 Tool Agent 自由决定，而是：

```text
Router 判断场景
  -> chat：纯聊天 + safe_utility + 明确 personal_action
  -> enterprise_knowledge：企业知识库
  -> tool_action：企业工具或个人工具
  -> clarify：追问
  -> unsafe_or_system：拒绝或确认
```

## 当前 MVP 验收标准

| 测试问题 | 期望 |
| --- | --- |
| 解释一下 RAG 是什么 | 走 `chat`，不查企业知识库，不调用企业工具 |
| 帮我写一段日报 | 走 `chat`，可使用会话上下文 |
| 今天香港天气怎么样 | 走 `chat` 可调用 `safe_utility` 天气工具 |
| 给张三发邮件说明会议延期 | 识别为 `personal_action`，先生成邮件草稿并请求确认 |
| 我们公司上传文件限制是多少 | 走 `enterprise_knowledge` |
| 上次我们决定下一步做什么 | 走 `chat` + session memory |
| 查一下项目 X 的最新工单状态 | 走 `tool_action` + 权限校验 |
| 清空数据库 | 走 `unsafe_or_system`，不直接执行 |

## 暂时不做

- Workspace Memory。
- 完整企业权限系统。
- 复杂长期记忆自动沉淀。
- source-aware retrieval。
- 复杂 RAG 策略矩阵。
- 所有工具都开放给普通聊天 Agent。

## 下一步建议

1. 拆出纯聊天链。
2. 给工具增加分类 metadata。
3. 普通 chat 只允许调用 `safe_utility` 和明确的 `personal_action`。
4. Router SSE 化，让流式主入口也能走简化路由。
5. 企业知识库继续保持克制调用。

## 当前项目实际流程对比

### 当前流式聊天入口

当前 `/api/agent/query/stream` 的真实流程是：

```text
POST /api/agent/query/stream
  -> get_current_user_id
  -> rate_limit
  -> get_agent_stream_response(query, session_id, user_id)
      -> conversation_memory_service.get_history_for_agent()
      -> agent_factory.create_agent_executor()
          -> 默认工具全集
             - rag_summary_tools
             - get_weather_tools
             - what_time_is_now
             - get_user_info_tools
             - reorder_documents_tools
      -> AgentExecutor.astream()
      -> 写入 chat_messages
      -> update_memory()
      -> SSE done
```

当前特点：

- 流式聊天入口没有经过 Router。
- 默认使用完整 Tool Agent。
- 普通聊天、天气、时间、RAG、重排序、用户信息工具都在同一个工具池里。
- 会话记忆已经接入，能读取 Session Memory + Working Memory。

和 MVP 目标的差距：

| 项目 | 当前实际 | MVP 目标 |
| --- | --- | --- |
| 流式入口是否经过 Router | 否 | 是 |
| 普通聊天是否复用完整 Tool Agent | 是 | 否 |
| 工具是否分类 | 否 | 是 |
| 普通聊天可用工具范围 | 默认全部工具 | 只允许 `safe_utility` 和明确 `personal_action` |
| RAG 是否克制调用 | 依赖 Agent 自己判断 | Router 判断企业知识场景后再调用 |
| 会话记忆 | 已接入 | 保留并优化 |

### 当前非流式 Router 入口

当前 `/api/agent/router/query` 的真实流程是：

```text
POST /api/agent/router/query
  -> get_current_user_id
  -> rate_limit
  -> router_graph.invoke()
      -> load_context
      -> llm_router
      -> validate_decision
      -> conditional route
          rag_query       -> enterprise_rag_service
          agent_tool_call -> get_agent_response()
          chat            -> get_agent_response()
          system          -> 保守提示，不直接执行
          clarify         -> 追问
      -> persist_message
      -> format_response
```

当前特点：

- 非流式 Router 已经有意图识别。
- `rag_query` 会进入 `EnterpriseRagService`。
- `system` 路由不会直接执行危险操作。
- 但 `chat_node` 和 `agent_node` 都调用同一个 `get_agent_response()`。
- `get_agent_response()` 默认仍然创建完整 Tool Agent。

和 MVP 目标的差距：

| 项目 | 当前实际 | MVP 目标 |
| --- | --- | --- |
| Router 是否存在 | 已有 | 保留并简化场景分类 |
| chat 与 tool_action 是否分离 | 没有，最终都走完整 Agent | 必须分离 |
| chat 是否可以调用安全工具 | 可以，但没有限制范围 | 只允许安全小工具和明确个人工具 |
| enterprise_knowledge 是否独立 | 非流式中已独立 | 流式中也要独立 |
| clarify 是否存在 | 已有简单追问 | 按场景生成更具体追问 |

### 当前默认工具池

当前 `AgentFactory._get_default_tools()` 返回：

```text
rag_summary_tools
get_weather_tools
what_time_is_now
get_user_info_tools
reorder_documents_tools
```

按 MVP 工具分类，建议重新标记为：

| 工具 | 当前用途 | 建议类别 | 普通 chat 是否可用 | 备注 |
| --- | --- | --- | --- | --- |
| `get_weather_tools` | 查询天气 | `safe_utility` | 可以 | 低风险 |
| `what_time_is_now` | 查询当前时间 | `safe_utility` | 可以 | 低风险 |
| `rag_summary_tools` | 原业务知识库 RAG | `enterprise_read` 或 `knowledge_retrieval` | 不应默认给 chat | 应由 Router 判断知识场景后调用 |
| `reorder_documents_tools` | 文档重排序 | `internal_processing` | 不应默认给 chat | 更像内部检索流程组件，不该暴露给普通聊天 |
| `get_user_info_tools` | 从 JWT 获取用户信息 | `personal_read` | 谨慎 | 当前要求用户提供完整 JWT，不适合自然聊天工具 |

当前最明显的问题：

```text
普通聊天和工具调用没有工具池隔离。
```

这会导致：

- 普通解释类问题可能误触 RAG。
- 普通聊天可能调用重排序这类内部工具。
- `chat_node` 和 `agent_node` 名义上不同，执行上却几乎一样。

## 目标流程

### MVP 目标主流程

建议目标流程：

```text
POST /api/agent/query/stream
  -> get_current_user_id
  -> rate_limit
  -> load memory
  -> lightweight router
      -> chat
          -> pure chat model
          -> optional safe_utility tools
      -> enterprise_knowledge
          -> EnterpriseRagService / EnterpriseRetrievalService
      -> tool_action
          -> tool agent with scoped tool set
      -> clarify
          -> specific clarification question
      -> unsafe_or_system
          -> refuse or ask confirmation
  -> stream response
  -> persist message
  -> update memory
```

### 目标工具池拆分

建议拆成三个工具池：

```text
CHAT_SAFE_TOOLS
  - get_weather_tools
  - what_time_is_now

PERSONAL_ACTION_TOOLS
  - send_email_draft
  - calendar_query
  - create_todo

ENTERPRISE_TOOLS
  - jira_query
  - deployment_status_query
  - enterprise_rag_search
```

当前项目里可以先做最小拆分：

```text
CHAT_SAFE_TOOLS
  - get_weather_tools
  - what_time_is_now

FULL_AGENT_TOOLS
  - rag_summary_tools
  - get_weather_tools
  - what_time_is_now
  - get_user_info_tools
  - reorder_documents_tools
```

第一步先让普通 `chat` 只拿 `CHAT_SAFE_TOOLS`，把 RAG 和重排序从普通聊天默认工具池里拿掉。

## 改造优先级

### P0：工具池隔离

目标：

- `chat_node` 不再调用完整 Tool Agent。
- 普通聊天最多使用 `CHAT_SAFE_TOOLS`。
- `agent_tool_call` 才使用更大的工具池。

建议代码改动：

- 在 `agent_tools.py` 定义工具分类列表。
- 在 `AgentFactory` 支持不同 tool profile。
- `router_graph.chat_node` 调用 chat 专用链或 chat-safe agent。
- `router_graph.agent_node` 调用 full/scoped tool agent。

### P1：Router SSE 化

目标：

- `/api/agent/query/stream` 也先经过 Router。
- 保持流式体验。
- chat / enterprise_knowledge / tool_action 不再走同一条链。

### P2：纯聊天链

目标：

- 普通聊天不依赖 Agent scratchpad。
- 回答风格稳定。
- 工具只作为可选安全增强，不是默认自由选择。

### P3：企业知识库独立链路

目标：

- 企业知识问题进入 RAG 链路。
- 普通 chat 不直接看到 `rag_summary_tools`。
- 后续再把 BM25/Reranker 产品化。

## 实施记录：P0 到 P3 已完成

更新时间：2026-05-14

### P0：工具池隔离

已完成：

- 在 `backend/app/agent/agent_tools.py` 中拆出工具池：
  - `CHAT_SAFE_TOOLS`
    - `get_weather_tools`
    - `what_time_is_now`
  - `FULL_AGENT_TOOLS`
    - `rag_summary_tools`
    - `get_weather_tools`
    - `what_time_is_now`
    - `get_user_info_tools`
    - `reorder_documents_tools`
- 在 `backend/app/agent/agent.py` 中为 `AgentFactory` 增加 `tool_profile` 支持：
  - `chat_safe`
  - `full`
- 默认完整 Agent 仍保持 `full` 工具池，兼容旧调用。
- 修复 `what_time_is_now()` 中 `strftime` 字符串引号冲突问题。

效果：

- 普通聊天不再默认拿到 RAG、用户信息、重排序等工具。
- 需要工具动作的链路才使用完整工具池。

### P1：Router SSE 化

已完成：

- 在 `backend/app/agent/router_graph.py` 中新增 `RouterGraph.stream()`。
- `/api/agent/query/stream` 已从直连 `get_agent_stream_response()` 改为先走 `router_graph.stream()`。
- 流式响应会先输出 `type=route` 事件，再按路由进入对应链路。

当前流式主入口：

```text
POST /api/agent/query/stream
  -> get_current_user_id
  -> rate_limit
  -> router_graph.stream()
      -> load_context
      -> llm_router
      -> validate_decision
      -> route event
      -> chat / enterprise_knowledge / tool_action / clarify / unsafe_or_system
```

效果：

- 流式入口也开始走 Router。
- `chat`、企业知识、工具动作、危险系统请求不再混在同一条默认 Agent 链路里。

### P2：纯聊天链

已完成：

- 在 `backend/app/agent/agent.py` 中新增 `PURE_CHAT_SYSTEM_PROMPT`。
- 新增不带 `agent_scratchpad` 的纯聊天链：
  - `AgentFactory.create_chat_chain()`
  - `get_chat_response()`
  - `get_chat_stream_response()`
- Router 的 `chat_node` 和流式 `chat` 分支已切到纯聊天链。
- 天气、当前时间作为显式 safe utility 增强：
  - 明确问天气时，代码显式调用 `get_weather_tools`。
  - 明确问当前时间/日期时，代码显式调用 `what_time_is_now`。
  - 工具结果作为上下文注入纯聊天链，而不是交给 Tool Agent 自由选择。

效果：

- 普通解释、写作、总结、改写、上下文聊天不再依赖 AgentExecutor。
- 普通 chat 不再使用旧的“优先 RAG”主提示词。
- safe utility 保留，但调用边界更清晰。

### P3：路由命名统一与企业知识链路独立

已完成：

- Router route 统一为 MVP 文档中的五类：
  - `chat`
  - `enterprise_knowledge`
  - `tool_action`
  - `clarify`
  - `unsafe_or_system`
- 旧 route 名称保留兼容映射：
  - `rag_query` -> `enterprise_knowledge`
  - `agent_tool_call` -> `tool_action`
  - `system` -> `unsafe_or_system`
- LangGraph 节点命名同步：
  - `enterprise_knowledge_node`
  - `tool_action_node`
  - `unsafe_or_system_node`
- `enterprise_knowledge_node` 继续独立调用 `EnterpriseRagService`。
- `tool_action_node` 继续走完整工具 Agent。
- `unsafe_or_system_node` 只返回保守提示，不直接执行危险操作。

效果：

- 对外返回和内部路由都使用 MVP 统一命名。
- 企业知识问题进入独立 RAG 链路。
- 普通 chat 不直接看到 `rag_summary_tools`。
- 危险系统请求不会直接执行。

### 已验证

- `py_compile` 通过。
- Router、流式入口、纯聊天链导入通过。
- SSE 烟测通过：
  - `route -> response -> done`
- safe utility 显式上下文测试通过。
- 旧 route 名称兼容测试通过：
  - `rag_query` 输出为 `enterprise_knowledge`
  - `agent_tool_call` 输出为 `tool_action`
  - `system` 输出为 `unsafe_or_system`
