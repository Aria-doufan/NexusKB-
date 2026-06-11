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

1. 接入更可靠的 `decision_chain`，让 `understand_request` 能根据用户问题真实判断 action。
2. 稳定 `RagState.sse_events` 的事件命名和前端展示。
3. 用离线评测继续调参 `StrategyRouter` 的 topK、reranker 和 decompose 策略。
4. 完善证据不足、web fallback、debug trace 和前端引用展示。
5. 保持工具调用边界：工具只能由用户请求和受控 action 触发，不能由检索文档触发。

## 当前项目实际流程对比

### 当前流式聊天入口

当前 `/api/agent/query/stream` 的真实流程是：

```text
POST /api/agent/query/stream
  -> get_current_user_id
  -> rate_limit
  -> router_graph.stream(query, user_id, session_id)
      -> AgenticRagGraph.invoke()
          -> load_context: conversation memory + long-term memory
          -> understand_request
          -> safety_check
          -> direct_answer / retrieve / tool_call / clarify / refuse
          -> finalize_trace
          -> persist_message
      -> route 兼容事件
      -> RagState.sse_events
      -> response
      -> done
```

当前特点：

- 流式聊天入口已经经过 `RouterGraph` 兼容层。
- `RouterGraph` 不再拥有旧多分支图，主状态机由 `AgenticRagGraph` 拥有。
- 普通直接回答、RAG、工具调用、澄清和拒绝由 action 分支隔离。
- 会话压缩记忆和长期记忆由 `AgenticRagGraph.load_context()` 统一加载。

当前差距：

| 项目 | 当前实际 | 后续目标 |
| --- | --- | --- |
| 流式入口是否经过 Router | 是，经过兼容入口 | 保持兼容并减少重复字段映射 |
| 普通聊天是否复用完整 Tool Agent | 否，直接回答和工具调用分支分离 | 用真实 decision_chain 提升 action 判断准确率 |
| 工具是否分类 | 由 `AGENTIC_RAG_TOOLS` 和 `AgenticToolRunner` 控制 | 增加更细风险分类和审计 |
| RAG 是否克制调用 | 由 `needs_retrieval/action=retrieve` 控制 | 用评测优化误检索/漏检索 |
| 会话记忆 | 已接入会话摘要、最近历史和长期记忆 | 继续评测长期记忆相关性和安全注入边界 |

### 当前 Router / Agentic RAG 入口

当前 `/api/agent/router/query` 与 `/api/agent/query/stream` 都通过 `RouterGraph` 兼容入口进入 `AgenticRagGraph`：

```text
POST /api/agent/router/query 或 /api/agent/query/stream
  -> get_current_user_id
  -> rate_limit
  -> router_graph.invoke() / router_graph.stream()
      -> AgenticRagGraph.invoke()
          -> load_context: 会话压缩记忆 + 长期记忆
          -> understand_request: AgenticActionDecision
          -> safety_check
          -> action 分支
              direct_answer -> 直接回答
              retrieve      -> RagEvidenceWorkflow
              tool_call     -> AgenticToolRunner
              clarify       -> 澄清反问
              refuse        -> 安全拒绝
          -> finalize_trace
          -> persist_message: 成功回答写入会话历史
      -> 兼容 RouterResponse / SSE events
```

当前特点：

- `RouterGraph` 不再拥有旧多分支 LangGraph，只做兼容包装。
- `AgenticRagGraph` 是单一主状态机，避免 RouterGraph 与 EnterpriseRagGraph 双重拥有流程。
- 企业知识问题进入 `RagEvidenceWorkflow`，由 planner、strategy、retrieval、evaluation、retry 和 generation 串联。
- 工具调用通过 `AgenticToolRunner` 执行受控工具，不再让普通聊天默认持有完整工具池。
- 澄清和拒绝成为 `action` 分支，而不是旧 route 节点。

当前 MVP 差距已经从“路由是否接入”转为“策略质量与可观测性是否足够”：

| 项目 | 当前实际 | 后续目标 |
| --- | --- | --- |
| 主入口统一 | 已统一到 RouterGraph 兼容入口 + AgenticRagGraph | 保持旧 API 兼容并减少重复包装 |
| chat / RAG / tool 边界 | 已用 action 分支表达 | 用真实 decision_chain 替换默认保守决策 |
| 企业 RAG 独立性 | 已由 RagEvidenceWorkflow 拥有 | 继续完善 debug、评测和前端引用展示 |
| clarify / refuse | 已有固定分支 | 让澄清问题更具体、拒绝原因更可审计 |

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

## 当前实现状态

更新时间：2026-06-11

### P0：工具池隔离

当前工具边界已经迁移到 `AgenticRagGraph` 的 action 分支：

- 普通直接回答走 `direct_answer`，不默认创建完整 Tool Agent。
- 工具类请求走 `tool_call`，由 `AgenticToolRunner` 执行 `AGENTIC_RAG_TOOLS` 中的受控工具。
- 企业知识库问题走 `retrieve`，不通过普通聊天工具池触发检索。

### P1：Router SSE 化

当前流式入口：

```text
POST /api/agent/query/stream
  -> get_current_user_id
  -> rate_limit
  -> router_graph.stream()
      -> AgenticRagGraph.invoke()
      -> 兼容 route 事件: route = agentic_rag
      -> RagState.sse_events
      -> response
      -> done
```

`RouterGraph.stream()` 不再手动串联旧 `load_context -> router -> validate -> branch`，而是统一委托 `AgenticRagGraph.invoke()` 后包装 SSE。

### P2：直接回答 / 工具 / RAG 边界

当前边界：

| action | 执行链路 |
| --- | --- |
| `direct_answer` | `direct_answer_node` 生成不检索企业库的直接回答 |
| `retrieve` | `RagEvidenceWorkflow` 生成有证据回答或证据不足响应 |
| `tool_call` | `AgenticToolRunner` 执行受控工具 |
| `clarify` | `clarify_node` 返回澄清问题 |
| `refuse` | `refuse_node` 返回安全拒绝 |

### P3：企业知识链路独立

当前企业知识链路由 `RagEvidenceWorkflow` 拥有：

```text
planner
  -> strategy_select
  -> retrieve / retrieve_decomposed
  -> evaluate_context
  -> decide_next_action
  -> rewrite_query / expand_top_k retry
  -> external search fallback
  -> generate_answer 或 build_insufficient_evidence
  -> finalize_trace
```

`EnterpriseRagGraph` 只保留兼容包装，内部委托 `RagEvidenceWorkflow`。

### 当前后续重点

- 接入更可靠的 `decision_chain`，让 `AgenticActionDecision` 不依赖默认状态。
- 稳定 SSE 事件协议，减少前端对兼容 `response` 事件的依赖。
- 继续用评测数据调参 `StrategyRouter`。
- 完善证据不足、web fallback 和 debug trace 的可视化。
