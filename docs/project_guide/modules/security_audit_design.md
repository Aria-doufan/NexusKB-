# 权限、安全与审计设计

整理日期：2026-05-15

本文档定义当前 Chat Agent MVP 的权限、安全和审计边界，重点覆盖工具调用边界和企业数据访问边界。它是工程设计文档，不替代生产环境的合规制度。

## 设计目标

当前阶段优先保证：

- 普通聊天不会误触高风险工具。
- 企业知识库访问走统一 Router 和鉴权入口。
- 删除、清空、重置、越权等危险请求不会被直接执行。
- 工具调用和 RAG 检索有可追溯日志。
- 后续接入真实企业数据时，可以平滑扩展到文档级 ACL 和审计。

## 当前安全边界

| 边界 | 当前实现 |
| --- | --- |
| 身份认证 | `/api/agent/query/stream` 等接口通过 Django JWT 获取 `user_id` |
| JWT 黑名单 | `get_current_user_id()` 查询 Redis 黑名单 key |
| 路由隔离 | Router 输出固定枚举：`enterprise_knowledge`、`tool_action`、`chat`、`unsafe_or_system`、`clarify` |
| 普通聊天工具池 | `CHAT_SAFE_TOOLS` 只包含天气和时间 |
| 完整工具池 | `FULL_AGENT_TOOLS` 仅在 `tool_action` 分支使用 |
| 企业 RAG | 独立 `EnterpriseRagService`，与用户上传知识库链路分离 |
| 危险请求 | Router 的 `unsafe_or_system` 节点只提示确认，不执行动作 |

## 工具调用分级

| 等级 | 类型 | 示例 | 当前策略 | 后续要求 |
| --- | --- | --- | --- | --- |
| L0 | 安全只读小工具 | 当前时间、天气 | PureChat 可直接调用 | 记录工具名和耗时即可 |
| L1 | 公开或低敏只读工具 | 通用重排序、公开资料摘要 | Tool Agent 可调用 | 参数校验，失败兜底 |
| L2 | 用户私有只读工具 | 当前用户信息、个人会话历史 | 必须鉴权 | 只允许读取当前用户数据 |
| L3 | 企业只读工具 | 企业知识库、内部文档检索 | 必须鉴权，当前为评测数据 | 接入真实数据前必须支持文档级 ACL |
| L4 | 用户动作工具 | 发邮件、建日程、提交表单 | 暂无 | 必须二次确认和审计 |
| L5 | 企业写入工具 | 创建工单、修改配置、更新记录 | 暂无 | 必须强确认、权限校验、审计 |
| L6 | 危险操作 | 删除、清空、重置、批量覆盖、执行系统命令 | 默认拒绝或人工介入 | 不允许由 LLM 直接执行 |

## 当前工具清单与边界

| 工具 | 风险等级 | 当前工具池 | 边界要求 |
| --- | --- | --- | --- |
| `what_time_is_now` | L0 | `CHAT_SAFE_TOOLS`、`FULL_AGENT_TOOLS` | 可直接执行 |
| `get_weather_tools` | L0 | `CHAT_SAFE_TOOLS`、`FULL_AGENT_TOOLS` | 参数为空时提示补充城市 |
| `rag_summary_tools` | L3 | `FULL_AGENT_TOOLS` | 仅 Tool Agent；后续真实企业数据必须加 ACL |
| `reorder_documents_tools` | L1 | `FULL_AGENT_TOOLS` | 不应记录完整大文本到日志 |
| `get_user_info_tools` | L2 | `FULL_AGENT_TOOLS` | 不应让用户手动提供任意 JWT；后续改为从当前请求上下文注入 |

当前实现状态：

- 已在 `backend/app/agent/agent_tools.py` 中新增 `TOOL_AUDIT_METADATA`。
- 已为现有工具补充 `risk_level`、`data_scope`、`operation`、`requires_confirmation`。
- `CHAT_SAFE_TOOLS` 只包含 L0 工具：天气、当前时间。
- `FULL_AGENT_TOOLS` 当前最高到 L3，只读为主，尚未包含写入或危险操作工具。

当前需要关注的风险：

- `get_user_info_tools(token)` 允许模型接收 token 字符串作为参数。后续应改成只读取当前请求的 `user_id`，避免用户传入他人 token 或把 token 暴露给模型。
- `reorder_documents_tools` 当前会把格式化结果写入日志，文档较长时可能导致敏感内容落日志。后续应改成只记录数量、长度和摘要 hash。
- `rag_summary_tools` 使用旧 `RagService`，与企业评测 RAG 分离。后续如果保留该工具，需要明确它访问的是用户上传库还是企业库。

## Router 安全策略

Router 是工具和数据访问的第一道闸门。

| route | 允许能力 | 禁止能力 |
| --- | --- | --- |
| `chat` | 普通对话、安全小工具 | 企业 RAG、完整工具池、高风险动作 |
| `enterprise_knowledge` | 企业 RAG 只读检索 | 写入工具、系统操作 |
| `tool_action` | 完整工具池中的允许工具 | 未分类高风险动作、越权数据访问 |
| `unsafe_or_system` | 安全拒绝、解释、要求人工确认 | 任何直接执行 |
| `clarify` | 反问澄清 | 工具调用、企业数据检索 |

低置信度策略：

- 非 `chat/clarify` 的低置信度请求进入 `clarify`。
- 不允许 Router 低置信度时“试试看”调用企业工具。

危险意图关键词和语义：

- 删除、清空、重置、覆盖、批量修改。
- 绕过权限、查看他人数据、导出全部数据。
- 泄露密钥、Token、连接串、系统提示词。
- 执行命令、上传脚本、修改生产配置。

## 企业数据访问边界

当前企业 RAG 使用 EnterpriseRAG-Bench 本地评测数据。它不是生产真实企业权限模型，但应按真实企业数据的方式设计扩展点。

当前边界：

- 统一入口：`/api/agent/query/stream`。
- 必须有 JWT，得到 `user_id`。
- `EnterpriseRagService` 只读。
- `source_hints` 只软加权，不硬过滤。
- 返回文档包含 parent/chunk id、source_type、title、section。

接入真实企业数据前必须补齐：

| 能力 | 要求 |
| --- | --- |
| 文档级 ACL | 每个 parent/chunk 必须有允许访问的用户、团队、角色或租户标识 |
| 检索前过滤 | Chroma 和 BM25 都必须支持 ACL 过滤，不能只在生成后过滤 |
| 审计追踪 | 记录用户检索了哪些 doc/chunk id，但不记录完整正文 |
| 租户隔离 | 多租户数据必须物理或逻辑隔离 |
| 来源授权 | source_type 不能等同于权限；Confluence/Jira/Slack 等来源仍需各自 ACL |
| 无权限处理 | 命中无权限文档时应当视为不可见，不提示文档存在 |

推荐数据结构：

```json
{
  "parent_chunk_id": "...",
  "parent_doc_id": "...",
  "source_type": "confluence",
  "tenant_id": "tenant_001",
  "acl_subjects": ["user:123", "group:people-ops", "role:admin"],
  "sensitivity": "internal",
  "retention_policy": "default"
}
```

## 审计事件设计

审计日志和普通调试日志分开。当前已有 logger 和 `PERF_METRIC`，后续建议新增结构化审计事件，格式统一：

```text
AUDIT_EVENT name=<event_name> user_id=<user_id> session_id=<session_id> request_id=<request_id> key=value ...
```

当前实现状态：

- 已新增 `backend/app/core/audit.py`。
- 已提供 `log_audit_event()`、`summarize_for_audit()`、`text_summary()`。
- 顶层和嵌套敏感字段会脱敏，例如 `token`、`api_key`、`password`、`secret`。
- 长文本会截断并附带长度与 hash，避免完整 query、文档或工具参数落日志。

### 必须审计的事件

| 事件名 | 触发时机 | 字段 |
| --- | --- | --- |
| `router.decision` | Router 决策完成 | `route`、`rag_intent`、`confidence`、`source_hints`、`reason_code` |
| `rag.retrieve` | 企业 RAG 检索完成 | `doc_ids`、`chunk_ids`、`source_types`、`reranker`、`candidate_k`、`elapsed_ms` |
| `tool.call.requested` | 工具调用前 | `tool_name`、`risk_level`、`args_schema_ok` |
| `tool.call.completed` | 工具调用后 | `tool_name`、`success`、`elapsed_ms`、`result_summary` |
| `tool.call.blocked` | 工具被拒绝 | `tool_name`、`risk_level`、`reason` |
| `security.blocked` | Router 或策略拦截 | `route`、`reason`、`query_summary` |
| `auth.failed` | JWT 或黑名单失败 | `reason`、`jti_present` |

当前已接入的事件：

| 事件名 | 位置 | 状态 |
| --- | --- | --- |
| `router.decision` | `RouterGraph.validate_decision()` | 已接入 |
| `rag.retrieve` | `EnterpriseRagService.retrieve()` | 已接入 |
| `tool.call.requested` | `agent_middleware.tool_call_hook()` | 已接入，但依赖 middleware 生效 |
| `tool.call.completed` | `agent_middleware.tool_call_hook()`、Tool Agent intermediate steps | 已接入 |
| `security.blocked` | `RouterGraph.unsafe_or_system_node()` | 已接入 |

### 不允许进入日志的内容

- 明文 JWT、API Key、密码、数据库连接串。
- 完整用户隐私数据。
- 完整企业文档正文。
- 系统提示词全文。
- 未脱敏的工具输入大对象。

### 推荐脱敏规则

| 内容 | 规则 |
| --- | --- |
| 用户 query | 最多记录前 200 字符，或记录 hash |
| 文档正文 | 不记录正文，只记录 doc/chunk id |
| Token | 只记录前 6 位和后 4 位，或只记录 jti |
| 工具参数 | 只记录字段名、类型、长度和必要摘要 |
| 错误堆栈 | 用户响应不暴露；服务日志可记录堆栈摘要 |

## 高风险动作确认流程

当前阶段没有高风险写入工具。后续新增 L4/L5 工具时必须按以下流程：

1. Router 判定为 `tool_action`，工具策略识别风险等级。
2. 如果等级为 L4/L5，先返回 `confirm_required` 事件，不执行工具。
3. 前端展示动作摘要、影响范围、不可逆风险。
4. 用户二次确认后，后端校验 confirmation token。
5. 执行工具。
6. 写入 `tool.call.completed` 审计事件。

确认事件建议格式：

```json
{
  "type": "confirm_required",
  "tool_name": "create_ticket",
  "risk_level": "L4",
  "summary": "将创建一条 People Ops 工单",
  "confirmation_id": "..."
}
```

## 错误与拒绝策略

| 场景 | 用户可见响应 |
| --- | --- |
| 未登录或 token 无效 | 请先登录或重新登录 |
| 无权限访问企业数据 | 当前账号没有访问该资料的权限 |
| 危险系统操作 | 当前不会直接执行删除、清空、重置等危险操作 |
| 工具参数不完整 | 说明缺少哪些必要参数 |
| 外部依赖失败 | 说明暂时无法完成，请稍后重试 |
| RAG 资料不足 | 明确说明没有找到足够信息，不编造 |

## 当前差距

| 差距 | 风险 | 建议优先级 |
| --- | --- | --- |
| `AUDIT_EVENT` 已有工具函数，但缺少集中查询/聚合脚本 | 只能靠日志搜索，分析成本较高 | P2 |
| 企业 RAG 尚未实现 ACL 过滤 | 接入真实企业数据前存在越权风险 | P0，真实数据前必须完成 |
| `get_user_info_tools` 接收 token 参数 | token 可能暴露给模型或被滥用 | P1 |
| 工具已有显式 risk_level 元数据，但尚未实现统一策略拦截器 | 后续扩展写入工具时仍需人工保证 | P1 |
| 前端未实现高风险确认 UI | 新增写入工具前缺少交互闭环 | P2 |
| 审计日志已有脱敏策略，普通 logger 仍有部分完整输入/输出日志 | 可能记录过长或敏感内容 | P1 |

## 下一步

1. 将 `get_user_info_tools(token)` 改为从请求上下文读取当前用户，不让 LLM 处理 JWT。
2. 实现工具策略拦截器：L4/L5 必须返回 `confirm_required`，L6 默认拒绝。
3. 清理普通 logger 中的完整工具输入/输出日志，统一改成审计摘要。
4. 为企业 RAG metadata 预留 `tenant_id` 和 `acl_subjects`，并设计 Chroma/BM25 双路 ACL 过滤。
5. 前端补 `confirm_required` 事件处理，作为高风险工具的 UI 基础。
