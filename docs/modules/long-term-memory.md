# 长期记忆模块设计

## 职责

长期记忆模块负责保存跨会话仍然有价值的信息，例如用户偏好、长期项目背景、稳定约束、重要事实和持续任务。它与当前 session 的上下文摘要不同：会话记忆解决“这次对话前面说过什么”，长期记忆解决“这个用户长期需要系统记住什么”。

## 当前链路

```text
用户消息 + 助手回复
  -> 会话历史写入 MySQL
  -> ConversationMemoryService 更新 session summary
  -> LongTermMemoryService 抽取长期事实
  -> MySQL 写入权威记录
  -> Chroma 写入语义索引

下一次请求
  -> RouterGraph.load_context
  -> LongTermMemoryService.search(query, user_id)
  -> Chroma 按 user_id/status 过滤召回
  -> 回查 MySQL active 记录
  -> 注入 Chat/Agent system prompt 的长期记忆区块
```

## 核心文件

| 文件 | 说明 |
| --- | --- |
| `backend/app/services/long_term_memory.py` | 长期记忆抽取、存储、检索、删除、去重和向量索引同步。 |
| `backend/app/models/chat_history.py` | `LongTermMemory` ORM 模型。 |
| `backend/app/agent/router_graph.py` | 在 `load_context` 中加载长期记忆，并在持久化后触发抽取。 |
| `backend/app/agent/agent.py` | 把长期记忆格式化进 Chat/Agent system prompt，并在流式链路持久化后抽取长期记忆。 |
| `backend/app/router/chat.py` | `/api/memories` 和 `/api/memories/{memory_id}` 管理接口。 |
| `backend/app/router/chat_service.py` | 长期记忆 API 的业务服务封装。 |
| `backend/app/schemas/models.py` | 长期记忆响应 schema。 |
| `backend/tests/test_long_term_memory_unit.py` | 长期记忆、RouterGraph 接入和 API schema 的单元测试。 |

## 数据结构

MySQL 表：`long_term_memories`

关键字段：

```text
id                  # 记忆 ID
user_id             # 用户隔离字段
session_id          # 来源会话
memory              # 记忆内容
memory_type         # preference / profile / project / fact / instruction / relationship / constraint / other
source              # 来源，当前默认为 chat
source_message_ids  # 来源消息 ID 列表
hash                # 规范化文本 hash，用于精确去重
metadata            # 扩展元数据
score               # 抽取置信度或相关性分数
status              # active / deleted
created_at
updated_at
deleted_at
```

Chroma collection：`long_term_memories`

向量文档 metadata 必须包含：

```text
memory_id
user_id
session_id
memory_type
status
```

这些字段是用户隔离和 active 过滤的基础，不能被 LLM 抽取出的普通 metadata 覆盖。

## 用户隔离

长期记忆所有检索和去重都必须带过滤条件：

```python
filter={"user_id": user_id, "status": "active"}
```

即使 Chroma 返回候选结果，也会回查 MySQL：

- 只返回当前 `user_id` 的 active 记录。
- stale vector document 不会直接决定最终结果。
- 删除后的记忆不会继续阻止用户重新保存相同偏好。

## 删除策略

删除长期记忆时采用双层处理：

1. MySQL 权威记录软删除：`status="deleted"`，设置 `deleted_at`。
2. Chroma 向量文档尽力删除：`vector_store.delete(ids=[memory_id])`。

如果 Chroma 删除失败，只记录 warning，不回滚 MySQL 删除。后续语义去重会回查 MySQL，避免 stale vector 继续生效。

## Prompt 注入方式

长期记忆不会伪装成历史对话，而是作为单独 system context 拼接进 prompt。格式化时会清理换行和危险类型字符，降低 prompt injection 风险。

示例：

```text
以下是与当前用户相关、可能有助于回答的长期记忆。它们不是当前对话原文，而是系统从历史交互中抽取的事实。若与用户当前表达冲突，以用户当前表达为准。

1. [preference] 用户偏好回答简洁直接。
2. [project] 用户正在改造 NexusKB 长期记忆模块。
```

## 管理 API

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/api/memories?limit=50&offset=0` | GET | 查询当前登录用户的长期记忆。 |
| `/api/memories/{memory_id}` | DELETE | 删除当前登录用户的一条长期记忆。 |

API 不接受外部传入的 `user_id`，始终使用 JWT 中的当前用户 ID。

## 当前验证

已覆盖的测试点包括：

- ORM 字段存在。
- Prompt formatting 与 sanitization。
- Chat/Agent 函数签名和 RouterGraph 参数透传。
- Chroma search / semantic duplicate 使用 `user_id + status` 过滤。
- hash fallback 不跨用户。
- Chroma 写入 metadata 保留权威过滤字段。
- 删除记忆后尽力删除 Chroma 向量。
- stale vector 不会压制 active MySQL fallback 或重新写入。
- `/api/memories` schema、分页约束和 404 行为。

运行命令：

```bash
PYTHONPATH=backend pytest backend/tests/test_long_term_memory_unit.py -q
```

## 当前限制

- 抽取质量依赖 LLM 输出，后续需要加入人工修正、记忆冲突检测和来源审计。
- 当前管理 API 支持列表和删除，尚未提供用户编辑/合并记忆接口。
- Chroma 是索引层，真实一致性仍以 MySQL 为准。
- 完整 live 验证需要 MySQL、Redis、JWT、LLM、Ollama/Chroma 等服务同时可用。

## 下一步

- 增加前端长期记忆管理页。
- 增加记忆来源消息展示。
- 增加冲突检测，例如用户偏好发生变化时自动标记旧记忆。
- 增加管理端审计日志。
