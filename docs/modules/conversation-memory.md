# 会话记忆压缩设计与 TODO

## 背景

当前会话历史通过 MySQL 的 `chat_sessions` 和 `chat_messages` 保存，调用 Agent 或 Router 时直接读取完整历史。随着会话轮数增加，完整历史会带来上下文变长、成本升高、延迟增加和早期关键信息丢失等问题。

当前记忆体系分为三层：

```text
Working Memory：最近 N 轮原文
Session Memory：更早历史的滚动摘要
Long-term Memory：跨 session 的用户偏好、长期事实和稳定约束
```

本文重点说明 Working Memory 与 Session Memory；Long-term Memory 的 MySQL + Chroma 语义召回设计见 [长期记忆模块设计](./long-term-memory.md)。

## 目标

- 保留最近几轮原文，保证当前对话连贯。
- 将较早历史压缩成 session summary，减少传给 LLM 的上下文长度。
- 摘要持久化到 MySQL。
- RouterGraph 优先使用压缩后的 memory context。
- RouterGraph 和现有 SSE Agent 链路都使用同一套压缩记忆。

## 记忆层级

### Working Memory

最近 N 轮原文，默认 `6` 轮。

用途：

- 保留用户最近的表达方式和上下文。
- 避免摘要丢失刚刚发生的细节。
- 直接传给 Agent 或 Router LLM。

### Session Memory

当前 session 的滚动摘要，覆盖较早历史。

用途：

- 保留本会话中的目标、约束、关键结论、已完成事项、未完成事项。
- 压缩长历史，避免每次把完整历史塞进 prompt。

## 第一版策略

配置：

```text
recent_window_turns = 6
summary_trigger_turns = 10
```

规则：

- 当总轮数小于等于 `summary_trigger_turns` 时，直接使用完整历史。
- 当总轮数超过 `summary_trigger_turns` 时：
  - 最近 `recent_window_turns` 轮保留原文。
  - 更早的新增历史增量压缩进 `summary`。
  - 使用 `summarized_turn_count` 记录已经被摘要覆盖到第几轮，避免重复摘要。

## 数据表

新增表：

```text
chat_session_memories
```

字段：

```text
id
session_id
user_id
summary
summarized_turn_count
created_at
updated_at
```

说明：

- `session_id` 关联现有 `chat_sessions.id`。
- `summary` 保存当前 session 的滚动摘要。
- `summarized_turn_count` 表示已经摘要覆盖的历史轮数。
- 不修改现有 `chat_sessions` 表，降低侵入。

## 服务设计

新增模块：

```text
backend/app/services/conversation_memory.py
```

核心结构：

```python
class MemoryContext(BaseModel):
    summary: str
    recent_history: list[tuple[str, str]]
    compressed_turns: int
    total_turns: int

class ConversationMemoryService:
    async def get_memory_context(session_id: str, user_id: str) -> MemoryContext
    async def update_memory(session_id: str, user_id: str) -> MemoryContext
```

## 摘要 Prompt

摘要目标不是普通复述，而是提取后续回答需要记住的信息。

摘要应保留：

- 用户明确表达过的目标、偏好、约束。
- 已确认的项目背景和技术选择。
- 已完成和未完成事项。
- 重要结论、接口、文件路径、数据来源。
- 后续回答必须记住的上下文。

摘要应删除：

- 寒暄。
- 重复确认。
- 临时错误信息。
- 无关细节。

## RouterGraph 接入

`load_context` 节点改为：

```text
读取 MemoryContext
将 summary 转成一条系统上下文消息或虚拟历史
保留 recent_history 作为原文历史
```

第一版为了兼容现有 `get_agent_response(query, history)` 的 `(user, assistant)` 格式，可以把摘要作为一轮虚拟历史放在最前面：

```python
(
  "以下是本会话较早历史的摘要，请作为上下文参考。",
  memory.summary
)
```

然后拼接最近 N 轮原文。

## TODO

### 第一阶段：设计与数据模型

- [x] 新增本设计文档。
- [x] 新增 `ChatSessionMemory` SQLAlchemy 模型。
- [x] 确认 `init_db()` 能自动创建 `chat_session_memories` 表。

### 第二阶段：服务层

- [x] 新增 `backend/app/services/conversation_memory.py`。
- [x] 定义 `MemoryContext`。
- [x] 实现读取完整历史。
- [x] 实现最近 N 轮切片。
- [x] 实现增量摘要选择逻辑。
- [x] 实现摘要生成链。
- [x] 实现 summary 持久化。
- [x] 实现 `get_history_for_agent()` 兼容现有 `(user, assistant)` 历史格式。

### 第三阶段：RouterGraph 接入

- [x] `RouterGraph.load_context` 改为读取压缩记忆。
- [x] `RouterGraph.persist_message` 写入消息后触发记忆更新。
- [x] `/api/agent/query/stream` 接入压缩记忆。

### 第四阶段：验证

- [x] 构造 12 轮测试会话。
- [x] 验证最近 6 轮原文保留。
- [x] 验证前面历史被摘要。
- [x] 验证 `summarized_turn_count` 不重复摘要旧内容。
- [x] 验证 RouterGraph 能读取摘要 + 最近历史。
- [x] 清理测试会话。

### 后续扩展

- [ ] 引入 token 估算，替代纯轮数阈值。
- [x] 增加 Long-term Memory 表。
- [x] 增加长期记忆列表和删除 API。
- [ ] 增加记忆修正、合并和冲突处理。

## 更新记录

- 2026-05-14：创建两层会话记忆压缩设计文档。
- 2026-05-14：完成 `ChatSessionMemory` 模型、`ConversationMemoryService` 和 RouterGraph 接入；验证 12 轮会话会压缩前 6 轮并保留最近 6 轮，RouterGraph `load_context` 可读取摘要 + 最近历史。
- 2026-05-14：将 `/api/agent/query/stream` SSE 链路接入两层记忆，流式响应前读取压缩历史，响应落库后更新 session summary。

## 当前验证结果

- `chat_session_memories` 表已通过 `init_db()` 创建。
- 12 轮测试会话中，`summarized_turn_count=6`。
- `recent_history` 保留第 7-12 轮，共 6 轮。
- `to_agent_history()` 输出 7 条历史：1 条摘要虚拟历史 + 6 条最近原文历史。
- 重复调用 `get_memory_context()` 后，`summarized_turn_count` 仍为 6，没有重复摘要旧内容。
- 测试会话 `memory-smoke-session` 和 `memory-router-load-smoke` 已清理。
- SSE 链路已改为使用 `conversation_memory_service.get_history_for_agent()`。
