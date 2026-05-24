# 记忆模块版本档案

## 文档目的

本文档专门记录项目记忆模块的版本演进，方便后续对比每一版的记忆机制、能力边界和改进效果。

设计分析、mem0 理解和详细落地方案继续放在 [memory-module.md](./memory-module.md)。这里重点记录“我们自己的记忆模块从 v1 到 v2 到后续版本到底变了什么”。

## 版本总览

| 版本 | 记忆机制 | 核心能力 | 状态 |
| --- | --- | --- | --- |
| v1 | Working Memory + Session Memory | 最近原文窗口 + 当前会话滚动摘要 | 已完成 |
| v2 | Working Memory + Session Memory + Long-term Memory | 新增跨会话事实级长期记忆、检索注入、查看和删除接口 | 第一阶段已落地 |
| v3 | v2 + 时间、实体、修正历史 | 时间归一化、实体链接、记忆变更审计、用户纠错 | 规划中 |

## v1：会话压缩记忆

### 机制描述

v1 是两层会话记忆：

```text
Working Memory
  最近 N 轮原文，保证当前对话连贯

Session Memory
  当前 session 的滚动摘要，压缩更早历史
```

默认参数：

- `RECENT_WINDOW_TURNS = 6`
- `SUMMARY_TRIGGER_TURNS = 10`

当当前会话不超过 10 轮时，系统直接使用完整历史；超过 10 轮后，早期历史会被压缩进 `chat_session_memories.summary`，后续只保留最近 6 轮原文。

### 主要文件

- `backend/app/services/conversation_memory.py`
- `backend/app/models/chat_history.py`
- `backend/app/agent/router_graph.py`
- `backend/app/agent/agent.py`

### 优点

- 降低长会话 token 成本。
- 保留最近原文，当前对话衔接稳定。
- 通过 `summarized_turn_count` 避免重复压缩同一段历史。

### 局限

- 只在当前 session 内有效，不能稳定跨会话复用。
- 记忆是一段摘要，不能单条检索、删除、追踪来源。
- 无法按当前 query 召回最相关记忆。
- 用户偏好、项目背景、长期任务会随着换 session 丢失上下文。

## v2：事实级长期记忆

### 机制描述

v2 在 v1 旁边新增 Long-term Memory，不替换原来的 Working Memory 和 Session Memory。

```text
Working Memory
  最近 6 轮原文

Session Memory
  当前 session 的滚动摘要

Long-term Memory
  跨 session 的事实级记忆 item
```

v2 第一阶段采用 ADD-only 策略：每轮对话落库后，从用户消息和助手回复中抽取可复用事实，只追加新记忆，不自动覆盖旧记忆、不自动硬删除旧记忆。

### 本阶段已落地能力

- 新增 `long_term_memories` 数据表模型。
- 新增 `LongTermMemoryService`。
- 对每轮对话做长期记忆抽取。
- 使用 hash 精确去重。
- 使用 Chroma 独立 collection 存储长期记忆向量。
- 回答前按当前 query 检索用户相关长期记忆。
- 将长期记忆作为独立系统上下文注入 Chat 和 Tool Agent。
- 新增用户可控接口：
  - `GET /api/memories`
  - `DELETE /api/memories/{memory_id}`
- 长期记忆语义搜索已在问答链路内部使用；独立 `GET /api/memories/search?q=...` 仍属于后续扩展。

### 数据结构

`long_term_memories`：

```text
id
user_id
session_id
memory
memory_type
source
source_message_ids
hash
metadata
score
status
created_at
updated_at
deleted_at
```

### 记忆类型

- `preference`：用户长期偏好。
- `profile`：用户身份或稳定背景。
- `project`：项目背景、目标、架构信息。
- `decision`：重要决策和原因。
- `task`：长期任务、计划、待办。
- `assistant_output`：助手给出的可复用设计结论或接口约定。
- `other`：暂时不能归类但值得保存的信息。

### 相比 v1 的改进

- 从“会话级摘要”升级为“事实级记忆 item”。
- 从“同 session 可用”升级为“同 user 跨 session 可用”。
- 从“整体塞入摘要”升级为“按 query 检索 top_k 记忆”。
- 从“用户不可控”升级为“用户可查看、可搜索、可删除”。
- 从“难以解释来源”升级为“每条记忆记录 session 和 source message ids”。

### 当前限制

- 第一阶段仍然只做 ADD-only，不自动修正旧记忆。
- Chroma 相似度去重失败时会回退到 hash 去重。
- 暂未实现 BM25、实体链接和时间归一化。
- 删除是软删除，并尝试同步删除向量记录；如果向量删除失败，MySQL 状态仍以 `deleted` 为准。

## v3：时间、实体和修正历史

### 目标机制

v3 计划在 v2 基础上增加更强治理能力：

- 时间归一化：把“昨天”“下周”“最近”解析为具体日期或日期范围。
- 实体链接：为项目名、文件名、人名、产品名建立实体索引。
- 记忆历史：记录 ADD、UPDATE、DELETE、ARCHIVE、SUPERSEDE。
- 用户纠错：用户说“我现在不喜欢 X 了”时，新增新事实并将旧事实标记为 superseded。
- 混合检索：向量检索 + BM25 + 实体 boost + reranker。

### 预期价值

- 记忆更可解释。
- 相对时间在几个月后仍然可理解。
- 同一实体的多条事实可以被稳定召回。
- 用户纠错不会粗暴覆盖历史，而是保留变更轨迹。
