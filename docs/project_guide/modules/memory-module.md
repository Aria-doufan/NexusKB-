# 记忆模块设计记录

## 文档目的

本文档用于持续记录项目记忆模块的设计演进，重点覆盖三类内容：

- 当前现状：项目已有的会话历史、压缩摘要、上下文注入方式。
- 改进方向：参考 mem0 等记忆系统后，拟引入的长期记忆、事实抽取、检索、修正机制。
- 预期效果：改造后对回答质量、上下文一致性、token 成本和长期个性化能力的影响。

版本化对比单独记录在：[记忆模块版本档案](./memory-versions.md)。

当前版本先记录对 mem0 项目的理解，后续再补充本项目现状分析和落地方案。

## mem0 理解

### 一句话理解

mem0 不是简单的“聊天记录保存器”，也不是只把历史对话压缩成摘要的组件。它更像一层独立的长期记忆系统：从对话中抽取可复用的事实级记忆，按用户、智能体、运行上下文做隔离，再通过语义检索、关键词检索、实体关联和时间语义，把相关记忆注入后续对话。

换句话说，mem0 关注的是“哪些信息值得以后被想起来”，而不是“怎样把所有历史塞回 prompt”。

### 核心设计变化

mem0 在 2026 年 4 月的新记忆算法中强调了几个关键方向：

1. ADD-only extraction

   新记忆抽取阶段尽量只做追加，不频繁在抽取时 UPDATE 或 DELETE。这样可以降低模型误判导致的覆盖、误删和信息丢失风险。

   旧式记忆系统常见做法是：发现新事实后，马上判断它是否要替换旧事实。但这会把“事实抽取”和“事实裁决”耦合到一次 LLM 判断里，风险较高。mem0 的思路是先把事实留下，后续检索和排序阶段再决定哪条更相关、更当前。

2. 事实级记忆

   mem0 把对话拆成多条独立、可检索、可追踪的 memory item。例如：

   - 用户偏好某种技术栈。
   - 用户正在做某个项目。
   - 助手给过某个明确建议。
   - 用户某个计划发生在具体日期。

   这和“滚动摘要”不同。摘要会把很多细节压成一段文本，后续只能整体使用；事实级记忆可以单独召回、单独删除、单独更新，也可以做去重和关联。

3. 多作用域隔离

   mem0 的记忆操作至少要求有一个作用域标识，例如：

   - `user_id`
   - `agent_id`
   - `run_id`

   这些字段决定记忆属于谁、属于哪个智能体、属于哪次运行上下文。这个设计对多用户系统很重要，可以避免不同用户之间的记忆污染，也方便将来支持“用户长期记忆”“Agent 行为记忆”“单次任务记忆”等不同层级。

4. 最近消息缓存

   mem0 仍然保留最近消息。它的作用不是替代长期记忆，而是帮助抽取器理解代词、上下文和省略表达。例如用户说“这个方案我更喜欢第二个”，最近消息可以帮助系统知道“第二个”指什么。

   这说明长期记忆和短期上下文不是二选一，而是协作关系：

   - 短期上下文负责当前对话连贯性。
   - 长期记忆负责跨会话、跨时间的稳定事实。

5. 语义检索 + BM25 + 实体增强

   mem0 的检索不是单纯向量相似度。它会结合：

   - 语义向量检索：适合表达相近但措辞不同的问题。
   - BM25/关键词检索：适合专有名词、编号、具体术语。
   - 实体匹配增强：识别查询和记忆中的人名、项目名、地点、产品名等实体，并提升相关记忆分数。

   这个组合很重要。单纯向量检索容易漏掉精确名称；单纯关键词检索又不理解语义。混合检索能让长期记忆更稳定。

6. 实体链接

   mem0 会抽取实体，并把实体和 memory id 建立关联。例如“Poppy 是用户的狗”和“Poppy 昨天去体检”可以被链接起来。

   这个机制的价值是：当用户以后问“Poppy 怎么样了”，系统不仅能召回名字匹配的记忆，还能召回和 Poppy 相关的一系列事件、状态和变化。

7. 时间语义

   mem0 特别强调将相对时间锚定为具体时间。例如：

   - “昨天”要根据对话发生日期解析。
   - “上周”要记录成具体周或日期范围。
   - “最近开始读某本书”要尽量绑定到 observation date。

   这样做是为了让记忆在几个月后仍然可用。否则“用户上周开会”这种记忆很快就失去意义。

8. 历史可追踪

   mem0 为记忆变更保存 history。每条记忆可以追踪 ADD、UPDATE、DELETE 等事件。这对调试、审计和用户可控性很关键。

   一个成熟记忆系统不只需要“记住”，还要支持：

   - 为什么记住了。
   - 从哪条消息抽取出来。
   - 后来是否被修改。
   - 是否被用户删除或纠正。

### 记忆写入流程理解

mem0 的典型写入流程可以概括为：

1. 接收对话消息。
2. 构造作用域过滤条件，例如 `user_id`、`agent_id`、`run_id`。
3. 读取最近消息，用于辅助理解上下文。
4. 搜索已有相关记忆，用于去重和关联。
5. 调用 LLM 抽取新增事实级记忆。
6. 对抽取结果做批量 embedding。
7. 用 hash 做去重，避免重复写入。
8. 写入向量库，并保存 memory payload。
9. 写入 history 记录。
10. 抽取实体，并将实体链接到 memory id。
11. 保存最近消息缓存。

这个流程说明：mem0 的“记忆写入”不是简单 insert 一段文本，而是一个抽取、去重、向量化、实体关联、历史记录的流水线。

### 记忆检索流程理解

mem0 的典型检索流程可以概括为：

1. 接收用户查询。
2. 校验 filters，确保至少有用户、智能体或运行作用域。
3. 对查询做语义 embedding。
4. 做向量检索，获取语义相关候选。
5. 做关键词检索，补充精确匹配候选。
6. 抽取查询实体，并通过实体库找到关联 memory id。
7. 融合语义分、BM25 分、实体增强分。
8. 按阈值和 top_k 返回最相关记忆。

这说明 mem0 的记忆召回不是“把所有记忆都塞进上下文”，而是按当前问题挑选最可能有用的少量记忆。

### 对我们项目的启发

我们当前的记忆机制更接近“Session Memory 压缩”：保留最近几轮原文，再把更早历史滚动摘要成一段 session summary。这个机制能降低上下文长度，但它没有形成真正的长期记忆层。

从 mem0 看，我们后续应该把记忆分成三层：

1. Working Memory

   最近几轮原文，用于保证当前对话连贯。

2. Session Memory

   当前会话的滚动摘要，用于压缩长对话。

3. Long-term Memory

   跨会话的事实级记忆，用于记录用户偏好、项目背景、长期目标、重要决策、助手已给出的方案和后续要复用的信息。

其中 Long-term Memory 不应该只是一段 summary，而应该是一组结构化、可检索、可删除、可追踪的 memory item。

### 值得借鉴但不必照搬的点

mem0 的完整方案包含 SDK、云服务、自托管服务、多种向量库、多种模型适配、实体库和历史数据库。对我们当前项目来说，不一定要直接引入整个 mem0 包。

更合适的路径是先借鉴它的核心设计：

- 新增事实级长期记忆表。
- 每轮对话后异步抽取记忆。
- 回答前按 query 检索用户相关记忆。
- 先实现语义检索和 hash 去重。
- 后续再加 BM25、实体链接、时间归一化、删除和修正。

这样可以保持项目结构可控，也能逐步验证效果。

## 当前现状

### 已有能力

当前项目已经有一套两层会话记忆机制，核心目标是解决长会话上下文过长的问题。

现有结构可以概括为：

```text
Working Memory：最近 N 轮原文
Session Memory：当前 session 的滚动摘要
```

代码中的默认参数是：

- `RECENT_WINDOW_TURNS = 6`
- `SUMMARY_TRIGGER_TURNS = 10`

也就是说，当会话轮数不超过 10 轮时，系统直接使用完整历史；超过 10 轮后，把更早的历史压缩进 `summary`，只保留最近 6 轮原文继续传给 Agent 或 Chat 链路。

### 数据结构

当前记忆相关数据主要落在 MySQL：

1. `chat_sessions`

   保存会话本身，包含：

   - `id`
   - `user_id`
   - `title`
   - `metadata`
   - `created_at`
   - `updated_at`

2. `chat_messages`

   保存原始消息，包含：

   - `id`
   - `session_id`
   - `role`
   - `content`
   - `metadata`
   - `created_at`

   当前历史读取逻辑会按时间顺序把消息重新组合成 `(user_message, assistant_message)` 形式。

3. `chat_session_memories`

   保存当前会话的滚动摘要，包含：

   - `id`
   - `session_id`
   - `user_id`
   - `summary`
   - `summarized_turn_count`
   - `created_at`
   - `updated_at`

   这张表只保存 session 级摘要，不保存事实级长期记忆。

### 当前调用链路

当前主链路如下：

1. 用户请求进入 `/api/agent/query/stream` 或非流式 query 接口。
2. `RouterGraph.load_context` 调用 `conversation_memory_service.get_memory_context(session_id, user_id)`。
3. `ConversationMemoryService` 从 MySQL 读取当前 session 的完整历史和已有摘要。
4. 如果轮数未超过阈值，直接返回完整历史。
5. 如果轮数超过阈值，调用 LLM 将早期历史增量压缩进 `chat_session_memories.summary`。
6. `MemoryContext.to_agent_history()` 把 summary 伪装成一轮虚拟历史，再拼接最近 6 轮原文。
7. Router 根据问题选择 `chat`、`tool_action`、`enterprise_knowledge` 等链路。
8. 回答完成后，`persist_message` 写入 `chat_messages`，再触发 `conversation_memory_service.update_memory()` 更新摘要。

流式 Chat 和流式 Agent 链路也会在没有显式传入 history 时读取 `conversation_memory_service.get_history_for_agent()`，并在回答落库后更新 session summary。

### 当前优点

现有机制已经解决了几个基础问题：

- 不再每次把完整历史传给 LLM，长会话 token 成本更可控。
- 最近 6 轮原文保留，当前对话连贯性比较好。
- 更早历史通过 summary 保留，不至于完全丢失。
- `summarized_turn_count` 可以避免重复压缩同一段历史。
- 记忆跟 `session_id` 和 `user_id` 绑定，有基本用户隔离。
- Router、Pure Chat、Tool Agent 的流式链路已经接入同一套压缩记忆。

### 当前限制

现有机制的主要问题是：它是“会话压缩”，还不是“长期记忆”。

具体限制包括：

1. 记忆粒度过粗

   现在早期历史会被压成一段 summary。summary 对模型友好，但对系统不友好：不能单独检索某个事实，不能单独删除某条偏好，也不能知道某个事实来自哪条消息。

2. 缺少跨会话长期记忆

   `chat_session_memories` 绑定 `session_id`。换一个 session 后，旧 session 的 summary 不会自然进入新 session。用户长期偏好、项目背景、持续任务、命名实体等信息无法稳定跨会话复用。

3. 没有按 query 检索记忆

   当前是“把当前 session 的压缩历史整体喂给模型”，不是“根据当前问题召回相关记忆”。如果 summary 很长或主题很多，模型仍然需要在一段大文本里自己找重点。

4. 缺少事实级去重

   现有 summary 只能靠 LLM 自己避免重复表达。系统层没有 memory hash、相似度去重、同批去重，也没有“这条事实已经记过”的明确判断。

5. 缺少更新、删除和纠错机制

   用户如果说“刚才记错了”“不要记这个”“我现在不喜欢 X 了”，系统目前没有事实级记忆可以精确修改，只能依赖后续 summary 覆盖旧表达。

6. 缺少时间归一化

   如果用户说“昨天”“下周”“最近”，summary 可能原样保留这些相对时间。几个月后再看，这些记忆会变得模糊。

7. 缺少实体关联

   系统不会把项目名、人名、文件名、产品名等实体抽出来建立索引。后续用户提到同一个实体时，只能依赖 summary 或最近历史。

8. 可解释性弱

   现在很难回答：

   - 为什么系统记住了这件事？
   - 它从哪条消息抽取出来？
   - 什么时候更新过？
   - 当前回答用了哪些记忆？

9. 敏感记忆缺少治理

   长期记忆如果扩展到用户偏好、身份、项目、健康、财务等内容，需要有更清楚的分类、删除、过期、用户可见和审计策略。当前 session summary 还没有这些治理能力。

## 改进方案

### 总体目标

目标不是替换现有会话压缩，而是在它旁边新增一层 Long-term Memory。

改造后的结构建议为：

```text
Working Memory
  最近 6 轮原文，负责当前对话连贯性

Session Memory
  当前 session 的滚动摘要，负责长会话压缩

Long-term Memory
  跨 session 的事实级记忆，负责用户偏好、项目背景、长期目标、重要决策和可复用上下文
```

三层记忆各司其职，不互相替代。

### 第一阶段：事实级长期记忆

第一阶段先实现最小可用的长期记忆，不追求一步到位复制 mem0。

新增服务：

```text
backend/app/services/long_term_memory.py
```

建议核心方法：

```python
class LongTermMemoryService:
    async def extract_and_store(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ) -> list[LongTermMemoryItem]:
        ...

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 8,
    ) -> list[LongTermMemoryItem]:
        ...

    async def delete_memory(
        self,
        memory_id: str,
        user_id: str,
    ) -> None:
        ...
```

第一阶段只做 ADD-only，不做自动 UPDATE/DELETE。这样可以降低误删风险，也更容易观察效果。

### 数据表设计

建议新增 `long_term_memories`：

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

字段说明：

- `id`：单条长期记忆 id。
- `user_id`：用户隔离字段，必须有。
- `session_id`：来源 session，便于追溯。
- `memory`：事实级记忆文本。
- `memory_type`：记忆类型，例如 preference、profile、project、decision、task、assistant_output、other。
- `source`：来源，例如 chat、agent、tool、document。
- `source_message_ids`：来源消息 id 列表，便于解释和审计。
- `hash`：对 memory 文本做 hash，用于精确去重。
- `metadata`：扩展字段，可存实体、时间、置信度、语言、标签等。
- `score`：抽取阶段置信度或后续质量分。
- `status`：active、deleted、archived。
- `deleted_at`：软删除时间。

如果要支持向量检索，可以有两种路径：

1. 直接复用 Chroma，新增 memory 专用 collection。
2. MySQL 存结构化元数据，Chroma 存 embedding 和可检索 payload。

建议第一版采用第二种：MySQL 管理事实、状态和审计，Chroma 管理向量召回。

### 记忆抽取策略

每轮对话完成后，使用用户消息和助手回复抽取长期记忆。抽取结果必须是 JSON，建议格式：

```json
{
  "memories": [
    {
      "memory": "用户正在为当前项目改进记忆模块，希望参考 mem0 设计长期记忆机制。",
      "memory_type": "project",
      "reason": "该信息描述了持续项目目标，后续对话需要复用。",
      "confidence": 0.86
    }
  ]
}
```

抽取规则建议：

- 只抽取后续可能复用的信息。
- 不抽取寒暄、临时语气、无意义确认。
- 不把“用户问了什么”当记忆，要抽取问题中隐含的事实。
- 保留专有名词、文件名、接口名、日期、数量。
- 如果用户明确说“记住”，提高抽取优先级。
- 如果用户明确说“不要记”，禁止写入长期记忆。
- 默认用用户原语言记录，避免无必要翻译。

适合记录的类型：

- 用户长期偏好：回答风格、技术偏好、交互偏好。
- 用户身份和项目背景：角色、项目目标、当前系统架构。
- 长期任务和计划：后续要做什么、已达成哪些共识。
- 重要决策：为什么选某方案，不选某方案。
- 助手产出中可复用的内容：设计结论、接口约定、迁移方案。

不适合记录的类型：

- 普通问候。
- 一次性查询。
- 明显临时的信息。
- 未经用户确认的敏感推断。
- 模型自己猜测出来的个人属性。

### 去重策略

第一版先做两层去重：

1. 精确 hash 去重

   对规范化后的 `memory` 文本计算 hash。如果同一 `user_id` 下已经存在相同 hash，则跳过。

2. 相似度去重

   写入前用新 memory 检索该用户已有长期记忆。如果 top1 相似度高于阈值，例如 0.92，则跳过或记录为 duplicate candidate。

暂不做自动合并。因为自动合并需要判断新旧事实是否冲突，风险比追加更高。

### 记忆检索策略

回答前新增长期记忆召回：

```text
用户 query
  -> LongTermMemoryService.search(query, user_id)
  -> 返回 top_k 条相关长期记忆
  -> 注入 Chat/Agent/Router 上下文
```

第一版检索可以先做：

- Chroma 向量检索。
- `where={"user_id": user_id, "status": "active"}` 过滤。
- top_k 默认为 8。
- 最低相似度阈值，例如 0.35 或按实际 embedding 调整。

后续再增强：

- BM25 关键词召回。
- memory_type 权重。
- 时间新鲜度权重。
- 实体匹配加权。
- reranker 重排。

### 上下文注入策略

长期记忆不应该伪装成普通对话历史。建议作为单独系统上下文注入：

```text
以下是与当前用户相关、可能有助于回答的长期记忆。它们不是当前对话原文，而是系统从历史交互中抽取的事实。若与用户当前表达冲突，以用户当前表达为准。

1. ...
2. ...
```

注入顺序建议：

1. 系统角色和安全规则。
2. 长期记忆。
3. session summary。
4. 最近 N 轮原文。
5. 当前用户问题。

这样模型能区分“长期事实”和“当前对话历史”，也能在冲突时优先当前输入。

### 与 RouterGraph 集成

建议在 `RouterGraph.load_context` 中扩展：

1. 继续读取 `conversation_memory_service.get_memory_context()`。
2. 新增读取 `long_term_memory_service.search(query, user_id)`。
3. 将长期记忆写入 `GraphState.long_term_memories`。
4. Router 决策时可以参考长期记忆，但不要让长期记忆强行改变用户当前意图。
5. Chat 和 Agent 节点生成回答时，把长期记忆作为独立上下文传入。

建议扩展 `GraphState`：

```python
long_term_memories: list[dict[str, Any]]
```

### 写入时机

建议先在回答落库后触发长期记忆抽取：

```text
persist_message
  -> add_message
  -> update session summary
  -> extract_and_store long-term memories
```

为了不影响首 token 延迟，流式链路可以先同步完成消息落库，再把长期记忆抽取放到后台任务。

第一版如果没有任务队列，可以用 FastAPI `BackgroundTasks` 或 `asyncio.create_task`。后续再迁移到 Celery 或独立 worker。

### 用户可控接口

长期记忆必须支持用户查看和删除。当前已落地接口：

```text
GET /api/memories
DELETE /api/memories/{memory_id}
```

语义搜索当前由 RouterGraph/长期记忆服务在问答链路内部调用，尚未作为公开路由暴露。后续可以扩展：

```text
GET /api/memories/search?q=...
PATCH /api/memories/{memory_id}
POST /api/memories/{memory_id}/feedback
```

最小可用阶段至少需要：

- 用户能看到系统记住了什么。
- 用户能删除某条记忆。
- 删除后检索不再召回。

### 第二阶段：时间、实体和修正

第一阶段稳定后，再补 mem0 更强的能力：

1. 时间归一化

   抽取时将“昨天”“下周”“最近”解析为具体日期或日期范围，写入 metadata。

2. 实体表

   新增 `memory_entities`：

   ```text
   id
   user_id
   memory_id
   entity_text
   entity_type
   created_at
   ```

   用于后续实体召回和关系追踪。

3. 记忆历史表

   新增 `long_term_memory_history`：

   ```text
   id
   memory_id
   event
   old_memory
   new_memory
   reason
   created_at
   ```

   支持 ADD、UPDATE、DELETE、ARCHIVE 的审计。

4. 用户修正

   当用户说“我现在不喜欢 X 了”，系统可以新增一条新事实，同时把相关旧事实标记为 superseded，而不是直接硬删。

5. 混合检索

   在向量检索之外加入 BM25 和实体 boost，再统一融合分数。

## 预期效果

### 回答连续性

现有机制只能在同一个 session 中保持较好的连续性。加入长期记忆后，用户换一个会话继续讨论同一项目，系统仍然能召回项目背景、已达成决策和用户偏好。

### 个性化能力

系统可以稳定记住用户偏好的回答风格、技术栈、项目目标和工作习惯，而不需要用户每次重复说明。

### token 成本

长期记忆通过 query 检索召回 top_k，而不是把所有历史和所有摘要都塞进 prompt。随着对话增长，成本增长会更慢。

### 信息保真度

事实级记忆能保留更多具体细节，例如接口名、文件路径、日期、项目名。相比大段 summary，它更不容易在多次压缩后丢失关键信息。

### 可解释性

每条长期记忆都有 id、来源 session、来源消息、创建时间和状态。后续可以在调试界面或接口中展示“本次回答使用了哪些记忆”。

### 用户控制

用户可以查看、删除、修正长期记忆。这一点对生产系统很重要，因为长期记忆一旦涉及用户偏好、身份、工作项目或敏感内容，就必须可控。

### 风险降低

采用 ADD-only 第一版可以降低自动覆盖和自动删除带来的风险。遇到冲突时，先保留新事实，检索和回答阶段以当前用户表达优先；等系统成熟后再引入 supersede 和 update 机制。

## 参考资料

- mem0 GitHub README: https://github.com/mem0ai/mem0
- mem0 核心实现: https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py
- mem0 记忆抽取 Prompt: https://github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py
- mem0 迁移说明: https://github.com/mem0ai/mem0/blob/main/MIGRATION_GUIDE_v1.0.md
