# NexusKB 版本更新记录

整理日期：2026-05-26

本文档记录 NexusKB 项目级版本演进，重点说明每个阶段完成了哪些能力、文档体系发生了什么变化、下一步应该继续补什么。模块内部的细粒度版本，例如长期记忆 v1/v2/v3，继续放在对应模块文档中维护。

## 版本记录口径

- 这里的“版本”是项目阶段版本，不等同于 Git tag 或正式发行包。
- 项目级能力、架构、文档体系和重要决策记录在本文。
- 单模块能力边界继续放在 `docs/project_guide/modules/` 下的模块文档。
- 执行路线、下一步任务和状态以 [工作台账](./WORKBOARD.md) 与 [RAGFlow Agentic RAG 执行路线图](./RAGFLOW_AGENTIC_RAG_ROADMAP.md) 为准。

## 当前版本概览

| 阶段版本 | 日期 | 主题 | 状态 |
| --- | --- | --- | --- |
| v0.4 | 2026-05-26 | 项目级版本更新记录补齐 | 当前文档版本 |
| v0.3 | 2026-05-24 | 项目指南与文档中心收敛 | 已完成 |
| v0.2 | 2026-05-16 | Agentic RAG 文档体系拆分 | 已完成 |
| v0.1 | 2026-05-13 ~ 2026-05-15 | 企业 RAG 主线能力成型 | 已完成 |

## v0.4：项目级版本更新记录补齐

日期：2026-05-26

### 更新内容

- 新增项目级版本更新记录，用于统一追踪 NexusKB 的阶段演进。
- 明确项目级版本记录与模块级版本档案的边界：
  - 项目级版本记录关注整体能力、架构、文档体系和阶段目标。
  - 模块级版本档案继续记录长期记忆、RAG 策略、评测脚本等细粒度能力变化。
- 将后续版本更新入口纳入 `docs/README.md`，便于从文档中心直接访问。

### 后续维护

- 每次完成一个阶段性能力，例如 RAG Debug 接口、统一响应结构、Citation Accuracy 或安全边界落地后，都应补充一条版本记录。
- 如果某个阶段对应多个提交，版本记录应总结“能力变化”和“对读者的影响”，不要只罗列 commit message。

## v0.3：项目指南与文档中心收敛

日期：2026-05-24

### 主要变化

- 建立 `docs/README.md` 作为项目文档统一入口。
- 建立 `docs/project_guide/` 作为给人看的项目指南目录，集中维护项目总览、当前架构、项目介绍、架构图和流程图。
- 建立 `docs/project_guide/modules/` 作为模块设计入口，覆盖后端、RouterGraph、企业 RAG、会话记忆、长期记忆、前端/用户服务、安全审计和最小验收标准。
- 将历史材料归入 `docs/archive/`，避免旧计划和当前主线混在一起。

### 对项目的影响

- 新读者可以从 `docs/README.md` 开始，按推荐阅读顺序理解项目。
- 当前稳定实现、探索性设计和历史记录的边界更清晰。
- 后续新增文档有明确放置规则：模块文档进 `project_guide/modules/`，实验记录进 `experiments/`，运维文档进 `ops/`，旧资料进 `archive/`。

## v0.2：Agentic RAG 文档体系拆分

日期：2026-05-16

### 主要变化

- 将 RAGFlow / Agentic RAG 方向拆分为四类专题文档：
  - [执行路线图](./RAGFLOW_AGENTIC_RAG_ROADMAP.md)
  - [架构设计](./RAGFLOW_AGENTIC_RAG_ARCHITECTURE.md)
  - [评估设计](./RAGFLOW_AGENTIC_RAG_EVALUATION.md)
  - [安全设计](./RAGFLOW_AGENTIC_RAG_SECURITY.md)
- 明确当前项目定位为“企业级多策略 Agentic RAG 检索增强问答系统”。
- 将下一项开发重点收敛为：RAG Debug 接口、统一 RAG Response Schema、SSE Event Schema。

### 对项目的影响

- 路线、架构、评测和安全不再混在一个大文档里。
- 后续开发可以围绕“可调试、可评测、可展示”的 RAG 工程闭环推进。
- 文档已经明确哪些能力已完成、部分完成、待实现和可选增强。

## v0.1：企业 RAG 主线能力成型

日期：2026-05-13 ~ 2026-05-15

### 主要变化

- 项目主线收敛到 `backend/` FastAPI + LangChain/LangGraph 企业 RAG Agent。
- Django 用户服务继续负责注册、登录、JWT、用户信息和文件相关能力。
- Vue 前端继续承担聊天、登录注册、会话管理、个人页和设置页面。
- 引入 EnterpriseRAG-Bench，并完成 parent-child 分块、Chroma 入库和多策略检索评测。
- 企业 RAG 链路形成 Chroma dense retrieval、BM25、RRF、条件 reranker 的组合路线。
- 新增独立 `EnterpriseRagService`，避免企业评测库链路和原业务知识库链路混用。
- RouterGraph 统一承接 `/api/agent/query/stream` 主入口，支持普通聊天、企业知识库、工具调用、系统保护和澄清分流。
- 接入 Working Memory 与 Session Memory，用于多轮会话上下文。
- 建立 `PERF_METRIC` 性能埋点，覆盖 Router、PureChat、Tool Agent、企业 RAG、MySQL 和记忆关键耗时。

### 关键决策

- 以 `backend/` 为当前主线项目，`DSagent/` 只作为参考实现。
- 企业 RAG 使用 parent-child 分块：child chunk 负责召回，parent chunk 负责生成上下文回填。
- `source_hints` 暂不做硬过滤，避免来源预测错误过滤掉正确答案。
- reranker 保留为策略能力，不无条件默认启用。
- 线上 reranker 候选窗口收敛到 10，在评测中兼顾效果和延迟。
- SSE 阶段事件进入真实前后端链路，用于展示 retrieving、reranking、summarizing 等过程状态。

### 对项目的影响

- NexusKB 从普通 RAG 问答原型，升级为具备 RouterGraph、多策略检索、会话记忆、性能观测和评测基础的企业知识库问答系统。
- 后续重点不再是“能不能问答”，而是统一响应结构、引用溯源、debug 可观察性、评测闭环和安全边界。

## 下一阶段建议

| 优先级 | 下一项 | 目标 |
| --- | --- | --- |
| P0 | 统一 RAG Response Schema | 稳定返回 answer、sources、strategy、metrics、debug_id 等字段 |
| P0 | 标准化 SSE Event Schema | 固定 route_decided、retrieving、reranking、generating、done、error 等事件结构 |
| P0 | 新增或完善 RAG Debug 接口 | 暴露 retrieved、fused、reranked、selected_context、sources、metrics |
| P0 | 标准化 sources | 每条 source 至少包含 doc_id、chunk_id、title、section、page、snippet、score |
| P1 | 补齐评测闭环 | 增加 Citation Accuracy、Answer Faithfulness、失败样例格式和策略对比报告 |
| P1 | 补齐安全边界 | 强化用户文档隔离、引用泄露防护、prompt injection 隔离和日志脱敏 |

## 维护规则

- 完成阶段性开发或文档收敛后，在本文新增一节。
- 每条版本记录应包含：日期、主要变化、关键决策、对项目的影响、后续动作。
- 如果某条更新只影响单个模块，优先更新模块文档；只有影响项目级能力或阅读入口时，才写入本文。
- 如果本文与工作台账或路线图状态不一致，优先更新工作台账和路线图，再同步本文。
