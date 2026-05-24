# 项目指南

整理日期：2026-05-24

这个目录是 NexusKB 的当前项目指南，用于快速理解项目定位、真实架构、核心模块和主要流程。Agentic RAG 的专题设计请看顶层 [RAGFlow 文档集](../RAGFLOW_AGENTIC_RAG_ROADMAP.md)。

## 推荐阅读顺序

1. [项目总览](./PROJECT_OVERVIEW.md)：项目定位、系统边界、主线能力和目标架构。
2. [当前架构快照](./CURRENT_ARCHITECTURE_REVIEW.md)：当前真实链路、技术栈和关键演进。
3. [项目介绍](./PROJECT_INTRO.md)：面向外部介绍项目是什么、解决什么问题、技术栈是什么。
4. [项目架构图集](./NEXUSKB_ARCHITECTURE_DIAGRAMS.md)：系统架构、数据流、RAG、记忆、安全等 Mermaid 图。
5. [项目流程图总览](./NEXUSKB_PROJECT_FLOWCHARTS.md)：整体流程和各模块流程图。
6. [模块设计索引](./modules/README.md)：进入每个模块的详细设计。

## 模块设计

| 模块 | 文档 | 说明 |
| --- | --- | --- |
| 后端总览 | [backend.md](./modules/backend.md) | FastAPI 主后端结构、核心职责和运行依赖 |
| Chat Agent MVP | [chat-agent-mvp.md](./modules/chat-agent-mvp.md) | 普通聊天、工具调用、企业知识库路由边界 |
| Agent Router | [agent-router.md](./modules/agent-router.md) | LangGraph RouterGraph 设计、状态和节点职责 |
| 企业 RAG 与检索 | [rag-retrieval.md](./modules/rag-retrieval.md) | parent-child 分块、混合召回、reranker 策略 |
| 会话记忆 | [conversation-memory.md](./modules/conversation-memory.md) | Working Memory、Session Memory 与压缩策略 |
| 长期记忆模块 | [memory-module.md](./modules/memory-module.md) | 长期记忆抽取、存储、召回和删除链路 |
| 记忆模块版本档案 | [memory-versions.md](./modules/memory-versions.md) | v1/v2/v3 记忆机制、能力边界和改进对比 |
| 前端与用户服务 | [frontend-user-service.md](./modules/frontend-user-service.md) | Vue 前端与 Django 用户服务边界 |
| 权限、安全与审计 | [security_audit_design.md](./modules/security_audit_design.md) | 工具调用边界、企业数据访问边界和审计事件设计 |
| 核心模块最小验收 | [minimum_acceptance_checklist.md](./modules/minimum_acceptance_checklist.md) | 每个核心模块的接口返回、延迟、错误兜底和测试命令 |

## Agentic RAG 专题

这些文档位于 `docs/` 顶层，用于记录 Agentic RAG / RAGFlow 方向的路线图、架构、评估和安全边界：

| 专题 | 文档 | 说明 |
| --- | --- | --- |
| 执行路线图 | [RAGFLOW_AGENTIC_RAG_ROADMAP.md](../RAGFLOW_AGENTIC_RAG_ROADMAP.md) | 阶段、状态、优先级和下一步任务 |
| 架构设计 | [RAGFLOW_AGENTIC_RAG_ARCHITECTURE.md](../RAGFLOW_AGENTIC_RAG_ARCHITECTURE.md) | Agentic RAG 主循环、状态、检索和调试设计 |
| 评估设计 | [RAGFLOW_AGENTIC_RAG_EVALUATION.md](../RAGFLOW_AGENTIC_RAG_EVALUATION.md) | 企业 RAG 评估集、指标和实验方法 |
| 安全设计 | [RAGFLOW_AGENTIC_RAG_SECURITY.md](../RAGFLOW_AGENTIC_RAG_SECURITY.md) | ACL、引用约束、prompt injection 与审计边界 |

## 与其他文档目录的关系

- `project_guide/` 关注“项目是什么、当前怎么工作、模块怎么设计”。
- `../ops/` 关注部署、模型配置和排障。
- `../experiments/` 关注评测和实验记录。
- `../archive/` 只保留历史材料，不作为当前入口。

如果模块文档和 RAGFlow 专题文档出现差异，短期以 RAGFlow 专题文档为准；等代码落地后，再把稳定设计同步回模块文档。
