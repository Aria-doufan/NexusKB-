# 模块设计索引

本目录按模块记录详细设计，面向人阅读。模块文档应该说明“为什么这样设计、当前怎么工作、核心文件在哪里、还有什么限制”，不要只写流水账。

模块做到什么程度算完成，项目级标准参考：[项目验收标准](../../ACCEPTANCE_CRITERIA.md)，模块级标准参考：[核心模块最小验收](./minimum_acceptance_checklist.md)。

## 当前模块

| 模块 | 文档 | 说明 |
| --- | --- | --- |
| 后端总览 | [backend.md](./backend.md) | FastAPI 主后端结构、核心职责和运行依赖 |
| Chat Agent MVP | [chat-agent-mvp.md](./chat-agent-mvp.md) | 普通聊天、工具调用、企业知识库路由边界 |
| Agentic RAG 主入口 | [agent-router.md](./agent-router.md) | RouterGraph 兼容层、AgenticRagGraph 状态机和 RAG evidence workflow 边界 |
| 企业 RAG 与检索 | [rag-retrieval.md](./rag-retrieval.md) | parent-child 分块、混合召回、reranker 策略 |
| 会话记忆 | [conversation-memory.md](./conversation-memory.md) | Working Memory、Session Memory 与压缩策略 |
| 长期记忆模块 | [memory-module.md](./memory-module.md) | 长期记忆抽取、存储、召回、去重和删除链路 |
| 记忆模块版本档案 | [memory-versions.md](./memory-versions.md) | v1/v2/v3 记忆机制、能力边界和改进对比 |
| 前端与用户服务 | [frontend-user-service.md](./frontend-user-service.md) | Vue 前端与 Django 用户服务边界 |
| 核心模块最小验收 | [minimum_acceptance_checklist.md](./minimum_acceptance_checklist.md) | 每个核心模块的接口返回、延迟、错误兜底和测试命令 |
| 权限、安全与审计 | [security_audit_design.md](./security_audit_design.md) | 工具调用边界、企业数据访问边界和审计事件设计 |

## RAGFlow 专题文档

这些文档位于 `docs/` 顶层，用于指导 Agentic RAG / RAGFlow 改造：

| 专题 | 文档 | 说明 |
| --- | --- | --- |
| 执行路线图 | [RAGFLOW_AGENTIC_RAG_ROADMAP.md](../../RAGFLOW_AGENTIC_RAG_ROADMAP.md) | 阶段、状态、优先级和下一步任务 |
| 架构设计 | [RAGFLOW_AGENTIC_RAG_ARCHITECTURE.md](../../RAGFLOW_AGENTIC_RAG_ARCHITECTURE.md) | Agentic RAG 主循环、状态、检索、调试和生成链路 |
| 评估设计 | [RAGFLOW_AGENTIC_RAG_EVALUATION.md](../../RAGFLOW_AGENTIC_RAG_EVALUATION.md) | 企业 RAG 评估集、检索指标、Evidence Coverage 和实验方法 |
| 安全设计 | [RAGFLOW_AGENTIC_RAG_SECURITY.md](../../RAGFLOW_AGENTIC_RAG_SECURITY.md) | ACL、引用约束、prompt injection、审计和最小安全边界 |

## 文档维护规则

- 新模块文档优先放在本目录。
- 已被本目录吸收的旧 `docs/modules/` 文档不再作为入口。
- 模块文档描述当前稳定实现；探索性设计、实验结果和阶段计划分别放到 `docs/RAGFLOW_AGENTIC_RAG_*.md`、`docs/experiments/`、`docs/superpowers/`。
