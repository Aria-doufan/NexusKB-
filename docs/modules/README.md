# 模块设计索引

本目录按模块记录详细设计。模块文档应该说明“为什么这样设计、当前怎么工作、核心文件在哪里、还有什么限制”，不要只写流水账。

模块做到什么程度算完成，统一参考：[项目验收标准](../ACCEPTANCE_CRITERIA.md)。

## 当前模块

| 模块 | 文档 | 说明 |
| --- | --- | --- |
| 后端总览 | [backend.md](./backend.md) | FastAPI 主后端结构、核心职责和运行依赖 |
| Chat Agent MVP | [chat-agent-mvp.md](./chat-agent-mvp.md) | 普通聊天、工具调用、企业知识库路由边界 |
| Agent Router | [agent-router.md](./agent-router.md) | LangGraph Router Graph 设计、状态和节点职责 |
| 会话记忆 | [conversation-memory.md](./conversation-memory.md) | Working Memory、Session Memory 与压缩策略 |
| 企业 RAG 与检索 | [rag-retrieval.md](./rag-retrieval.md) | parent-child 分块、混合召回、reranker 策略 |
| 前端与用户服务 | [frontend-user-service.md](./frontend-user-service.md) | Vue 前端与 Django 用户服务边界 |

## 写作模板

新模块文档建议使用以下结构：

```markdown
# 模块名称

## 职责
## 当前链路
## 核心文件
## 数据结构或接口
## 已完成
## 当前限制
## 下一步
```
