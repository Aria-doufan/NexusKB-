# 后端模块设计

整理日期：2026-05-15

## 职责

`backend` 是当前主线服务，负责：

- FastAPI 应用入口和路由注册。
- Chat Agent、Router Graph、RAG 检索增强生成。
- JWT 鉴权、Redis 限流、MySQL 会话持久化。
- 会话记忆压缩与 MySQL + Chroma 长期记忆。
- 文件入库、向量检索、重排序和统一响应。

更完整的历史架构说明仍保留在 [后端技术与架构总结](../../backend/BACKEND_SUMMARY.md)。

## 当前链路

```text
Client
  -> FastAPI
  -> router/chat.py
  -> ChatService / RouterGraph / RagService / EnterpriseRagService
  -> LLM、Chroma、BM25、Redis、MySQL
  -> JSON 或 SSE 响应
```

## 核心文件

| 文件 | 说明 |
| --- | --- |
| `backend/main.py` | FastAPI 应用入口、中间件、路由注册 |
| `backend/app/router/chat.py` | 聊天、Agent、RAG、Router 相关 API |
| `backend/app/router/chat_service.py` | 业务编排服务 |
| `backend/app/agent/router_graph.py` | LangGraph Router Graph |
| `backend/app/rag/rag_service.py` | 原业务知识库 RAG |
| `backend/app/rag/enterprise_rag_service.py` | 企业评测知识库 RAG |
| `backend/app/rag/vector_store.py` | Chroma 向量库和文档入库 |
| `backend/app/rag/reorder_service.py` | 文档重排序 |
| `backend/app/services/conversation_memory.py` | 会话记忆压缩 |
| `backend/app/services/long_term_memory.py` | 长期记忆抽取、MySQL 权威存储和 Chroma 语义索引 |
| `backend/app/services/database_session_manager.py` | 会话与消息持久化 |

## 当前限制

- API 文档主要手工维护，后续应考虑 OpenAPI 导出或生成式同步。
- 企业 RAG 与原业务 RAG 并存，后续需要明确环境变量和数据目录策略。
- 长期记忆已有列表和删除 API，后续还需要前端管理页、编辑/合并和冲突处理。
- 完整 live 验证依赖 MySQL、Redis、JWT、LLM、Ollama/Chroma 等外部服务同时可用。

## 下一步

- 用 Router 分流统一 `/api/agent/query/stream`。
- 建立最小测试清单：启动、Router、RAG、会话写入、SSE。
- 把检索策略矩阵落到 `EnterpriseRagService` 或独立策略层。
