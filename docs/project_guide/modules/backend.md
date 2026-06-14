# 后端模块设计

整理日期：2026-06-14

## 职责

`backend` 是当前主线服务，负责：

- FastAPI 应用入口和路由注册。
- Chat Agent、RouterGraph 兼容入口、AgenticRagGraph 主状态机和 RAG evidence workflow。
- JWT 鉴权、Redis 限流、MySQL 会话持久化。
- 文件入库、混合检索、RRF 融合、reranker、图谱索引和统一响应。
- 会话记忆、长期记忆、RAG debug trace 和离线评测脚本。

更完整的历史架构说明仍保留在 [后端技术与架构总结](../../../backend/BACKEND_SUMMARY.md)。

## 当前链路

```text
Client
  -> FastAPI
  -> router/chat.py
  -> ChatService
  -> RouterGraph 兼容入口
  -> AgenticRagGraph 主状态机
      -> direct_answer / retrieve / tool_call / clarify / refuse
      -> RagEvidenceWorkflow
      -> Chroma + BM25 + RRF + reranker + graph index + web fallback
  -> MySQL 会话记忆、Redis 限流缓存、RAG debug trace
  -> JSON 或 SSE 响应
```

## 核心文件

| 文件 | 说明 |
| --- | --- |
| `backend/main.py` | FastAPI 应用入口、中间件、路由注册 |
| `backend/app/router/chat.py` | 聊天、Agent、RAG、Router 相关 API |
| `backend/app/router/chat_service.py` | API 业务编排服务 |
| `backend/app/agent/router_graph.py` | 兼容旧 RouterGraph API，实际委托 `AgenticRagGraph` |
| `backend/app/rag/agentic_rag_graph.py` | LangGraph 主状态机，统一 direct/retrieve/tool/clarify/refuse 分支 |
| `backend/app/rag/rag_evidence_workflow.py` | 企业 RAG 证据工作流，编排检索、评估、重试、生成和 trace finalization |
| `backend/app/rag/retrieval_pipeline.py` | Chroma、BM25、RRF、source hint boost 等混合检索流水线 |
| `backend/app/rag/strategy_router.py` | reranker、decompose、retry、web fallback 等策略选择 |
| `backend/app/rag/graph_extraction.py` | 从文档中抽取实体和关系 |
| `backend/app/rag/graph_index_service.py` | 企业知识图谱索引构建与查询支撑 |
| `backend/app/rag/vector_store.py` | Chroma 向量库和文档入库 |
| `backend/app/rag/reorder_service.py` | 文档重排序 |
| `backend/app/services/conversation_memory.py` | 会话记忆压缩 |
| `backend/app/services/long_term_memory.py` | 长期记忆抽取、去重和召回 |
| `backend/app/services/rag_debug_trace_store.py` | RAG debug trace 存储 |
| `backend/app/services/database_session_manager.py` | 会话与消息持久化 |

## 当前限制

- `RouterGraph` 已经收敛为兼容层，但仍需要持续清理旧入口表述和调用路径。
- 普通聊天、企业 RAG、工具调用和拒绝/澄清分支已经在 `AgenticRagGraph` 中统一，但仍需要用测试覆盖各分支。
- API 文档主要手工维护，后续应考虑 OpenAPI 导出或生成式同步。
- 企业 RAG、原业务 RAG、图谱索引和 Elasticsearch 评测路径并存，后续需要继续明确环境变量、数据目录和默认检索后端策略。

## 下一步

- 建立最小测试清单：启动、RouterGraph 兼容入口、AgenticRagGraph 分支、RAG evidence workflow、会话写入、SSE。
- 补齐 RAG debug trace、RAG Response Schema 和 SSE Event Schema 的接口文档同步。
- 明确 Chroma、Elasticsearch、Graph Index 和 web fallback 在不同环境下的默认启用策略。
