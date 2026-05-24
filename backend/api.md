# FastAPI Backend API 文档

整理日期：2026-05-24

本文档描述 `backend/app/router/chat.py` 当前暴露的 FastAPI 接口。服务默认挂载在 `http://localhost:8000`，聊天、会话、向量、记忆和重排序接口统一使用 `/api` 前缀。

## 1. 认证方式

除公开健康检查和少数调试接口外，业务接口通常需要 Django 用户服务签发的 JWT：

```http
Authorization: Bearer <jwt-token>
```

FastAPI 会使用与 Django 一致的 `SECRET_KEY` / `ALGORITHM` 解析 token，并通过 Django 用户服务读取用户详情；JWT 黑名单检查依赖 Redis。

## 2. Agent / RouterGraph 接口

### POST `/api/agent/query/stream`

统一在线聊天入口，返回 Server-Sent Events。RouterGraph 会根据问题分流到普通聊天、工具调用、企业知识库 RAG、安全拦截或澄清链路。

请求体：

```json
{
  "session_id": "optional-session-id",
  "query": "用户问题"
}
```

响应：`text/event-stream`，事件中会携带 `request_id`、`debug_id`、`session_id` 等调试字段。

### POST `/api/agent/router/query`

非流式 RouterGraph 调试/评测入口，返回结构化 `RouterResponse`。

请求体：

```json
{
  "session_id": "optional-session-id",
  "query": "用户问题"
}
```

典型返回字段：

```json
{
  "response": "回答内容",
  "session_id": "session-id",
  "request_id": "request-id",
  "debug_id": "debug-id"
}
```

## 3. RAG 接口

### POST `/api/rag/query`

经典上传文档 RAG 入口，用于直接调用通用 RAG 检索和摘要链路。

请求体：

```json
{
  "query": "检索问题"
}
```

## 4. 会话接口

### GET `/api/session/{session_id}`

获取当前用户指定会话的历史记录。

### DELETE `/api/session/{session_id}`

删除当前用户指定会话及其历史记录。

### GET `/api/sessions`

获取系统中的会话 ID 列表。该接口主要用于调试/管理视图。

### GET `/api/sessions/{user_id}`

获取指定用户的会话 ID 列表。接口会校验当前 JWT 用户身份，不能越权读取其他用户会话。

## 5. 长期记忆接口

### GET `/api/memories`

列出当前用户 active 长期记忆。

查询参数：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `limit` | int | `50` | 返回数量，范围 1-100 |
| `offset` | int | `0` | 分页偏移量 |

### DELETE `/api/memories/{memory_id}`

软删除当前用户指定长期记忆。删除以 MySQL `status=deleted` 为准，向量库删除为 best-effort。

> 当前 FastAPI 路由没有暴露 `GET /api/memories/search`。语义召回由 RouterGraph/长期记忆服务在问答链路内部使用；如果评测脚本需要独立搜索端点，应先补接口再把它写成已实现 API。

## 6. 向量库接口

### POST `/api/vector/add/single`

上传单个 PDF/TXT 文件并写入当前用户的向量集合。

### POST `/api/vector/add/multiple`

上传多个 PDF/TXT 文件并写入当前用户的向量集合。

### DELETE `/api/vector/clean`

删除当前用户上传的所有向量数据。

## 7. 重排序接口

### POST `/api/reorder`

独立重排序调试接口，使用本地 reranker 对输入文档排序。

请求体：

```json
{
  "query": "如何实现 RAG 系统",
  "documents": [
    "候选文档 A",
    "候选文档 B"
  ]
}
```

## 8. 健康检查

健康检查接口不使用 `/api` 前缀。

### GET `/health/live`

进程存活检查。

### GET `/health/ready`

依赖就绪检查，通常包括 MySQL、Redis 等关键依赖。

## 9. 相关文档

- Django 用户服务 API：`../DjangoUserService/api.md`
- 文档入口：`../docs/README.md`
- 项目架构图：`../docs/project_guide/NEXUSKB_ARCHITECTURE_DIAGRAMS.md`
