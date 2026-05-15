# 项目总览

整理日期：2026-05-15

## 项目定位

本项目当前主线是：

> 基于 FastAPI + LangChain/LangGraph 的企业知识库 RAG Chat Agent 服务。

项目不只追求“能问答”，而是要形成一套可持续演进的工程闭环：

- 能接收普通聊天、企业知识问答和工具调用类请求。
- 能基于企业知识库进行可解释检索和生成。
- 能用 EnterpriseRAG-Bench 做可量化评测。
- 能记录会话记忆并支撑多轮上下文。
- 能逐步加入权限、安全、观测和长期记忆能力。

## 当前系统边界

| 模块 | 当前定位 | 主要目录 |
| --- | --- | --- |
| FastAPI 主后端 | Chat Agent、RAG、Router、会话、限流、统一响应 | `backend/` |
| Django 用户服务 | 用户注册、登录、Token、文件上传 | `DjangoUserService/` |
| Vue 前端 | 移动端风格聊天、登录注册、会话管理、个人页 | `front/` |
| 企业 RAG 数据与评测 | EnterpriseRAG-Bench 数据处理、入库、检索评测 | `backend/scripts/`、`backend/data/enterprise_rag_bench/` |
| 学习模块 | 后端模块化学习样例 | `backend_learning_modules/` |
| DSagent 参考实现 | 作为 LangGraph/Agent 设计参考，不作为当前主线直接迁移 | `DSagent/` |

## 当前主线能力

- Router Graph：已经具备 `rag_query`、`agent_tool_call`、`chat`、`system`、`clarify` 路由。
- 会话记忆：已经接入 Working Memory 与 Session Memory 两层记忆压缩。
- 企业 RAG：已经完成 parent-child 分块、Chroma 入库、BM25 混合召回、Qwen3 reranker 评测。
- 检索评测：已经形成 `hit@K`、`recall@K`、`mrr@K`、延迟等指标记录。
- 基础工程：FastAPI、Redis、MySQL、JWT、SSE、统一响应和异常处理已具备基础形态。

## 目标架构

```text
前端/客户端
  -> FastAPI Chat API
  -> Router Graph
       -> 普通聊天链路
       -> 企业知识库 RAG 链路
       -> 安全工具调用链路
       -> 澄清/拒绝/系统保护链路
  -> 会话记忆与持久化
  -> 观测、评测与持续优化
```

RAG 链路的目标形态：

```text
用户问题
  -> 意图识别和上下文补全
  -> Chroma 向量召回 + BM25 关键词召回
  -> RRF 融合
  -> 按策略决定是否启用 reranker
  -> parent chunk 回填
  -> grounded answer 生成
  -> 指标与失败样例记录
```

## 文档体系建议

你的想法是对的：项目文档应该同时承担“方向盘”和“航海日志”的角色。现在这套结构建议再补三个东西：

1. 决策记录：重要取舍单独记录，例如“为什么 reranker 不默认无条件启用”。
2. 验收标准：每个阶段不只写要做什么，也写做到什么算完成。
3. 风险清单：把权限、安全、模型延迟、数据质量、评测偏差这些问题持续挂出来。

当前这些内容先放在 [工作台账](./WORKBOARD.md) 里维护；如果后续变多，再拆成 `docs/decisions/` 和 `docs/risks.md`。
