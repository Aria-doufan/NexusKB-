# 模块设计索引

本目录按模块记录 NexusKB 的系统设计。每个模块文档应回答四个问题：它负责什么、请求如何流动、核心文件在哪里、当前还有什么限制。

## 当前模块

| 模块 | 文档 | 说明 |
| --- | --- | --- |
| 后端总览 | [backend.md](./backend.md) | FastAPI 主后端结构、核心职责、运行依赖和主要服务。 |
| Chat Agent MVP | [chat-agent-mvp.md](./chat-agent-mvp.md) | 普通聊天、工具调用、企业知识库路由边界。 |
| Agent Router | [agent-router.md](./agent-router.md) | LangGraph RouterGraph 设计、State 定义、节点职责和接口链路。 |
| 会话记忆 | [conversation-memory.md](./conversation-memory.md) | Working Memory、Session Memory、摘要压缩和 RouterGraph 接入。 |
| 长期记忆 | [long-term-memory.md](./long-term-memory.md) | MySQL + Chroma 双存储长期记忆、用户隔离、语义召回和管理 API。 |
| 企业 RAG 与检索 | [rag-retrieval.md](./rag-retrieval.md) | 文档切分、Chroma、BM25、混合召回、reranker 策略。 |
| 前端与用户服务 | [frontend-user-service.md](./frontend-user-service.md) | Vue 前端、Django 用户服务、JWT 和服务边界。 |

## 模块关系

```text
front/
  -> DjangoUserService/：登录、注册、JWT、用户信息
  -> backend/：聊天、RAG、会话、长期记忆、文档上传

backend/
  -> RouterGraph：统一请求分流
  -> Agent / Chat：生成回答和工具调用
  -> Enterprise RAG：企业知识库检索
  -> Conversation Memory：当前会话摘要
  -> Long-term Memory：跨会话用户事实
  -> MySQL / Redis / Chroma：持久化、缓存和向量检索
```

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

## 公开仓库注意事项

模块文档可以描述配置项名称、接口路径、目录结构和架构取舍，但不要写入真实密钥、真实数据库地址、私有数据样例、简历内容或个人身份信息。
