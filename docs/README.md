# NexusKB 文档中心

整理日期：2026-05-23

本目录是 NexusKB 的项目文档入口。文档目标是让第一次进入仓库的人能快速理解：项目是什么、怎么分层、核心模块在哪里、如何运行、哪些内容不能提交到 GitHub。

## 推荐阅读顺序

1. [项目大纲与系统框架](./PROJECT_OVERVIEW.md)：理解项目定位、系统边界、总体架构和后续方向。
2. [模块设计索引](./modules/README.md)：进入后端、Router、RAG、记忆、前端与用户服务等模块说明。
3. [后端技术与架构总结](../backend/BACKEND_SUMMARY.md)：查看 FastAPI 后端的技术栈、接口和数据流。
4. [FastAPI API 文档](../backend/api.md)：查看后端接口说明。
5. [Django 用户服务 API](../DjangoUserService/api.md)：查看用户服务接口说明。
6. [部署指南](./deployment.md) 与 [故障排除](./troubleshooting.md)：运行和排错时阅读。
7. [实验记录](./experiments/enterprise_retrieval_eval.md)：查看企业 RAG 检索评测记录。

## 文档分层

| 层级 | 位置 | 用途 |
| --- | --- | --- |
| 项目入口 | `README.md`、`docs/README.md` | 面向 GitHub 读者的项目说明、运行入口和安全提醒。 |
| 项目大纲 | `docs/PROJECT_OVERVIEW.md` | 项目定位、系统边界、总体框架、核心模块和演进方向。 |
| 模块设计 | `docs/modules/` | 分模块说明职责、链路、核心文件、数据结构和限制。 |
| API 文档 | `backend/api.md`、`DjangoUserService/api.md` | HTTP 接口说明、请求参数和响应格式。 |
| 实验记录 | `docs/experiments/` | 记录 RAG、检索、重排序等实验配置、命令和结果。 |
| 运维文档 | `docs/deployment.md`、`docs/troubleshooting.md`、`docs/huggingface_model.md` | 部署、模型配置和常见问题。 |

## 当前主文档

- [项目大纲与系统框架](./PROJECT_OVERVIEW.md)
- [模块设计索引](./modules/README.md)
- [后端模块设计](./modules/backend.md)
- [LangGraph Router 设计](./modules/agent-router.md)
- [会话记忆设计](./modules/conversation-memory.md)
- [长期记忆设计](./modules/long-term-memory.md)
- [企业 RAG 与检索](./modules/rag-retrieval.md)
- [前端与用户服务](./modules/frontend-user-service.md)
- [企业 RAG 检索评测实验](./experiments/enterprise_retrieval_eval.md)
- [部署指南](./deployment.md)
- [故障排除](./troubleshooting.md)

## 文档维护规则

- 架构、路由、数据流、存储策略发生变化时，同步更新 `README.md`、`docs/PROJECT_OVERVIEW.md` 和对应模块文档。
- API 变化时，同步更新 `backend/api.md` 或 `DjangoUserService/api.md`。
- 实验类内容必须记录运行命令、固定配置、指标定义、结果和结论。
- 不把真实 `.env`、个人材料、简历、企业私有数据、模型权重、向量库目录或运行缓存放入文档或提交到 GitHub。
- 临时计划和个人草稿不作为公开文档入口；需要保留时应放在本地或归档目录，并确保 `.gitignore` 生效。
