# 项目文档中心

整理日期：2026-05-15

这里是项目文档的统一入口。后续新增文档优先放在 `docs/` 下，并按本文的分类维护，避免计划、设计、实验和临时记录混在一起。

## 推荐阅读顺序

1. [项目总览](./PROJECT_OVERVIEW.md)：先看项目定位、系统边界、主线能力和文档原则。
2. [工作台账](./WORKBOARD.md)：查看已经完成、正在进行、接下来要做的工作。
3. [模块设计索引](./modules/README.md)：进入每个模块的详细设计。
4. [实验记录](./experiments/enterprise_retrieval_eval.md)：查看企业 RAG 检索评测结果。
5. [部署指南](./deployment.md) 与 [故障排除](./troubleshooting.md)：运行、部署、排错时使用。

## 文档分层

| 层级 | 位置 | 用途 |
| --- | --- | --- |
| 项目入口 | `README.md`、`docs/README.md` | 面向第一次进入项目的人，说明项目是什么、怎么跑、文档从哪里看 |
| 项目大纲 | `docs/PROJECT_OVERVIEW.md` | 说明项目定位、目标架构、当前边界、文档治理方式 |
| 工作台账 | `docs/WORKBOARD.md` | 记录已完成、进行中、下一步、风险与决策 |
| 模块设计 | `docs/modules/` | 分模块记录设计、职责、链路、核心文件、待改进项 |
| 实验记录 | `docs/experiments/` | 记录可复现实验配置、指标、结果、结论 |
| 运维手册 | `docs/deployment.md`、`docs/troubleshooting.md`、`docs/huggingface_model.md` | 部署、模型、排障和环境配置 |
| 历史归档 | `docs/archive/` | 保留阶段性计划、对话记录、旧版任务记录，作为追溯材料 |

## 当前主文档

- [项目总览](./PROJECT_OVERVIEW.md)
- [工作台账](./WORKBOARD.md)
- [项目验收标准](./ACCEPTANCE_CRITERIA.md)
- [模块设计索引](./modules/README.md)
- [企业 RAG 检索评测实验](./experiments/enterprise_retrieval_eval.md)
- [企业聊天助手定位与能力差距分析](./enterprise_chat_assistant_positioning.md)
- [项目学习指南](./project_learning_guide.md)
- [部署指南](./deployment.md)
- [故障排除](./troubleshooting.md)
- [Hugging Face 模型配置](./huggingface_model.md)

## API 文档

- [FastAPI 后端 API](../backend/api.md)
- [Django 用户服务 API](../DjangoUserService/api.md)

`front/api.md` 与 Django 用户服务 API 内容重复，后续维护时以 `DjangoUserService/api.md` 为准。

## 文档维护规则

- 新任务先进 `WORKBOARD.md` 的“待办/下一步”，完成后移动到“已完成”。
- 模块设计写到 `docs/modules/`，不要塞进工作台账。
- 实验必须记录运行命令、固定配置、指标定义、结果文件和结论。
- 临时讨论、旧计划、阶段性备忘放入 `docs/archive/`，不要作为当前入口。
- 每次调整架构、路由、数据流或模型策略，都要同步更新对应模块文档。
