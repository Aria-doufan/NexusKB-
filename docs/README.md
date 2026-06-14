# NexusKB 文档中心

整理日期：2026-06-14

这里是项目文档的统一入口。当前文档按“项目指南、Agentic RAG 专题、运维、实验、面试材料、图片资产、历史归档”维护，新增文档请优先放入对应目录，避免根目录继续膨胀。

## 推荐阅读顺序

1. [项目指南](./project_guide/README.md)：项目定位、当前架构、模块设计和流程图入口。
2. [版本更新记录](./VERSION_UPDATES.md)：项目级阶段版本、能力演进和后续维护规则。
3. [当前架构快照](./project_guide/CURRENT_ARCHITECTURE_REVIEW.md)：当前真实系统链路、技术栈和模块边界。
4. [项目架构图集](./project_guide/NEXUSKB_ARCHITECTURE_DIAGRAMS.md)：核心服务、数据流、RAG、记忆、安全等 Mermaid 图。
5. [项目流程图总览](./project_guide/NEXUSKB_PROJECT_FLOWCHARTS.md)：整体流程和各模块流程图。
6. [RAGFlow Agentic RAG 路线图](./RAGFLOW_AGENTIC_RAG_ROADMAP.md)：Agentic RAG 后续演进和任务状态。
7. [部署指南](./ops/deployment.md) 与 [故障排除](./ops/troubleshooting.md)：运行、部署和排错。

## 文档分层

| 层级 | 位置 | 用途 |
| --- | --- | --- |
| 项目入口 | `README.md`、`docs/README.md` | 面向第一次进入项目的人，说明从哪里开始读 |
| 项目指南 | `docs/project_guide/` | 项目是什么、当前怎么工作、模块如何划分 |
| 模块设计 | `docs/project_guide/modules/` | 后端、RouterGraph、RAG、记忆、前端/用户服务等模块说明 |
| Agentic RAG 专题 | `docs/RAGFLOW_AGENTIC_RAG_*.md` | Agentic RAG 架构、路线图、评估和安全设计 |
| 运维手册 | `docs/ops/` | 部署、模型配置、排障和环境说明 |
| 实验记录 | `docs/experiments/` | 可复现实验配置、指标、结果和结论 |
| 面试材料 | `docs/interview/` | 项目讲述、岗位匹配和学习路线材料 |
| 图片资产 | `docs/assets/`、`docs/images/` | 架构图、截图和文档插图 |
| 历史归档 | `docs/archive/` | 已被当前文档吸收或仅作追溯的历史材料 |

## 当前主文档

### 项目指南

- [项目指南入口](./project_guide/README.md)
- [版本更新记录](./VERSION_UPDATES.md)
- [项目总览](./project_guide/PROJECT_OVERVIEW.md)
- [项目介绍](./project_guide/PROJECT_INTRO.md)
- [当前架构快照](./project_guide/CURRENT_ARCHITECTURE_REVIEW.md)
- [项目架构图集](./project_guide/NEXUSKB_ARCHITECTURE_DIAGRAMS.md)
- [项目流程图总览](./project_guide/NEXUSKB_PROJECT_FLOWCHARTS.md)
- [模块设计索引](./project_guide/modules/README.md)

### Agentic RAG / RAGFlow

- [执行路线图](./RAGFLOW_AGENTIC_RAG_ROADMAP.md)
- [架构设计](./RAGFLOW_AGENTIC_RAG_ARCHITECTURE.md)
- [评估设计](./RAGFLOW_AGENTIC_RAG_EVALUATION.md)
- [安全设计](./RAGFLOW_AGENTIC_RAG_SECURITY.md)

### 评估和实验

- [企业 RAG 延迟优化实验](./experiments/enterprise_rag_latency_optimization.md)
- [长期记忆评估实验](./experiments/memory_eval.md)

### 运维和排障

- [部署指南](./ops/deployment.md)
- [故障排除](./ops/troubleshooting.md)
- [Hugging Face 模型配置](./ops/huggingface_model.md)

## API 文档

- [FastAPI 后端 API](../backend/api.md)
- [Django 用户服务 API](../DjangoUserService/api.md)

`front/api.md` 与 Django 用户服务 API 内容重复时，以 `DjangoUserService/api.md` 为准。

## 文档维护规则

- 新模块文档写入 `docs/project_guide/modules/`。
- 新运维/部署/排障文档写入 `docs/ops/`。
- 新实验记录写入 `docs/experiments/`，必须包含运行命令、配置、指标、结果和结论。
- 面试、项目讲述和岗位匹配材料写入 `docs/interview/`。
- 架构图、截图和文档插图写入 `docs/assets/` 或 `docs/images/`。
- 旧计划、阶段性备忘和已被吸收的内容写入 `docs/archive/`，不要作为当前入口。
- 每次调整架构、路由、数据流或模型策略，都要同步更新对应模块文档和图集。
