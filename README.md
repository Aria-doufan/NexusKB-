# NexusKB

NexusKB 是一个面向企业知识库场景的 Agentic RAG 智能问答系统。项目围绕“企业内部知识如何被可靠切分、索引、检索、过滤、评估并用于回答问题”展开，整合了 FastAPI 问答服务、LangGraph Agentic RAG workflow、Elasticsearch 企业检索后端、Chroma baseline、Django 用户服务、Vue 前端、MySQL 会话存储和 Redis 缓存。

与通用聊天应用不同，NexusKB 的重点不是单轮闲聊，而是可评估、可追踪、可迭代的企业文档问答链路：用户提出业务问题后，系统会规划检索策略，从企业知识库中召回相关资料，结合 metadata filter、上下文评估、fallback retry 和引用证据生成回答，并保留用户身份、会话历史、接口状态和调试轨迹。

## 项目定位

NexusKB 适合用于：

- 企业知识库问答与内部知识助手
- 内部制度、流程、工单、会议、代码变更和客户记录检索
- Agentic RAG workflow、检索策略规划和 fallback 机制验证
- Elasticsearch / Chroma 混合检索、metadata filter 和 reranker 评估
- 多服务 AI 应用、后端工程和 Agent 应用开发作品展示
- 智能客服、知识助手和企业搜索原型验证

## 核心能力

- Agentic RAG workflow：基于 LangGraph 编排 planner、strategy select、retrieve、context evaluation、query rewrite / decomposition / web fallback 等节点。
- 企业混合检索：Elasticsearch 作为 enterprise retrieval 默认后端，支持 dense kNN、BM25、RRF fusion 和 metadata filter；Chroma 保留为 baseline。
- Metadata-filtered retrieval：通过白名单 planner 生成 `none` / `soft` / `hard` filter 决策，支持 `source_type` 与 `doc_semantic_type` 的硬过滤、软 boost 和 hard → soft fallback。
- 文档语义类型：为企业资料标注 `policy_rule`、`issue_ticket`、`meeting_notes`、`email_thread`、`code_change` 等语义类型，并贯通准备、索引、检索、评估和 RAG context。
- 检索评估闭环：提供 EnterpriseRAG-Bench 数据准备、chunking profile 对比、Elasticsearch/Chroma 后端评估、metadata filter 实验和报告文档。
- 检索结果重排序：支持本地 reranker 模型，对初步召回片段进行二次排序。
- 会话记忆：使用 MySQL 持久化聊天历史，支持多轮对话上下文。
- 长期记忆实验：支持长期记忆抽取、去重、向量召回和评估脚本，用于验证跨会话记忆效果。
- 用户体系：独立 Django 用户服务，提供注册、登录、JWT 鉴权、Token 刷新和用户信息接口。
- 前端应用：Vue 3 前端提供登录、注册、聊天、会话、个人中心和设置页面。
- 缓存与限流：Redis 用于用户信息缓存、Token 黑名单和接口限流。
- 观测与审计：补充性能日志、debug trace、审计日志脱敏工具和评测输出，便于定位检索、记忆和回答链路问题。
- 多模型接入：支持 DashScope 云端模型，也预留 Ollama、本地 embedding 和 reranker 模型配置。

## 当前评估结果

`v0.3.0` 选择 `semantic_baseline_threshold` chunking profile，并将 Elasticsearch 作为企业检索默认后端。关键评估结果：

| 后端 / 模式 | questions | recall@10 | hit@5 | mrr@20 | ndcg@10 | avg latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chroma baseline | 500 | 0.9179 | 0.914 | 0.8593 | 0.8638 | 252.91 |
| Elasticsearch baseline | 500 | 0.9176 | 0.926 | 0.8753 | 0.8764 | 358.58 |
| ES soft `doc_semantic_type=policy_rule` | 500 | 0.9134 | 0.914 | 0.8345 | 0.8437 | 351.7 |

结论：Elasticsearch 在保持 recall parity 的同时提升 top-rank 指标；hard metadata filter 适合用户明确要求某类证据时使用，soft metadata filter 更适合隐含约束查询。

## 架构概览

![NexusKB 系统架构流程图](./docs/assets/system-architecture.png)

## 目录结构

```text
NexusKB-/
├── backend/                         # FastAPI AI/RAG 后端
│   ├── app/
│   │   ├── agent/                   # RouterGraph 兼容入口、工具 Agent 和中间件
│   │   ├── cache/                   # Redis 缓存封装
│   │   ├── config/                  # RAG、Chroma、Prompt 和模型配置
│   │   ├── core/                    # 统一响应、限流、日志等基础能力
│   │   ├── db/                      # MySQL / Redis 连接配置
│   │   ├── models/                  # 会话与消息数据模型
│   │   ├── prompt/                  # Prompt 模板
│   │   ├── rag/                     # AgenticRagGraph、RagEvidenceWorkflow、混合检索、重排序和图谱索引
│   │   ├── router/                  # FastAPI 路由与 ChatService
│   │   ├── schemas/                 # RAG、SSE、Debug Trace 等响应模型
│   │   ├── services/                # 会话记忆、长期记忆、RAG 调试追踪
│   │   └── utils/                   # 鉴权、文件处理、配置读取等工具
│   ├── scripts/                     # EnterpriseRAG-Bench 数据准备、索引构建和评测脚本
│   ├── data/                        # 本地索引、评测输出和运行数据
│   ├── tests/                       # 后端测试
│   ├── main.py                      # FastAPI 入口
│   └── requirements.txt
├── DjangoUserService/               # Django 用户服务：注册、登录、JWT、头像和用户资料
├── front/                           # Vue 3 前端
│   └── src/                         # assets、components、i18n、router、store、views
├── docs/                            # 项目指南、Agentic RAG 专题、实验、运维和归档文档
│   ├── project_guide/               # 项目总览、当前架构、模块设计和架构图
│   ├── experiments/                 # RAG 延迟优化、长期记忆评估等实验记录
│   ├── ops/                         # 部署、排障和模型配置说明
│   ├── interview/                   # 面试和项目讲述材料
│   └── archive/                     # 历史计划和阶段性记录
├── backend_learning_modules/        # 后端模块化学习样例
├── dataset/                         # EnterpriseRAG-Bench、RAGCare-QA 等数据集
├── models/                          # 本地模型权重，例如 Qwen3-Reranker
├── images/                          # 项目图片素材
├── docker-compose.elasticsearch.yml # Elasticsearch 检索评测环境
├── start-dev.ps1                    # Windows 本地开发一键启动脚本
└── requirements.txt                 # Python 聚合依赖
```

## 技术栈

后端问答服务：

- Python 3.12+
- FastAPI
- Uvicorn
- LangChain
- LangGraph
- Elasticsearch
- ChromaDB
- SQLAlchemy
- aiomysql
- Redis
- DashScope
- Ollama
- sentence-transformers
- ModelScope
- PyTorch
- unstructured

用户服务：

- Python 3.10+
- Django 5.2
- Django REST Framework
- Simple JWT
- PyMySQL / mysqlclient
- Celery
- django-redis
- drf-yasg

前端：

- Vue 3
- Vite
- Vue Router
- Pinia
- Vant
- Axios
- Vue I18n
- marked
- highlight.js
- DOMPurify

基础设施：

- MySQL
- Redis
- Elasticsearch
- ChromaDB
- DashScope API
- 可选本地模型服务

## 安装环境

建议准备：

- Python 3.12+：运行 FastAPI 问答服务
- Python 3.10+：运行 Django 用户服务
- Node.js 16+
- npm 或 pnpm
- MySQL
- Redis
- Git
- Docker Desktop 或本地 Elasticsearch 8.x

如果使用本地重排序模型，还需要准备 PyTorch 运行环境和足够的模型存储空间。

## 安装依赖

根目录提供了聚合依赖文件：

```bash
pip install -r requirements.txt
```

该文件会加载：

```text
-r backend/requirements.txt
-r DjangoUserService/requirements.txt
```

也可以分别安装：

```bash
cd backend
pip install -r requirements.txt
```

```bash
cd DjangoUserService
pip install -r requirements.txt
```

前端依赖：

```bash
cd front
npm install
```

或：

```bash
cd front
pnpm install
```

## 环境变量

FastAPI 服务：

```bash
cd backend
copy .env.example .env
```

需要配置 MySQL、Redis、JWT、DashScope 和 reranker 模型路径等参数。

Django 用户服务：

```bash
cd DjangoUserService
copy .env.example .env
```

需要配置数据库、Redis 和 `SECRET_KEY`。FastAPI 与 Django 的 JWT 密钥应保持一致，否则 FastAPI 无法校验 Django 生成的 Token。

## 启动服务

如需运行 Elasticsearch 检索评估或企业检索后端，先启动本地 Elasticsearch：

```powershell
docker compose -f docker-compose.elasticsearch.yml up -d
```

启动 MySQL 和 Redis 后，先启动用户服务：

```bash
cd DjangoUserService
python manage.py migrate
python manage.py runserver 8001
```

启动 FastAPI 问答服务：

```bash
cd backend
uvicorn main:app --reload
```

启动前端：

```bash
cd front
npm run dev
```

Windows 本地开发也可以使用根目录脚本同时拉起 Redis、Django、FastAPI 和前端：

```powershell
.\start-dev.ps1 -Migrate
```

默认使用 `NexusKB` conda 环境；如果想使用 `.venv` 或 PATH 中的 Python，可以传入 `-CondaEnv ''`。

## 数据说明

本项目使用企业知识库类数据进行 RAG 检索实验，重点关注企业文档、问题集、知识片段和检索评测结果。仓库不会直接提交原始数据、生成后的向量库或本地运行缓存。

本地运行时通常需要准备：

- 企业知识文档
- 问题集或评测集
- Elasticsearch 索引或 ChromaDB baseline 索引目录
- 可选 reranker 模型权重

相关配置位于：

```text
backend/app/config/chroma.yaml
backend/app/config/rag.yaml
```

相关脚本位于：

```text
backend/scripts/
```

其中当前企业 RAG semantic baseline 评估默认使用：

```text
backend/data/chunking_eval_outputs/stage2_semantic/semantic_baseline_threshold_p3000-o300_c700-o100/prepared/child_chunks_parent_child.jsonl
backend/data/chunking_eval_outputs/stage2_semantic/semantic_baseline_threshold_p3000-o300_c700-o100/prepared/parent_chunks_parent_child.jsonl
nexuskb_enterprise_chunks
```

长期记忆评估资产包括：

```text
backend/scripts/evaluate_long_term_memory.py
backend/scripts/memory_eval_golden_cases.jsonl
docs/experiments/memory_eval.md
```


## 未来方向

- 将 Elasticsearch 检索、metadata filter 和 agentic fallback 能力接入更多端到端用户场景。
- 完善 reranker 在 Elasticsearch 后端上的评测与对比。
- 增加回答引用来源展示和 evidence 可视化。
- 增加知识库上传、索引状态、metadata 管理和评估结果管理页面。
- 完善长期记忆的数据模型、数据库迁移、API 暴露和端到端验收。
- 增加 CI、Docker Compose 集成验证和发布前自动评测。
- 增加权限分级和文件上传安全检查。

## 更多文档

- [版本更新记录](./CHANGELOG.md)
- [GitHub Releases](https://github.com/Aria-doufan/NexusKB-/releases)
- [项目介绍](./docs/PROJECT_INTRO.md)
- [项目总览](./docs/PROJECT_OVERVIEW.md)
- [项目指南](./docs/project_guide/README.md)
- [部署说明](./docs/deployment.md)
- [运维部署说明](./docs/ops/deployment.md)
- [故障排除](./docs/troubleshooting.md)
- [Hugging Face / ModelScope 模型配置](./docs/huggingface_model.md)
- [RAGFlow Agentic RAG 架构设计](./docs/RAGFLOW_AGENTIC_RAG_ARCHITECTURE.md)
- [RAGFlow Agentic RAG 评测方案](./docs/RAGFLOW_AGENTIC_RAG_EVALUATION.md)
- [RAGFlow Agentic RAG 路线图](./docs/RAGFLOW_AGENTIC_RAG_ROADMAP.md)
- [RAGFlow Agentic RAG 安全边界](./docs/RAGFLOW_AGENTIC_RAG_SECURITY.md)
- [长期记忆评估流程](./docs/experiments/memory_eval.md)
- [企业 RAG 延迟优化记录](./docs/experiments/enterprise_rag_latency_optimization.md)
- [FastAPI API 文档](./backend/api.md)
- [Django 用户服务 API 文档](./DjangoUserService/api.md)
