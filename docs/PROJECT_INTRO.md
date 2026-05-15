# NexusKB 项目介绍

## 1. 项目概述

NexusKB 是一个面向企业知识库的 RAG 智能问答系统。系统目标是把企业内部文档、制度资料、产品说明、流程文件等非结构化内容组织成可检索、可追问、可评估的知识问答服务。

项目由三部分组成：

- FastAPI 问答服务：负责企业知识检索、RAG 编排、会话记忆、重排序和模型调用。
- Django 用户服务：负责注册、登录、JWT 鉴权、用户资料和文件接口。
- Vue 前端：负责用户登录、注册、聊天、会话管理、个人中心和设置页面。

项目重点不在于简单调用大模型，而在于构建一条完整的企业知识问答链路：文档进入知识库，文本被切分并写入向量数据库，用户问题触发检索和重排序，最终由 LLM 结合上下文生成回答。

## 2. 项目目标

NexusKB 主要解决以下问题：

- 企业资料分散，用户难以快速定位有效信息。
- 普通大模型回答缺少企业内部依据。
- 文档问答需要保留用户身份、会话历史和权限上下文。
- RAG 系统需要可替换的 embedding、reranker 和 LLM 配置。
- 企业知识库需要后续评测、迭代和工程化部署空间。

## 3. 功能模块

### 3.1 企业知识检索

系统支持将企业文档解析为文本块，并写入 ChromaDB。用户提问时，后端会根据问题召回相关知识片段，作为模型回答的依据。

### 3.2 RAG 问答

FastAPI 服务将用户问题、历史会话、检索结果和提示词模板组合起来，形成完整的 RAG 请求链路。回答生成可以接入 DashScope，也可以根据配置扩展到其他模型服务。

### 3.3 重排序

项目保留 reranker 模块，用于对初步召回结果进行二次排序。这样可以减少低相关片段进入最终上下文，提高回答稳定性。

### 3.4 会话记忆

聊天历史存储在 MySQL 中，后端通过会话管理模块读取近期上下文，使用户可以围绕同一主题连续追问。

### 3.5 用户与鉴权

Django 用户服务提供注册、登录、Token 刷新、用户信息更新和文件相关接口。FastAPI 通过 JWT 校验用户身份，并结合 Redis 检查 Token 黑名单。

### 3.6 前端应用

Vue 前端提供完整交互入口，包括登录、注册、聊天、会话列表、个人中心和设置页面。前端使用 Pinia 管理用户状态、会话状态和界面状态。

## 4. 技术架构

```text
Vue 前端
  ├─ 登录 / 注册 / 聊天 / 会话管理
  ├─ Axios 调用 FastAPI 与 Django
  └─ Pinia 保存用户与会话状态

Django 用户服务
  ├─ 用户注册登录
  ├─ JWT 生成与刷新
  ├─ 用户资料接口
  └─ Redis Token 黑名单

FastAPI 问答服务
  ├─ 聊天接口
  ├─ JWT 鉴权
  ├─ 会话记忆
  ├─ 企业知识检索
  ├─ Reranker 重排序
  └─ LLM 生成

数据与模型层
  ├─ MySQL：用户数据与会话历史
  ├─ Redis：缓存、限流、Token 状态
  ├─ ChromaDB：向量索引
  ├─ DashScope / Ollama：模型服务
  └─ 本地 reranker：检索结果重排序
```

## 5. 技术栈

### 5.1 FastAPI 问答服务

- Python 3.12+
- FastAPI
- Uvicorn
- LangChain
- LangGraph
- ChromaDB
- SQLAlchemy
- aiomysql
- Redis
- Pydantic
- DashScope
- Ollama
- sentence-transformers
- ModelScope
- PyTorch
- unstructured
- pypdf

### 5.2 Django 用户服务

- Python 3.10+
- Django 5.2
- Django REST Framework
- Simple JWT
- PyMySQL
- mysqlclient
- Celery
- Redis
- django-redis
- drf-yasg

### 5.3 Vue 前端

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

## 6. 安装环境

建议环境：

- Python 3.12+：运行 FastAPI 问答服务
- Python 3.10+：运行 Django 用户服务
- Node.js 16+
- npm 或 pnpm
- MySQL
- Redis
- Git

如果启用本地 reranker，还需要安装与机器匹配的 PyTorch 环境，并准备模型权重目录。

## 7. 依赖安装

根目录 `requirements.txt` 聚合了两个 Python 服务的依赖：

```text
-r backend/requirements.txt
-r DjangoUserService/requirements.txt
```

可以在根目录执行：

```bash
pip install -r requirements.txt
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

## 8. 环境变量

FastAPI 服务使用：

```text
backend/.env.example
```

Django 用户服务使用：

```text
DjangoUserService/.env.example
```

需要配置的重点内容包括：

- MySQL 连接信息
- Redis 连接信息
- JWT `SECRET_KEY`
- DashScope API 配置
- Django 服务地址
- Reranker 模型路径

FastAPI 与 Django 的 JWT 密钥需要保持一致。

## 9. 启动流程

启动数据库和缓存服务后，先运行 Django 用户服务：

```bash
cd DjangoUserService
python manage.py migrate
python manage.py runserver 8001
```

再运行 FastAPI 问答服务：

```bash
cd backend
uvicorn main:app --reload
```

最后启动前端：

```bash
cd front
npm run dev
```

## 10. 数据使用

项目面向企业知识库数据。数据通常包括：

- 企业文档
- 企业问答集
- 检索评测问题
- 文档切分后的 chunk
- ChromaDB 向量索引
- RAG 评测输出

这些数据和索引一般体积较大，且可能包含本地或业务信息，因此不直接提交到仓库。

相关配置：

```text
backend/app/config/chroma.yaml
backend/app/config/rag.yaml
```

相关脚本：

```text
backend/scripts/prepare_enterprise_rag_bench.py
backend/scripts/index_enterprise_chunks_chroma.py
backend/scripts/evaluate_enterprise_retrieval.py
backend/scripts/evaluate_enterprise_hybrid_retrieval.py
```

## 11. 仓库上传范围

建议上传：

- 后端源码
- 用户服务源码
- 前端源码
- API 文档
- 部署说明
- `.env.example`
- Python 依赖文件
- 前端依赖文件
- lock 文件
- 必要项目文档

不建议上传：

- `.env`
- 虚拟环境
- `node_modules`
- 日志文件
- 数据库文件
- Redis dump
- ChromaDB 索引
- 原始数据集
- 大模型权重
- 个人文档和临时计划文档

## 12. 项目改进点

当前版本围绕企业知识库问答做了多处扩展：

- 将普通问答服务改造成企业知识检索场景。
- 增加企业 RAG 数据准备、索引构建和评测脚本。
- 整合 FastAPI 问答服务与 Django 用户服务。
- 增加 Redis 缓存、限流和 Token 黑名单检查。
- 增加会话持久化与多轮记忆模块。
- 增加 reranker 模块，支持检索结果二次排序。
- 前端补充登录、注册、个人中心、会话管理等页面。
- 增加企业知识库方向的模块文档与部署说明。

## 13. 未来方向

- 建立更完整的企业知识库评测集。
- 支持 BM25 + 向量检索的混合召回。
- 支持答案引用来源展示。
- 支持知识库上传、删除、重建索引和索引状态展示。
- 将会话记忆升级为 token-aware 机制。
- 增加 Docker Compose 部署方案。
- 增加自动化测试和 CI。
- 增强文件上传安全检查。
- 增加企业用户权限分层。

## 14. 项目总结

NexusKB 是一个以企业知识库问答为核心的 RAG 系统原型。它覆盖了从用户登录、会话管理、知识检索、重排序、模型生成到前端展示的完整链路，适合继续扩展为企业内部知识助手或智能客服平台。
