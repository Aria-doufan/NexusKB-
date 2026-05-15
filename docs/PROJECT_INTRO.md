# 智能 RAG 对话服务项目介绍

## 1. 项目概述

本项目是一个基于 **FastAPI + LangChain + Vue 3 + Django** 构建的智能问答与对话服务系统。系统围绕 RAG（Retrieval-Augmented Generation，检索增强生成）能力设计，支持用户登录认证、会话管理、知识文档检索、智能问答、上下文记忆和前端交互展示。

项目采用前后端分离与多服务协作架构：

- `backend/`：核心智能问答服务，负责 RAG 检索、LLM 调用、会话记忆、向量库管理和聊天接口。
- `DjangoUserService/`：用户服务，负责注册、登录、JWT 鉴权、用户信息和文件相关接口。
- `front/`：Vue 前端应用，负责登录注册、聊天页面、会话列表、个人中心和设置页面。
- `docs/`：项目说明、部署说明、模块设计和实验文档。
- `images/`：项目展示截图。

为了保持 GitHub 仓库轻量，仓库不上传虚拟环境、运行日志、数据库文件、向量库缓存、大模型权重和原始数据集。这些内容需要在本地运行时按需准备。

## 2. 核心功能

### 2.1 智能对话

系统提供面向用户的聊天入口，支持连续对话、上下文记忆和基于知识库的问答。后端会结合用户问题、历史会话和检索结果生成回复。

### 2.2 RAG 检索增强

后端支持将文档切分为文本块，写入 Chroma 向量数据库，并在用户提问时进行相似度检索。系统可以结合检索到的知识片段生成更贴近业务资料的回答。

### 2.3 文档处理

后端配置中支持多种知识文件类型，包括：

- `txt`
- `pdf`
- `md`
- `pptx`
- `docx`

文档会经过文本抽取、切分、向量化和索引构建流程，最终进入 RAG 检索链路。

### 2.4 重排序能力

项目预留并实现了 reranker 相关模块，可以使用本地或下载的重排序模型对初步检索结果进行二次排序，提高最终上下文质量。

### 2.5 用户服务

`DjangoUserService` 提供独立用户服务，支持：

- 用户注册
- 用户登录
- JWT Token 生成与验证
- Token 刷新
- 修改密码
- 用户信息查询与更新
- 文件上传相关接口

### 2.6 前端交互

前端使用 Vue 3 构建，包含：

- 登录页
- 注册页
- AI 聊天页
- 会话管理页
- 个人中心
- 设置页
- 中英文国际化支持
- Pinia 状态管理

## 3. 系统架构

整体架构可以分为五层：

1. 前端交互层  
   Vue 3 应用负责用户界面、路由、状态管理和接口调用。

2. 用户服务层  
   Django 服务负责用户注册登录、JWT 鉴权和用户资料管理。

3. 智能问答服务层  
   FastAPI 服务负责聊天 API、RAG 检索、Agent 调度、会话记忆和统一响应。

4. 数据与缓存层  
   MySQL 存储用户与会话数据，Redis 用于缓存、限流和 Token 黑名单，ChromaDB 用于向量检索。

5. 模型与外部服务层  
   DashScope、Ollama、LangChain、sentence-transformers、ModelScope 和本地 reranker 模型共同支撑生成、嵌入与重排序能力。

简化调用流程：

```text
用户 -> Vue 前端 -> FastAPI 聊天接口
                  -> Django 用户服务校验身份
                  -> MySQL / Redis 获取用户与会话状态
                  -> ChromaDB 检索知识片段
                  -> Reranker 重排序
                  -> LLM 生成回答
                  -> 返回前端展示
```

## 4. 技术栈

### 4.1 后端智能问答服务

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

### 4.2 用户服务

- Python 3.10+
- Django 5.2
- Django REST Framework
- Simple JWT
- PyMySQL / mysqlclient
- Celery
- Redis
- django-redis
- drf-yasg

### 4.3 前端

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

### 4.4 数据与基础设施

- MySQL：用户数据、会话数据
- Redis：缓存、限流、Token 黑名单
- ChromaDB：向量数据库
- DashScope API：云端大语言模型与部分模型服务
- Ollama：可选本地模型服务

## 5. 安装环境

### 5.1 基础环境

建议本地准备：

- Python 3.12+：用于 FastAPI 智能问答服务
- Python 3.10+：用于 Django 用户服务
- Node.js 16+ 或更高版本
- npm 或 pnpm
- MySQL
- Redis
- Git

如果使用本地模型，还需要：

- 可用的 PyTorch 运行环境
- 足够的磁盘空间存放模型权重
- 可选 GPU 环境

### 5.2 Python 依赖

根目录提供了聚合依赖文件：

```bash
pip install -r requirements.txt
```

该文件会安装：

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

项目同时保留了 `pyproject.toml` 和 `uv.lock`，如果使用 `uv`，可以在对应服务目录执行：

```bash
uv sync
```

### 5.3 前端依赖

```bash
cd front
npm install
```

或：

```bash
cd front
pnpm install
```

## 6. 环境变量配置

### 6.1 FastAPI 服务

复制示例配置：

```bash
cd backend
copy .env.example .env
```

需要重点配置：

- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_DATABASE`
- `DJANGO_API_URL`
- `SECRET_KEY`
- `ALGORITHM`
- `ALIYUN_ACCESS_KEY_SECRET`
- `ALIYUN_BASE_URL`
- `REDIS_HOST`
- `REDIS_PORT`
- `RERANKER_MODEL_PATH`

### 6.2 Django 用户服务

复制示例配置：

```bash
cd DjangoUserService
copy .env.example .env
```

需要重点配置：

- `SECRET_KEY`
- `DB_HOST`
- `DB_PORT`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`
- `REDIS_HOST`
- `REDIS_PORT`
- `REDIS_PASSWORD`

注意：FastAPI 与 Django 服务应使用一致的 `SECRET_KEY` 和 JWT 算法，否则 FastAPI 无法正确校验 Django 生成的 Token。

## 7. 启动方式

### 7.1 启动 MySQL 和 Redis

Windows 示例：

```bash
net start mysql
net start redis
```

如果 Redis 是解压版，也可以直接运行：

```bash
redis-server
```

### 7.2 启动 Django 用户服务

```bash
cd DjangoUserService
python manage.py migrate
python manage.py runserver 8001
```

用户服务默认运行在：

```text
http://127.0.0.1:8001
```

### 7.3 启动 FastAPI 智能问答服务

```bash
cd backend
uvicorn main:app --reload
```

智能问答服务默认运行在：

```text
http://127.0.0.1:8000
```

### 7.4 启动前端

```bash
cd front
npm run dev
```

或：

```bash
cd front
pnpm run dev
```

前端默认由 Vite 启动，实际访问地址以终端输出为准。

## 8. 数据使用说明

项目运行需要区分三类数据：

### 8.1 用户与会话数据

用户、登录状态和会话历史主要存储在 MySQL 中。Django 用户服务负责用户数据，FastAPI 后端负责聊天会话相关数据。

### 8.2 知识库数据

知识文件默认由后端配置管理，相关配置位于：

```text
backend/app/config/chroma.yaml
```

其中重要字段包括：

- `data_path`：知识文件目录
- `persist_directory`：ChromaDB 持久化目录
- `allow_knowledge_file_types`：允许导入的文件类型
- `chunk_size`：文本切分长度
- `chunk_overlap`：文本块重叠长度
- `collection_name`：向量库集合名称

当前仓库不会上传 `backend/data/`，因为其中通常包含本地知识库文件、向量索引和运行数据。部署或复现时需要自行准备数据并重新构建索引。

### 8.3 模型文件

本地 reranker 或 embedding 模型通常体积较大，例如 Qwen reranker 权重不适合直接提交到 GitHub。仓库不会上传 `models/` 目录。

如果需要使用本地模型，可以：

1. 按文档下载模型到本地目录。
2. 在 `.env` 中配置 `RERANKER_MODEL_PATH`。
3. 确认后端代码可以访问该路径。

模型相关说明可参考：

```text
docs/huggingface_model.md
backend/README_RERANKER.md
backend/scripts/download_qwen3_reranker_modelscope.py
```

## 9. GitHub 上传策略

为了保持仓库干净，以下内容不建议上传：

- `.env` 和任何真实密钥
- `.venv/`、`venv/` 等虚拟环境
- `node_modules/`
- `dist/`
- `__pycache__/`
- 日志文件
- 本地数据库文件
- Redis `dump.rdb`
- ChromaDB 向量库文件
- 原始数据集和训练集
- 大模型权重
- 个人简历、计划安排、工作板和临时文档

应该上传：

- 后端源码
- 前端源码
- 用户服务源码
- API 文档
- 部署文档
- `.env.example`
- `requirements.txt`
- `package.json`
- lock 文件
- 项目截图和必要说明文档

## 10. 未来方向

后续可以从以下几个方向继续完善：

### 10.1 RAG 效果优化

- 引入更稳定的 embedding 模型
- 完善 reranker 策略
- 支持 parent-child chunk 检索
- 增加混合检索：BM25 + 向量检索
- 建立标准化 RAG 评测集
- 记录召回率、命中率、回答准确率等指标

### 10.2 会话记忆增强

- 从固定轮数记忆升级为 token-aware 记忆窗口
- 增加长期记忆摘要
- 支持用户级个性化记忆
- 区分短期上下文、长期偏好和知识库事实

### 10.3 工程化与部署

- 增加 Docker Compose 部署方案
- 增加 CI 检查
- 增加自动化测试
- 增加生产环境配置模板
- 完善日志追踪和错误告警
- 统一 FastAPI 与 Django 的配置管理方式

### 10.4 权限与安全

- 完善 JWT 过期、刷新和黑名单机制
- 增加接口权限分级
- 增加文件上传安全检查
- 移除所有生产环境默认密钥
- 增加敏感配置扫描流程

### 10.5 前端体验

- 优化聊天流式输出
- 增加知识引用展示
- 增加会话搜索和归档
- 增加文档上传与索引状态展示
- 增加错误提示和加载状态
- 提升移动端适配体验

## 11. 项目定位

本项目适合作为企业知识库问答、医疗问答实验、课程设计、毕业设计或智能客服系统原型。它已经具备完整的前端、用户服务、智能问答服务和 RAG 检索链路，后续可以继续向更完整的企业级知识助手方向演进。
