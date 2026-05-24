# 部署指南

整理日期：2026-05-24

本文档描述 NexusKB 当前仓库可用的部署和启动方式。项目由三个服务组成：FastAPI 后端、Django 用户服务、Vue/Vite 前端，并依赖 MySQL、Redis、Chroma 数据目录和可选 reranker 模型。

## 1. 服务组成

| 服务 | 默认端口 | 主要职责 |
| --- | --- | --- |
| FastAPI backend | `8000` | Chat API、RouterGraph、RAG、长期记忆、向量库、SSE |
| DjangoUserService | `8001` | 注册登录、JWT、用户资料、文件/头像接口 |
| Vue/Vite front | `5173` | Web UI、聊天、会话、用户页面 |
| MySQL | `3306` | 用户、会话、消息、长期记忆结构化数据 |
| Redis | `6379` | JWT 黑名单、缓存、限流、健康检查依赖 |
| Chroma | 本地目录 | 用户文档、企业知识库、长期记忆向量 |

## 2. Python 环境

本项目本地开发约定使用 conda 环境 `nexuskb`：

```powershell
conda activate nexuskb
$env:PYTHONUTF8='1'
```

如果在自动脚本或非交互 shell 中运行，优先使用直接解释器路径：

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda\envs\nexuskb\python.exe -m pytest backend\tests
```

## 3. FastAPI 后端环境变量

FastAPI 后端读取 `backend/.env`。当前代码实际使用的关键变量包括：

```env
# LLM / Router / Agent
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# MySQL
MYSQL_USER=root
MYSQL_PASSWORD=change-me
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=chat_history

# Django 用户服务
DJANGO_API_URL=http://127.0.0.1:8001

# JWT，必须与 Django 用户服务保持一致
SECRET_KEY=change-me
ALGORITHM=HS256

# Reranker，可选
RERANKER_MODEL_PATH=D:\Hugging_Face\models\Qwen3-Reranker-0.6B

# 长期记忆，可选阈值
LONG_TERM_MEMORY_MIN_RELEVANCE=0.35
LONG_TERM_MEMORY_DUPLICATE_RELEVANCE=0.92
```

注意：`backend/app/db/redis_config.py` 当前使用代码常量 `localhost:6379 db=3`，不是从 `.env` 读取 `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB`。如果需要生产可配置 Redis，应先调整代码再更新部署说明。

## 4. Django 用户服务环境变量

Django 服务使用自己的 `.env` / settings 配置。需要保证：

- 数据库连接指向同一 MySQL 实例或约定的用户服务库。
- JWT `SECRET_KEY` 和 `ALGORITHM` 与 FastAPI 一致。
- Redis 可用于 token 黑名单和缓存。
- 媒体文件目录有读写权限。

## 5. 本地启动顺序

### 5.1 启动依赖

确保 MySQL 和 Redis 已启动，并确认 Chroma 数据目录可读写。

### 5.2 启动 Django 用户服务

```powershell
conda activate nexuskb
$env:PYTHONUTF8='1'
cd DjangoUserService
python manage.py migrate
python manage.py runserver 8001
```

### 5.3 启动 FastAPI 后端

```powershell
conda activate nexuskb
$env:PYTHONUTF8='1'
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 5.4 启动前端

```powershell
cd front
npm install
npm run dev
```

前端开发代理通常将：

- `/api/*` 转发到 FastAPI。
- `/user/*`、`/file/*` 转发到 DjangoUserService。

## 6. 生产部署建议

### 6.1 FastAPI

可以使用 Uvicorn worker 或 ASGI 进程管理器部署：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

如果使用 Gunicorn，请先确认生产镜像/环境中已安装 `gunicorn`，当前仓库不把它作为强制依赖。

### 6.2 前端

```bash
cd front
npm run build
```

将 `front/dist` 交给 Nginx 或静态文件服务，并配置 `/api`、`/user`、`/file` 反向代理。

### 6.3 Nginx 代理示例

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /path/to/front/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /user/ {
        proxy_pass http://127.0.0.1:8001/user/;
    }

    location /file/ {
        proxy_pass http://127.0.0.1:8001/file/;
    }
}
```

## 7. Docker 说明

当前仓库没有维护完整 Dockerfile / docker-compose 作为权威部署入口。历史文档中的 Docker 配置只能作为模板参考，不能视为当前可直接运行的生产部署方案。

## 8. 健康检查

FastAPI：

```text
GET http://127.0.0.1:8000/health/live
GET http://127.0.0.1:8000/health/ready
```

Django：

```text
GET http://127.0.0.1:8001/user/detail/
```

需要带 JWT 的接口请使用 Django 登录获取 token 后再访问。

## 9. 安全建议

- 生产环境必须启用 HTTPS。
- FastAPI 与 Django 的 JWT 配置必须一致，并使用强随机 `SECRET_KEY`。
- Redis 和 MySQL 不应暴露到公网。
- 上传目录、Chroma 数据目录和 debug trace 目录需要按用户/服务账号限制权限。
- RAG debug trace 可能包含用户问题和知识库片段，生产环境应纳入日志保留和脱敏策略。
