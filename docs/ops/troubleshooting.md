# 故障排除

整理日期：2026-05-24

本文档记录 NexusKB 当前常见运行问题。部署和启动流程见 [部署指南](./deployment.md)。

## 1. LLM / API Key 错误

**现象**：RouterGraph、Agent 或 RAG 生成时报 `Invalid API Key`、`401`、模型连接失败。

**处理**：

- 检查 `backend/.env` 中的 `DEEPSEEK_API_KEY`。
- 检查 `DEEPSEEK_BASE_URL`，本地 Ollama 兼容接口通常类似 `http://localhost:11434`，DeepSeek 官方接口通常类似 `https://api.deepseek.com`。
- 检查 `DEEPSEEK_MODEL` 是否与服务端可用模型一致。
- 如果测试只需要导入模块，可在测试环境设置占位 key，但真实调用必须使用有效 key。

## 2. MySQL 连接失败

**现象**：`OperationalError`、`Can't connect to MySQL server`、会话历史或长期记忆无法读写。

**处理**：

- 检查 MySQL 服务是否启动。
- 检查 `MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_DATABASE`。
- 确认数据库存在，用户有建表、读写权限。
- FastAPI 会在 `backend/app/db/db_config.py` 使用这些变量创建 SQLAlchemy async engine。

## 3. Redis 不可用

**现象**：JWT 黑名单检查、限流或 readiness 失败，接口返回 `503 Redis unavailable`。

**处理**：

- 确认 Redis 服务运行在 `localhost:6379`。
- 当前 FastAPI Redis 配置使用 `backend/app/db/redis_config.py` 中的常量 `db=3`。
- 如果 `.env` 中配置了 `REDIS_HOST` / `REDIS_DB` 但不生效，这是当前代码限制；需要修改 Redis 配置代码后才会读取环境变量。

## 4. JWT 鉴权失败

**现象**：FastAPI 接口返回 `401 Unauthorized`，或无法从 Django 用户服务获取用户详情。

**处理**：

- 确认 DjangoUserService 正常运行。
- 确认 `DJANGO_API_URL` 指向 Django 服务地址，例如 `http://127.0.0.1:8001`。
- 确认 FastAPI 和 Django 使用相同的 `SECRET_KEY` 和 `ALGORITHM`。
- 确认请求头为 `Authorization: Bearer <jwt-token>`。
- 如果 Redis 不可用，JWT 黑名单检查会返回 503，而不是忽略黑名单。

## 5. Chroma / 向量库问题

**现象**：上传文档无法检索、企业 RAG 无结果、目录权限错误。

**处理**：

- 普通上传文档向量库路径参考 `backend/app/config/chroma.yaml`。
- 企业 RAG parent-child 数据默认使用 `backend/data/chromadb_enterprise_parent_child`。
- 长期记忆向量使用 `long_term_memories` collection。
- 检查数据目录是否存在且当前进程有读写权限。
- 企业 RAG 检索为空时，确认已运行企业 chunk 入库脚本。

## 6. Reranker 模型加载失败

**现象**：`RuntimeError: 重排序模型加载失败` 或模型目录不存在。

**处理**：

- 检查 `RERANKER_MODEL_PATH`。
- 确认 Qwen3-Reranker-0.6B 模型文件完整。
- 参考 [Hugging Face 模型配置](./huggingface_model.md)。
- 如果 GPU 内存不足，可以先切换 CPU 或降低 batch size。

## 7. 前端访问后端失败

**现象**：浏览器 `Network error`、CORS 错误、登录成功但聊天失败。

**处理**：

- 确认 Vite dev server 代理配置正确：`/api` 到 FastAPI，`/user` / `/file` 到 Django。
- 确认 FastAPI 运行在 `8000`，Django 运行在 `8001`。
- 浏览器控制台查看实际请求路径。
- 检查 JWT 是否已写入前端状态，并带到 FastAPI 请求头。

## 8. 文件上传失败

**现象**：`File too large`、`Unsupported file type`、向量化失败。

**处理**：

- 普通向量上传仅支持当前后端允许的文件类型。
- 检查上传文件大小限制。
- 检查临时目录、媒体目录和 Chroma 目录权限。
- 对 PDF 解析失败的问题，先用纯 TXT 文件做 smoke test。

## 9. 长期记忆没有生效

**现象**：跨 session 提问没有利用历史偏好或项目上下文。

**处理**：

- 检查 `long_term_memories` 表是否有当前用户 active 记忆。
- 检查 Chroma `long_term_memories` collection 是否写入。
- 检查 RouterGraph 是否把召回记忆注入 Agent 或 Enterprise RAG。
- 注意当前公开 API 只有 `GET /api/memories` 和 `DELETE /api/memories/{memory_id}`；独立 `/api/memories/search` 尚未作为路由暴露。

## 10. 端口被占用

Windows：

```powershell
netstat -ano | findstr :8000
```

Linux/macOS：

```bash
lsof -i :8000
```

找到进程后终止或换端口启动服务。

## 11. 环境检查命令

PowerShell：

```powershell
conda activate nexuskb
$env:PYTHONUTF8='1'
D:\Anaconda\envs\nexuskb\python.exe -m pytest backend\tests
```

FastAPI 健康检查：

```text
http://127.0.0.1:8000/health/live
http://127.0.0.1:8000/health/ready
```

交互式 API 文档：

```text
http://127.0.0.1:8000/docs
```

## 12. 提交问题时请提供

1. 完整错误日志。
2. 操作系统、Python/Node 版本和 conda 环境名。
3. FastAPI、Django、前端三个服务的启动命令。
4. `.env` 中的变量名列表，不要贴真实密钥。
5. 复现步骤和请求路径。
