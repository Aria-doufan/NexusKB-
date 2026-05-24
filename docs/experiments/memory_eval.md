# 长期记忆评估流程

## 目的

本文档定义记忆系统 v2 之后的固定评估流程。后续只要修改长期记忆抽取、去重、检索、上下文注入、删除接口或用户隔离逻辑，都需要按这里的流程验证。

评估目标不是证明“数据库里有记录”，而是证明：

- 该记住的能写入。
- 需要时能按 query 召回。
- 跨 session 回答会使用相关记忆。
- 删除后不再召回。
- 不同用户之间不会互相看到记忆。
- 指标能长期横向比较。

## 测试资产

Golden cases：

```text
backend/scripts/memory_eval_golden_cases.jsonl
```

每条 case 包含：

```json
{
  "case_id": "project_memory_strategy",
  "category": "project",
  "setup_turns": ["记住：..."],
  "search_query": "记忆模块第一阶段策略",
  "cross_session_query": "我们这个项目的记忆模块第一阶段采用什么策略？",
  "expected_memory_terms": ["ADD-only", "不自动"],
  "expected_answer_terms": ["ADD-only", "不自动"],
  "delete_after": false
}
```

评估脚本：

```text
backend/scripts/evaluate_long_term_memory.py
```

输出目录：

```text
backend/data/memory_eval/eval/
```

每次运行会输出：

- `<output_name>_summary.json`
- `<output_name>_details.jsonl`
- `<output_name>_details.csv`

## 运行前准备

启动后端服务：

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

确保环境满足：

- MySQL 可用，且后端启动时已创建 `long_term_memories`。
- Redis 可用，避免鉴权 ready 状态异常。
- `DEEPSEEK_API_KEY` 可用，用于 Router、回答和记忆抽取。
- Ollama embedding 服务可用，用于 Chroma 记忆检索。

## 推荐命令

如果已经有可用 JWT：

```powershell
backend\.venv\Scripts\python.exe backend\scripts\evaluate_long_term_memory.py `
  --base-url http://127.0.0.1:8000 `
  --token "<user_jwt>" `
  --other-token "<other_user_jwt>" `
  --settle-seconds 3 `
  --output-name memory_v2_smoke
```

如果使用 `backend/.env` 中的 `SECRET_KEY` 和 `ALGORITHM` 自动生成测试 JWT：

```powershell
backend\.venv\Scripts\python.exe backend\scripts\evaluate_long_term_memory.py `
  --base-url http://127.0.0.1:8000 `
  --user-id memory-eval-user `
  --other-user-id memory-eval-other-user `
  --settle-seconds 3 `
  --output-name memory_v2_smoke
```

快速 smoke test：

```powershell
backend\.venv\Scripts\python.exe backend\scripts\evaluate_long_term_memory.py --limit 1 --output-name memory_smoke_one
```

## 指标解释

| 指标 | 含义 |
| --- | --- |
| `pass_rate` | case 总通过率 |
| `memory_search_hit_rate` | 长期记忆服务的语义召回是否返回包含期望事实的记忆；当前公开 API 尚未暴露 `/api/memories/search` |
| `answer_hit_rate` | 新 session 提问后，回答是否包含期望要点 |
| `delete_pass_rate` | 删除型 case 删除后是否不再召回 |
| `isolation_pass_rate` | 第二个用户是否查不到第一个用户的记忆 |
| `average_search_latency_ms` | 长期记忆搜索平均耗时 |
| `average_answer_latency_ms` | 跨 session 回答平均耗时 |

## v2 第一阶段建议门槛

小样本 smoke 阶段：

```text
memory_search_hit_rate >= 0.75
answer_hit_rate >= 0.75
delete_pass_rate = 1.0
isolation_pass_rate = 1.0
```

扩展到 20 条以上 golden cases 后：

```text
memory_search_hit_rate >= 0.80
answer_hit_rate >= 0.80
delete_pass_rate = 1.0
isolation_pass_rate = 1.0
```

性能建议：

```text
average_search_latency_ms <= 500ms
```

如果本地 embedding 或 Chroma 首次加载较慢，应记录冷启动和热启动两组结果。

## 固定验收场景

当前 golden cases 覆盖：

- 用户偏好：回答风格。
- 项目背景：记忆模块第一阶段策略。
- 版本记录：v1/v2/v3 机制档案。
- 删除控制：删除后不再召回。
- 用户隔离：可选第二用户 token 验证。

后续新增能力时必须扩展 case：

- 时间归一化：昨天、下周、最近。
- 实体链接：项目名、文件名、接口名。
- 记忆修正：用户偏好从 A 改成 B。
- 敏感治理：不记录未经确认的敏感推断。

## 失败分析

优先看 `details.jsonl`：

- `search_pass=false`：抽取失败、向量写入失败、检索阈值过高、搜索 query 不合理。
- `answer_pass=false` 且 `search_pass=true`：上下文注入不生效，或模型没有遵循长期记忆。
- `delete_pass=false`：软删除状态未过滤，或 Chroma 删除失败后仍被召回。
- `isolation_pass=false`：`user_id` filter 失效，属于高优先级问题。

## 后续流程约定

每次修改记忆模块后：

1. 跑 `python -m compileall backend/app backend/scripts/evaluate_long_term_memory.py`。
2. 启动后端。
3. 跑 `evaluate_long_term_memory.py`。
4. 把 summary 关键指标记录到本文件或新建 `docs/experiments/memory_eval_YYYYMMDD.md`。
5. 如果指标下降，先定位失败 case，再决定是否调整抽取 prompt、检索阈值、注入策略或 golden case。
