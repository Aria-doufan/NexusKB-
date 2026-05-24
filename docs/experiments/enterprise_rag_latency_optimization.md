# 企业 RAG 延迟优化实施记录

## 记录目的

本文档详细记录企业 RAG 延迟优化相关的每一步尝试、代码改动、验证命令和结果。总览进度同步记录到 `docs/WORKBOARD.md`，这里保留可回溯的实施细节。

## 2026-05-15：阶段性 SSE、并行检索、BM25 预热

### 背景

全量离线评测已经证明 `strategy_matrix` 相比 `chroma_only` 有明确准确性收益，但企业 RAG 平均延迟明显高于普通聊天。当前优先做不牺牲检索精度的优化：

1. 阶段性 SSE 事件，降低用户等待感。
2. Chroma/BM25 并行检索，降低检索阶段真实耗时。
3. BM25 启动预热，避免首个企业 RAG 请求承担索引构建成本。

### 尝试 1：企业 RAG 阶段性 SSE 事件

改动文件：

- `backend/app/agent/router_graph.py`
- `backend/app/rag/enterprise_rag_service.py`

实现方式：

- `EnterpriseRagService.get_documents_and_summary()` 新增 `stage_callback` 参数。
- `retrieve()` 在检索开始、检索完成、reranker 开始、reranker 完成时回调阶段事件。
- 摘要生成前后回调 `summarizing` 阶段事件。
- `RouterGraph.stream()` 的 `enterprise_knowledge` 分支改为边执行边从队列读取阶段事件，并转成 SSE 输出。

当前 SSE 事件示例：

```json
{"type":"stage","stage":"retrieving","status":"start","route":"enterprise_knowledge"}
{"type":"stage","stage":"retrieving","status":"done","route":"enterprise_knowledge"}
{"type":"stage","stage":"reranking","status":"start","route":"enterprise_knowledge"}
{"type":"stage","stage":"reranking","status":"done","route":"enterprise_knowledge"}
{"type":"stage","stage":"summarizing","status":"start","route":"enterprise_knowledge"}
{"type":"stage","stage":"summarizing","status":"done","route":"enterprise_knowledge"}
```

预期收益：

- 用户可以在 RAG 最终回答生成前看到系统已进入检索、重排、总结阶段。
- 不改变检索策略和回答内容，不影响准确性。

### 尝试 2：Chroma/BM25 并行检索

改动文件：

- `backend/app/rag/enterprise_rag_service.py`

实现方式：

- 抽出 `_chroma_search()` 和 `_bm25_search_with_perf()`，分别保留 `PERF_METRIC`。
- `retrieve()` 中使用 `asyncio.create_task()` 和 `asyncio.gather()` 并发等待两路召回结果。
- 保持后续 RRF、source hint soft boost、reranker 策略不变。

预期收益：

- 检索阶段总耗时从“Chroma 耗时 + BM25 耗时”接近变为“两者最大值 + RRF 耗时”。
- 准确性不变，因为候选集合和融合逻辑没有改变。

### 尝试 3：BM25 启动预热

改动文件：

- `backend/main.py`
- `backend/app/rag/enterprise_rag_service.py`

实现方式：

- `EnterpriseRagService` 新增 `prewarm_bm25_index()`。
- FastAPI startup 中创建后台任务调用预热。
- 预热失败只记录 warning，首个请求仍可回退到按需构建。

预期收益：

- 首个企业 RAG 请求不再承担 BM25 内存索引构建成本。
- 服务启动过程不被预热强制阻塞。

### 尝试 4：评测输出防覆盖

改动文件：

- `backend/scripts/evaluate_enterprise_hybrid_retrieval.py`

实现方式：

- 新增 `--output-name` 参数，用于指定输出文件前缀。
- 后续 `reranker_candidate_k=10` 与 `20` 可分别输出到 `strategy_matrix_k10_*` 和 `strategy_matrix_k20_*`，避免覆盖历史 `strategy_matrix_*`。

### 尝试 5：线上 reranker 候选窗口收敛到 10

改动文件：

- `backend/app/rag/enterprise_rag_service.py`

实现方式：

- 新增 `RERANKER_CANDIDATE_K = 10`。
- reranker 只重排 RRF 后的前 10 个 parent candidates，剩余候选保持 RRF 顺序接在后面。
- SSE `reranking` 阶段和 `enterprise_rag.reranker` 指标中的 `candidates` 改为实际送入 reranker 的候选数。
- `strategy` 元数据增加 `reranker_candidate_k`，便于前端或日志侧确认当前策略。

依据：

- 2026-05-15 `strategy_matrix_k10` 离线评测显示，`candidate_k=10` 相比既有 `candidate_k=20` 平均延迟减少 231.51ms，且 `hit@1`、`hit@5`、`f1@5`、`mrr@20` 均未下降。

### 已完成验证

语法验证：

```powershell
backend\.venv\Scripts\python.exe -m py_compile backend\app\rag\enterprise_rag_service.py backend\app\agent\router_graph.py backend\main.py backend\scripts\evaluate_enterprise_hybrid_retrieval.py
```

结果：通过。

BM25 预热验证：

```powershell
@'
import asyncio
from app.rag.enterprise_rag_service import enterprise_rag_service

async def main():
    await enterprise_rag_service.prewarm_bm25_index()
    print('prewarm_ok')

asyncio.run(main())
'@ | .\.venv\Scripts\python.exe -
```

结果：

- `prewarm_ok`
- `enterprise_rag.bm25_index_build elapsed_ms=3637.90 documents=28434`

检索阶段事件验证：

```powershell
@'
import asyncio
from app.rag.enterprise_rag_service import enterprise_rag_service

async def main():
    async def callback(payload):
        print('stage_event', payload)
    docs = await enterprise_rag_service.retrieve(
        query='What is the contractor to FTE conversion process?',
        k=2,
        search_k=5,
        use_reranker=False,
        stage_callback=callback,
    )
    print('docs', len(docs))

asyncio.run(main())
'@ | .\.venv\Scripts\python.exe -
```

结果：

- 输出 `retrieving start` 与 `retrieving done` 阶段事件。
- 返回 `docs 2`。
- Chroma、BM25、RRF、retrieve_total 均有 `PERF_METRIC` 日志。

### 待验证

1. 用更多真实企业问题抽样对比优化后的 p50/p90/p95。
2. 观察前端阶段提示是否需要更细粒度状态文案，例如“检索资料数”“重排候选数”。

## 2026-05-15：真实 SSE 端到端验证

### 服务环境

- Redis：本地 `6379` 已启动。
- MySQL：本地 `3306` 已监听并可完成启动建表检查。
- Ollama：本地 `11434` 已监听。
- FastAPI：`http://127.0.0.1:8000`。
- 测试鉴权：使用 `backend/.env` 中 `SECRET_KEY` 和 `ALGORITHM` 生成 1 小时有效期测试 JWT。

### 启动验证

FastAPI startup 日志确认：

- `企业 RAG BM25 索引预热任务已启动`
- `enterprise_rag.bm25_index_build elapsed_ms=3649.43 documents=28434`
- `企业 RAG BM25 索引预热完成`

### SSE 样本 1：basic RAG，不启用 reranker

请求：

```text
POST /api/agent/query/stream
query = What is the contractor to FTE conversion process in the enterprise knowledge base?
```

收到事件：

```text
route: enterprise_knowledge, rag_intent=basic
stage: retrieving start
stage: retrieving done
stage: summarizing start
stage: summarizing done
response
done
```

最终 strategy：

```json
{
  "retrieval": "chroma+bm25+rrf",
  "reranker": false,
  "reranker_candidate_k": null,
  "source_hint_mode": "soft_weight",
  "rag_intent": "basic",
  "router_confidence": 0.95
}
```

关键指标：

- `router.to_route_event`: 1262.30ms
- `enterprise_rag.bm25_search`: 44.06ms
- `enterprise_rag.chroma_search`: 3441.59ms
- `enterprise_rag.retrieve_phase`: 3442.81ms
- `enterprise_rag.total`: 7194.80ms

### SSE 样本 2：constrained RAG，启用 reranker

请求：

```text
POST /api/agent/query/stream
query = Compare the contractor to FTE conversion process with related onboarding milestones and security/legal gating requirements in the enterprise knowledge base. What dependencies and risks should the manager track across the 30/60/120 day ramp?
```

收到事件：

```text
route: enterprise_knowledge, rag_intent=constrained
stage: retrieving start, search_k=60, reranker=true
stage: retrieving done, candidates=88, documents=60
stage: reranking start, candidates=10
stage: reranking done, candidates=10
stage: summarizing start, documents=8
stage: summarizing done, documents=8
response
done
```

最终 strategy：

```json
{
  "retrieval": "chroma+bm25+rrf",
  "reranker": true,
  "reranker_candidate_k": 10,
  "source_hint_mode": "soft_weight",
  "rag_intent": "constrained",
  "router_confidence": 0.95
}
```

关键指标：

- `router.to_route_event`: 1964.38ms
- `enterprise_rag.bm25_search`: 73.14ms
- `enterprise_rag.chroma_search`: 266.38ms
- `enterprise_rag.reranker`: 1981.03ms, `candidates=10`
- `enterprise_rag.retrieve_phase`: 2248.94ms
- `enterprise_rag.total`: 8882.89ms

结论：

- 真实 `/api/agent/query/stream` SSE 可以收到阶段事件。
- reranker 分支真实返回 `reranking` 阶段事件，且候选数为 10。
- 最终 `response.strategy.reranker_candidate_k=10` 已经在真实服务链路中返回。

### 前端接收处理

改动文件：

- `front/src/views/AIChat.vue`

实现方式：

- 处理 `route` 事件，保存 `message.route` 与 `message.ragIntent`。
- 处理 `stage` 事件，保存到 `message.stages`，并在助手占位消息中显示当前阶段状态。
- 处理最终 `response.strategy`，保存到 `message.ragStrategy`，同时保留 `documents`。

验证：

```powershell
npm run build
```

结果：通过。构建产物存在大 chunk 警告，但不影响本次功能验证。
