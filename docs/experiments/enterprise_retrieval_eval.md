# 企业 RAG 检索评测实验记录

## 记录目的

本文档用于集中记录 EnterpriseRAG-Bench 检索实验数据，方便对比不同召回与重排方案的效果。

当前优先比较三组方案：

1. Chroma only
2. Chroma + BM25 混合召回
3. Chroma + BM25 混合召回 + Reranker

评测目标不是最终验收，而是先建立可复用的检索能力仪表盘，用数据指导后续 `rag_intent` 策略、`source_hints` 软加权和 reranker 接入方式。

## 固定评测配置

| 项目 | 值 |
| --- | --- |
| 评测日期 | 2026-05-14 |
| 问题集 | `backend/data/enterprise_rag_bench/questions.jsonl` |
| 问题数量 | 500 |
| Chroma collection | `enterprise_rag_bench_parent_child` |
| Chroma persist dir | `backend/data/chromadb_enterprise_parent_child` |
| Embedding model | `qwen3-embedding:latest` |
| Ollama base URL | `http://localhost:11434` |
| 评测 K 值 | `1,5,10,20` |
| baseline search_k | 50 child chunks |
| source filter | 不启用硬过滤 |

## 指标定义

| 指标 | 含义 |
| --- | --- |
| `hit@K` | Top K parent documents 中是否至少命中一个 expected doc |
| `recall@K` | Top K parent documents 命中的 expected docs 占比 |
| `mrr@K` | Top K 内首个正确文档的 reciprocal rank 均值 |
| `average_latency_ms` | 单问题平均检索耗时 |
| `elapsed_sec` | 整轮评测总耗时 |

## 实验总表

| 实验 ID | 方案 | 状态 | 问题数 | hit@1 | hit@5 | hit@10 | hit@20 | recall@1 | recall@5 | recall@10 | recall@20 | mrr@20 | 平均延迟 ms | 总耗时 s | 输出文件 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp-001 | Chroma only child vector search | 已完成 | 500 | 0.614 | 0.760 | 0.794 | 0.814 | 0.5165 | 0.7155 | 0.7627 | 0.7858 | 0.6744 | 129.23 | 64.67 | `backend/data/enterprise_rag_bench/eval/baseline_chroma_child_summary.json` |
| exp-002 | Chroma + BM25 hybrid retrieval | 已完成 | 500 | 0.640 | 0.814 | 0.886 | 0.900 | 0.5364 | 0.7775 | 0.8658 | 0.8868 | 0.7190 | 370.40 | 185.26 | `backend/data/enterprise_rag_bench/eval/hybrid_bm25_rrf_summary.json` |
| exp-003 | Chroma + BM25 hybrid retrieval + reranker | 已完成 | 500 | 0.734 | 0.840 | 0.882 | 0.900 | 0.6227 | 0.8029 | 0.8645 | 0.8868 | 0.7846 | 1149.55 | 574.89 | `backend/data/enterprise_rag_bench/eval/hybrid_bm25_rrf_reranker_summary.json` |

## 实验明细

### exp-001: Chroma only child vector search

状态：已完成

运行命令：

```powershell
python backend/scripts/evaluate_enterprise_retrieval.py
```

核心配置：

| 参数 | 值 |
| --- | --- |
| method | `baseline_chroma_child` |
| search_k_child_chunks | 50 |
| where_source_type | `null` |
| parent 去重 | 启用，按 child 检索顺序保留首个 parent_doc_id |

结果摘要：

| 指标 | 值 |
| --- | ---: |
| questions | 500 |
| hit@1 | 0.614 |
| hit@5 | 0.760 |
| hit@10 | 0.794 |
| hit@20 | 0.814 |
| recall@1 | 0.5165 |
| recall@5 | 0.7155 |
| recall@10 | 0.7627 |
| recall@20 | 0.7858 |
| mrr@20 | 0.6744 |
| average_latency_ms | 129.23 |
| elapsed_sec | 64.67 |

输出文件：

- `backend/data/enterprise_rag_bench/eval/baseline_chroma_child_summary.json`
- `backend/data/enterprise_rag_bench/eval/baseline_chroma_child_details.jsonl`
- `backend/data/enterprise_rag_bench/eval/baseline_chroma_child_details.csv`

初步结论：

- Chroma-only baseline 已经具备可用召回能力，`hit@20=0.814`。
- `hit@1=0.614` 说明首位排序仍有明显优化空间，reranker 可能主要提升 Top1/MRR。
- `hit@20` 与 `recall@20` 的提升空间仍存在，BM25 混合召回应优先验证是否能补足向量召回漏召。
- 当前不启用 `source_hints` 硬过滤是合理的，因为之前联调发现来源预测错误会过滤掉正确答案。

待补充分析：

- Top20 未命中的问题类型分布。
- 各 source_type 的命中率差异。
- 多 expected docs 问题的 recall 损失情况。
- 高相似度但错误 parent 的典型案例。

### exp-002: Chroma + BM25 hybrid retrieval

状态：已完成

计划目标：

- 验证 BM25 是否能补足向量检索漏召。
- 对比 `hit@20`、`recall@20` 是否相对 exp-001 提升。
- 观察平均延迟是否仍能接受。

运行命令：

```powershell
backend\.venv\Scripts\python.exe backend\scripts\evaluate_enterprise_hybrid_retrieval.py --method hybrid_bm25_rrf
```

核心配置：

| 参数 | 值 |
| --- | --- |
| method | `hybrid_bm25_rrf` |
| chroma_search_k | 50 child chunks |
| bm25_search_k | 50 child chunks |
| fusion_strategy | RRF |
| rrf_k | 60 |
| parent 去重 | 启用，按融合后 child 顺序保留首个 parent_doc_id |
| source filter | 不启用硬过滤 |

结果摘要：

| 指标 | 值 |
| --- | ---: |
| questions | 500 |
| hit@1 | 0.640 |
| hit@5 | 0.814 |
| hit@10 | 0.886 |
| hit@20 | 0.900 |
| recall@1 | 0.5364 |
| recall@5 | 0.7775 |
| recall@10 | 0.8658 |
| recall@20 | 0.8868 |
| mrr@20 | 0.7190 |
| average_latency_ms | 370.40 |
| elapsed_sec | 185.26 |

输出文件：

- `backend/data/enterprise_rag_bench/eval/hybrid_bm25_rrf_summary.json`
- `backend/data/enterprise_rag_bench/eval/hybrid_bm25_rrf_details.jsonl`
- `backend/data/enterprise_rag_bench/eval/hybrid_bm25_rrf_details.csv`

结论：

- 相比 exp-001，`hit@20` 从 0.814 提升到 0.900，提升 0.086。
- 相比 exp-001，`recall@20` 从 0.7858 提升到 0.8868，提升 0.1010。
- 相比 exp-001，`mrr@20` 从 0.6744 提升到 0.7190，说明混合召回不仅补充了漏召，也轻微改善了排序。
- `hit@1` 从 0.614 提升到 0.640，Top1 提升有限，后续 reranker 仍有验证价值。
- 平均延迟从 129.23ms 增加到 370.40ms，约增加 241.17ms。当前脚本每次运行会重新构建内存 BM25 索引；服务化后可常驻索引，线上单次查询延迟预计会低于脚本端到端评测值。

### exp-003: Chroma + BM25 hybrid retrieval + reranker

状态：已完成

计划目标：

- 验证 reranker 是否提升 Top1 排序质量和 `mrr@20`。
- 对比 reranker 带来的延迟成本。
- 判断是否值得在 Router RAG 链路中默认启用，或仅针对复杂 `rag_intent` 启用。

模型状态：

- 已通过 ModelScope 下载完整模型到 `D:\Hugging_Face\models\Qwen3-Reranker-0.6B`。
- 已确认存在 `model.safetensors`，大小约 1.11GB。
- 已用 Qwen3 官方 Transformers yes/no logit 方式完成本地推理验证。
- 不再使用 `sentence_transformers.CrossEncoder` 直接加载该模型，因为它会退化成 `Qwen3ForSequenceClassification` 并提示 `score.weight` 缺失，分数不可靠。
- 已将当前 Python 环境从 CPU 版 PyTorch 替换为 CUDA 版：`torch 2.11.0+cu130`、`torchvision 0.26.0+cu130`、`torchaudio 2.11.0+cu130`。
- 已验证 `torch.cuda.is_available() == True`，GPU 为 NVIDIA GeForce RTX 4070 SUPER。

全量运行命令：

```powershell
backend\.venv\Scripts\python.exe backend\scripts\evaluate_enterprise_hybrid_retrieval.py --method hybrid_bm25_rrf_reranker --reranker-device cuda --reranker-max-length 512 --reranker-candidate-k 20 --reranker-batch-size 4
```

核心配置：

| 参数 | 值 |
| --- | --- |
| hybrid_candidate_k | 20 |
| reranker_model | `D:\Hugging_Face\models\Qwen3-Reranker-0.6B` |
| rerank_top_n | 20 |
| fusion_strategy | RRF |
| reranker_scoring | Qwen3 官方 yes/no causal-LM logit |
| reranker_device | `cuda` |
| reranker_max_length | 512 |
| reranker_batch_size | 4 |
| source filter | 不启用硬过滤 |

结果摘要：

| 指标 | 值 |
| --- | ---: |
| questions | 500 |
| hit@1 | 0.734 |
| hit@5 | 0.840 |
| hit@10 | 0.882 |
| hit@20 | 0.900 |
| recall@1 | 0.6227 |
| recall@5 | 0.8029 |
| recall@10 | 0.8645 |
| recall@20 | 0.8868 |
| mrr@20 | 0.7846 |
| average_latency_ms | 1149.55 |
| elapsed_sec | 574.89 |

输出文件：

- `backend/data/enterprise_rag_bench/eval/hybrid_bm25_rrf_reranker_summary.json`
- `backend/data/enterprise_rag_bench/eval/hybrid_bm25_rrf_reranker_details.jsonl`
- `backend/data/enterprise_rag_bench/eval/hybrid_bm25_rrf_reranker_details.csv`

结论：

- reranker 显著提升排序质量：相比 exp-002，`hit@1` 从 0.640 提升到 0.734，`mrr@20` 从 0.7190 提升到 0.7846。
- `hit@20` 和 `recall@20` 与 exp-002 持平是合理结果，因为 reranker 只重排混合召回候选，不新增候选。
- `hit@10` 从 0.886 降到 0.882，说明 reranker 对少量 case 有重排损伤，后续需要看失败样例决定是否调低 rerank 候选窗口、改 prompt/instruction，或只对特定 `rag_intent` 启用。
- 平均延迟从 exp-002 的 370.40ms 增至 1149.55ms；如果服务化时 BM25 索引常驻内存，实际线上延迟会低于脚本端到端统计，但 reranker 仍应作为策略开关而不是无条件默认。
- 当前脚本 `backend/scripts/evaluate_enterprise_hybrid_retrieval.py` 已改为 Qwen3 官方 yes/no logit 评分方式，避免 `CrossEncoder` 分类头缺失造成错误结果。

## 横向对比结论

当前已完成 exp-001、exp-002、exp-003。

阶段性结论：

- BM25 + RRF 混合召回值得进入下一阶段，因为它对 `hit@20`、`recall@20`、`mrr@20` 都有明确提升。
- Reranker 值得保留为策略能力，因为它把 `hit@1` 从 exp-002 的 0.640 提升到 0.734，把 `mrr@20` 从 0.7190 提升到 0.7846。
- Reranker 不提升 `hit@20` 和 `recall@20`，所以它不能替代混合召回，只适合做排序增强。
- 当前工程取舍是延迟：GPU 上 rerank 20 个候选后平均检索耗时约 1.15 秒，应按 `rag_intent` 或复杂查询开关启用。

后续对比时优先看：

1. Reranker 造成 `hit@10` 轻微下降的失败样例。
2. 是否将 `reranker_candidate_k` 从 20 调到 10 或按 `rag_intent` 动态控制。
3. 失败案例是否集中在特定 `source_type` 或 `question_type`。
4. `source_hints` 软加权是否能进一步提升混合召回的候选质量。

## 失败样例记录模板

| 实验 ID | question_id | question_type | source_types | expected_doc_ids | top_parent_doc_ids | 失败现象 | 初步原因 | 后续动作 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | - | - | - | - | - | - | - | - |

## 后续执行顺序

1. 对 exp-002 和 exp-003 的失败样例做差异分析。
2. 设计 `rag_intent` 策略矩阵：简单问题默认 BM25+Chroma，复杂/精确问题再启用 reranker。
3. 验证 `reranker_candidate_k=10` 和 `reranker_candidate_k=20` 的效果/延迟差异。
4. 评估 `source_hints` 软加权或 source-aware retrieval。
