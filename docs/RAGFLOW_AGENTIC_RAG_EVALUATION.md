# RAGFlow Agentic RAG 评测方案

整理日期：2026-05-16

## 1. 文档定位

本文定义 RAGFlow Agentic RAG 的评测闭环，包括评测数据、baseline、指标公式、通过阈值、失败样例格式和策略对比报告。执行路线见 [执行路线图](./RAGFLOW_AGENTIC_RAG_ROADMAP.md)，工程 schema 和策略矩阵见 [架构设计](./RAGFLOW_AGENTIC_RAG_ARCHITECTURE.md)。

评测是本项目的核心竞争力之一。项目不能只说“混合检索和 reranker 效果更好”，而要通过固定数据集、固定 baseline 和固定指标证明策略变化是否有效。

## 2. 评测目标

评测需要回答五个问题：

1. dense 向量检索是否优于 BM25，在哪些问题类型上更好？
2. dense + BM25 + RRF 是否比单路召回更稳？
3. reranker 是否提升 Top1 / MRR，延迟成本是多少？
4. Agentic 策略矩阵是否比固定策略更好？
5. 最终回答的引用是否能支撑答案？

## 3. 评测数据

### 3.1 第一阶段数据集

| 数据 | 状态 | 说明 |
| --- | --- | --- |
| EnterpriseRAG-Bench | 已完成 / 部分接入 | 当前项目已有相关数据处理、索引和评测脚本。 |
| 用户上传文档样例集 | 待实现 | 用于演示 PDF/TXT/Markdown 等入库问答。 |
| 人工构造面试演示集 | 待实现 | 20 到 50 条高质量问题，覆盖事实、流程、对比、多跳、资料不足。 |

### 3.2 推荐样例格式

```json
{
  "query_id": "q_001",
  "query": "试用期员工请假流程是什么？",
  "question_type": "procedure",
  "gold_doc_ids": ["doc_employee_handbook"],
  "gold_chunk_ids": ["chunk_leave_001", "chunk_leave_002"],
  "gold_answer_points": [
    "需要提前提交请假申请",
    "直属主管审批",
    "超过一定天数需要 HR 审批"
  ],
  "source_type": "upload",
  "metadata_constraints": {
    "department": "HR"
  }
}
```

如果早期没有 `gold_chunk_ids`，可以先只使用 `gold_doc_ids` 计算检索指标，再逐步补 chunk 级标注。

### 3.3 子问题分解样例要求

用于评测 decompose 的样例必须显式标注复杂问题类型和证据覆盖要求：

```json
{
  "query_id": "q_multi_001",
  "query": "试用期员工请假和正式员工请假在审批流程上有什么区别？",
  "question_type": "comparison",
  "expected_sub_queries": [
    "试用期员工请假审批流程是什么？",
    "正式员工请假审批流程是什么？",
    "试用期员工和正式员工请假审批流程的差异是什么？"
  ],
  "gold_doc_ids": ["doc_employee_handbook"],
  "gold_chunk_ids": ["chunk_probation_leave", "chunk_regular_leave"],
  "gold_answer_points": [
    "试用期员工请假审批要求",
    "正式员工请假审批要求",
    "两者审批节点或规则差异"
  ],
  "required_evidence_groups": [
    ["chunk_probation_leave"],
    ["chunk_regular_leave"]
  ]
}
```

`required_evidence_groups` 用于衡量多证据覆盖：每组至少命中一个 chunk 才算证据完整。comparison / multi-hop 不能只看 Hit@K；如果只命中一个子问题的证据，答案很容易片面。

## 4. Baseline 策略

至少保留以下 baseline：

| 策略名 | 说明 | 状态 |
| --- | --- | --- |
| `chroma_only` | 只使用 dense vector retrieval | 已完成 |
| `bm25_only` | 只使用 BM25 sparse retrieval | 已完成 / 部分完成 |
| `dense_bm25_rrf` | dense + BM25 + RRF 融合 | 已完成 |
| `dense_bm25_rrf_reranker` | 融合后对候选进行 reranker 精排 | 已完成 |
| `strategy_matrix` | 根据 rag_intent 选择策略 | 部分完成 / 待工程化 |
| `strategy_matrix_hyde` | 对 semantic query 启用 HyDE | 待实现 |
| `strategy_matrix_decompose` | 对 multi-hop / comparison 启用子查询拆解 | 待实现 |

`strategy_matrix_decompose` 必须只在 `multi_hop` 和 `comparison` 子集上单独报告增益，同时也要报告全量数据集指标，避免复杂问题收益掩盖普通问题延迟上升。

每次新增策略必须和 baseline 对比，不单独汇报孤立指标。

## Offline Baseline Evaluation Commands

The offline baseline is split into retrieval and generation tracks.

Retrieval baseline:

```powershell
conda run -n nexuskb python backend/scripts/evaluate_enterprise_hybrid_retrieval.py --method strategy_matrix_decompose --standard-output
```

Generation baseline with RAGAS and GPT judge:

```powershell
$env:OPENAI_API_KEY = "<your key>"
conda run -n nexuskb python backend/scripts/evaluate_enterprise_rag_generation.py --limit 50 --judge-provider openai --judge-model gpt-4o --embedding-model text-embedding-3-small
```

Generated run artifacts are written under `backend/data/eval_outputs/`.

Report delta comparison defaults:

- Retrieval baseline: `backend/data/eval_baselines/current/retrieval_summary.json`.
- Generation baseline: `backend/data/eval_baselines/generation/current/generation_ragas_summary.json`.
- `--baseline-dir <path>` can override either default.

## Enterprise Chroma-Only Semantic Chunking Evaluation

Use `backend/scripts/evaluate_enterprise_chunking_profiles.py` to compare parent-child chunking profiles for the enterprise corpus without changing online Agentic RAG behavior. This chunking track is Chroma-only: Elasticsearch is intentionally excluded until a Chroma winner exists, so chunk-size and boundary effects are isolated before adding a second retrieval backend.

All profiles in a stage must share the same sample fingerprint: `sample_size`, `seed`, document count/hash, and question count/hash. The runner writes the stage fingerprint to `sample_fingerprint.json` and rejects comparisons where profiles were prepared from different document/question samples. In this script, `--sample-size` controls document sampling; use `--limit` to cap evaluated questions.

Primary metrics are `recall@10`, `evidence_coverage@10`, and `hit@5`. Secondary checks are `mrr@20`, `ndcg@10`, latency, total child chunks, and semantic-type breakdowns when available.

Current comparable run command:

```powershell
conda run --no-capture-output -n NexusKB python backend/scripts/evaluate_enterprise_chunking_profiles.py --stage stage1 --method chroma_bm25_rrf --sample-size 100 --seed 42
```

Smoke command:

```powershell
conda run --no-capture-output -n NexusKB python backend/scripts/evaluate_enterprise_chunking_profiles.py --stage stage1 --method chroma_bm25_rrf --sample-size 25 --seed 42 --limit 5
```

Stage 1 compares fixed-size recursive child chunking profiles:

| Profile | Boundary mode | Parent size / overlap | Child size / overlap | recall@10 | hit@5 | mrr@20 | ndcg@10 | avg latency ms | child chunks |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_baseline` | `recursive` | `3000 / 300` | `700 / 100` | 0.9158 | 0.918 | 0.8551 | 0.8598 | 226.16 | 12011 |
| `fixed_smaller_child` | `recursive` | `3000 / 300` | `500 / 80` | 0.9128 | 0.908 | 0.8341 | 0.8434 | 260.92 | 16473 |
| `fixed_larger_child` | `recursive` | `3000 / 300` | `900 / 120` | 0.9192 | 0.908 | 0.8555 | 0.8613 | 227.04 | 9527 |
| `fixed_larger_parent` | `recursive` | `4000 / 400` | `700 / 100` | 0.9114 | 0.904 | 0.8494 | 0.8542 | 230.53 | 11588 |

Stage 2 compares semantic-boundary child chunking profiles with the same sample fingerprint:

```powershell
conda run --no-capture-output -n NexusKB python backend/scripts/evaluate_enterprise_chunking_profiles.py --stage stage2_semantic --method chroma_bm25_rrf --sample-size 100 --seed 42
```

| Profile | Boundary mode | Parent size / overlap | Child size / overlap | recall@10 | hit@5 | mrr@20 | ndcg@10 | avg latency ms | child chunks |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `semantic_baseline_threshold` | `semantic` | `3000 / 300` | `700 / 100` | 0.9179 | 0.914 | 0.8593 | 0.8638 | 252.91 | 10842 |
| `semantic_smaller_child` | `semantic` | `3000 / 300` | `500 / 80` | 0.9140 | 0.908 | 0.8384 | 0.8465 | 259.11 | 14884 |
| `semantic_larger_child` | `semantic` | `3000 / 300` | `900 / 120` | 0.9135 | 0.904 | 0.8564 | 0.8615 | 249.11 | 8702 |

The Stage 1 and Stage 2 comparable runs used `sample_size=100`, `seed=42`, `documents_count=722`, and `questions_count=500`.

Current selection: `semantic_baseline_threshold` with Elasticsearch-backed `chroma_bm25_rrf` is the selected retrieval configuration. The semantic chunking profile slightly trails `fixed_larger_child` on Chroma-only `recall@10` and latency, but improves ranking quality (`mrr@20` and `ndcg@10`) and better represents the semantic-boundary chunking direction of the project. Elasticsearch is the default enterprise retrieval backend going forward; Chroma remains the lower-latency local baseline.

Reranker check on `fixed_larger_child`:

| Method | recall@10 | hit@5 | mrr@20 | ndcg@10 | avg latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chroma_bm25_rrf` | 0.9192 | 0.908 | 0.8555 | 0.8613 | 227.04 |
| `chroma_bm25_rrf_reranker` | 0.9075 | 0.906 | 0.8642 | 0.8636 | 836.21 |

The reranker improves ranking metrics slightly but reduces `recall@10` and increases latency by roughly 3.7x, so it remains an optional ranking-sensitive experiment rather than the default retrieval path.

Outputs are written under `backend/data/chunking_eval_outputs/<stage>/`, including one `run_record.json` per profile, `comparison_summary.json`, `sample_fingerprint.json`, and a stage-level `report.md` titled `Enterprise Chroma Chunking Evaluation Report`. The record format includes `source_type` so future uploaded-document RAG evaluation can reuse the same schema after its query set and expected evidence labels exist.

## Elasticsearch Enterprise Retrieval Backend

Elasticsearch is the selected default enterprise retrieval backend for the chosen `semantic_baseline_threshold` chunking profile. Chroma remains the local lower-latency baseline, but ES is the default backend for enterprise-oriented evaluation and documentation because it preserves recall parity while improving top-rank retrieval quality.

Start local Elasticsearch:

```powershell
docker compose -f docker-compose.elasticsearch.yml up -d
```

Index the selected semantic chunks into Elasticsearch with `--reset`:

```powershell
conda run --no-capture-output -n NexusKB python backend/scripts/index_enterprise_chunks_elasticsearch.py --child-chunks-path backend/data/chunking_eval_outputs/stage2_semantic/semantic_baseline_threshold_p3000-o300_c700-o100/prepared/child_chunks_parent_child.jsonl --parent-chunks-path backend/data/chunking_eval_outputs/stage2_semantic/semantic_baseline_threshold_p3000-o300_c700-o100/prepared/parent_chunks_parent_child.jsonl --index-name nexuskb_enterprise_chunks --reset
```

The full semantic ES indexing run wrote 10,842 chunks and stores both `source_type` and `doc_semantic_type` as keyword metadata alongside dense vectors.

Run Elasticsearch evaluation on the same selected semantic sample:

```powershell
conda run --no-capture-output -n NexusKB python backend/scripts/evaluate_enterprise_hybrid_retrieval.py --backend elasticsearch --method chroma_bm25_rrf --questions-path backend/data/chunking_eval_outputs/stage2_semantic/semantic_baseline_threshold_p3000-o300_c700-o100/prepared/questions.jsonl --child-chunks-path backend/data/chunking_eval_outputs/stage2_semantic/semantic_baseline_threshold_p3000-o300_c700-o100/prepared/child_chunks_parent_child.jsonl --parent-chunks-path backend/data/chunking_eval_outputs/stage2_semantic/semantic_baseline_threshold_p3000-o300_c700-o100/prepared/parent_chunks_parent_child.jsonl --output-dir backend/data/chunking_eval_outputs/stage2_semantic/semantic_baseline_threshold_p3000-o300_c700-o100/eval_es --k-values 1,5,10,20
```

Semantic chunking backend comparison:

| Backend | questions | recall@10 | hit@5 | mrr@20 | ndcg@10 | avg latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chroma | 500 | 0.9179 | 0.914 | 0.8593 | 0.8638 | 252.91 |
| Elasticsearch | 500 | 0.9176 | 0.926 | 0.8753 | 0.8764 | 358.58 |

Elasticsearch keeps `recall@10` effectively at parity with Chroma while improving `hit@5`, `mrr@20`, and `ndcg@10`. The trade-off is higher latency, so Chroma remains useful for local low-latency baselines while Elasticsearch becomes the default enterprise retrieval backend. Keep reranker comparisons separate until Elasticsearch evaluation supports `--backend elasticsearch` with reranking.

## 5. 检索指标公式

### 5.1 Hit@K

定义：如果 TopK 检索结果中至少有一个结果命中 `gold_doc_ids` 或 `gold_chunk_ids`，则该 query 的 Hit@K = 1，否则为 0。

```text
Hit@K = 命中 query 数 / 总 query 数
```

建议同时统计：

```text
hit_doc@1, hit_doc@5, hit_doc@10
hit_chunk@1, hit_chunk@5, hit_chunk@10
```

早期如果没有 chunk 标注，先使用 doc 级指标。

### 5.2 Recall@K

定义：TopK 检索结果覆盖的 gold 数量占该 query 所有 gold 数量的比例。

```text
Recall@K(query) = |TopK_gold_hits| / |Gold_set|
Recall@K = 所有 query 的 Recall@K 平均值
```

多文档问题必须看 Recall@K，因为只命中一个文档不代表证据完整。

### 5.3 MRR@K

定义：第一个命中 gold 的结果排名为 rank，则该 query 的 reciprocal rank 为 `1 / rank`。如果 TopK 内没有命中，则为 0。

```text
MRR@K = mean(1 / first_hit_rank)
```

MRR 更适合衡量 reranker 是否把正确结果排到更前面。

### 5.4 nDCG@K（可选增强）

如果后续有分级相关性标注，可以加入 nDCG@K。第一阶段不是必须。

### 5.5 Evidence Coverage@K

子问题分解需要额外统计证据覆盖率。定义：如果一个 query 标注了 `required_evidence_groups`，TopK 结果中每覆盖一个 group 记 1 分。

```text
EvidenceCoverage@K(query) = covered_required_groups / total_required_groups
EvidenceCoverage@K = 所有 decompose 样例的平均值
```

通过阈值建议：

| 阶段 | 阈值 |
| --- | --- |
| 最小合格 | `strategy_matrix_decompose.evidence_coverage@10` 不低于 `dense_bm25_rrf_reranker.evidence_coverage@10` |
| 优秀版本 | decompose 在 `multi_hop` / `comparison` 子集上 EvidenceCoverage@10 提升 >= 5 个百分点 |
| 强展示版本 | EvidenceCoverage@10 提升 >= 8 个百分点，且 p95 检索延迟增长可解释 |

## 6. 引用与生成指标

### 6.1 Citation Accuracy

第一版定义：最终返回的 `sources` 中，至少一个 source 命中 `gold_doc_ids` 或 `gold_chunk_ids`，则该 query 的 citation hit = 1。

```text
Citation Accuracy@N = 引用命中 query 数 / 总 query 数
```

推荐拆成两级：

```text
citation_doc_accuracy = 引用命中 gold_doc_ids 的比例
citation_chunk_accuracy = 引用命中 gold_chunk_ids 的比例
```

通过阈值可以先设置为：

| 阶段 | 阈值 |
| --- | --- |
| 最小合格 | citation_doc_accuracy >= 0.70 |
| 优秀版本 | citation_doc_accuracy >= 0.80，citation_chunk_accuracy >= 0.60 |
| 强展示版本 | citation_doc_accuracy >= 0.85，citation_chunk_accuracy >= 0.70 |

阈值需要结合数据集难度调整，但必须在报告中固定。

### 6.2 Citation Grounding

定义：答案中的关键陈述是否能被引用片段支持。第一版可以人工抽检 20 条，后续可用 LLM-as-judge。

人工标注建议：

```text
2 = 完全被引用支持
1 = 部分被引用支持
0 = 引用不支持答案或答案与引用矛盾
```

通过阈值：

```text
平均分 >= 1.5，且 0 分比例 <= 20%
```

### 6.3 Answer Faithfulness

定义：答案是否只基于检索上下文回答，没有编造上下文外信息。

第一版建议使用人工抽检或 LLM-as-judge，输出：

```json
{
  "faithfulness_score": 0.0_to_1.0,
  "unsupported_claims": ["..."],
  "reason": "..."
}
```

通过阈值：

```text
faithfulness_score 平均 >= 0.75
unsupported_claims 平均 <= 1 条 / query
```

## 7. 延迟指标

需要记录阶段耗时，而不只记录 total latency。

| 指标 | 含义 |
| --- | --- |
| `route_ms` | RouterGraph 判断耗时 |
| `rewrite_ms` | query rewrite / HyDE / decompose 耗时 |
| `dense_ms` | 向量检索耗时 |
| `bm25_ms` | BM25 检索耗时 |
| `fusion_ms` | RRF 融合耗时 |
| `rerank_ms` | reranker 耗时 |
| `context_build_ms` | parent 回填、压缩和引用选择耗时 |
| `generate_ms` | LLM 生成耗时 |
| `total_ms` | 总耗时 |

至少输出：

```text
avg_total_ms
p50_total_ms
p95_total_ms
avg_rerank_ms
p95_rerank_ms
```

第一版建议阈值：

| 场景 | p95 total latency 目标 |
| --- | --- |
| 非 reranker 查询 | <= 4s，不含外部 LLM 极端波动 |
| reranker 查询 | <= 6s，不含外部 LLM 极端波动 |
| debug 接口 | <= 8s |

如果当前模型服务波动较大，报告中应拆分检索延迟和生成延迟，避免 LLM 延迟掩盖检索策略差异。

## 8. 策略通过阈值

### 8.1 最小合格版本

| 指标 | 通过标准 |
| --- | --- |
| `dense_bm25_rrf.hit_doc@5` | 不低于 `chroma_only.hit_doc@5` |
| `dense_bm25_rrf.recall_doc@10` | 不低于 `chroma_only.recall_doc@10` |
| `dense_bm25_rrf_reranker.mrr@10` | 高于或等于 `dense_bm25_rrf.mrr@10` |
| `citation_doc_accuracy` | >= 0.70 |
| `p95_retrieve_ms` | 有记录，并能分解阶段耗时 |

### 8.2 优秀版本

| 指标 | 通过标准 |
| --- | --- |
| `dense_bm25_rrf.hit_doc@5` | 相比 `chroma_only` 有明确提升，建议 >= +3 个百分点 |
| `dense_bm25_rrf_reranker.mrr@10` | 相比 `dense_bm25_rrf` 有明确提升，建议 >= +3 个百分点 |
| `strategy_matrix.mrr@10` | 不低于固定 reranker 策略，同时平均延迟更低或相近 |
| `citation_doc_accuracy` | >= 0.80 |
| `faithfulness_score` | >= 0.75 |

### 8.3 强展示版本

| 指标 | 通过标准 |
| --- | --- |
| 多策略报告 | 至少包含 5 种 baseline 对比 |
| 失败样例分析 | 至少归因 20 条失败样例 |
| p95 latency | 能解释 reranker、HyDE、decompose 的延迟成本 |
| citation_chunk_accuracy | >= 0.70，或清楚说明 chunk 标注不足 |

## 9. 失败样例格式

建议每条失败样例保存为 JSONL：

```json
{
  "query_id": "q_001",
  "query": "...",
  "question_type": "multi_hop",
  "strategy_name": "dense_bm25_rrf_reranker",
  "gold_doc_ids": ["doc_a"],
  "gold_chunk_ids": ["chunk_a_1"],
  "dense_results": ["chunk_x", "chunk_y"],
  "bm25_results": ["chunk_a_1", "chunk_z"],
  "fused_results": ["chunk_x", "chunk_a_1"],
  "reranked_results": ["chunk_x", "chunk_y", "chunk_a_1"],
  "selected_sources": ["chunk_x", "chunk_y"],
  "answer": "...",
  "failure_type": "reranker_top1_not_gold",
  "failure_reason": "BM25 召回了正确 chunk，但 reranker 将其排低，最终未进入上下文。",
  "metrics": {
    "hit_doc@5": 1,
    "citation_hit": 0,
    "total_ms": 3520
  }
}
```

推荐 failure_type 枚举：

| failure_type | 含义 |
| --- | --- |
| `missed_all_gold` | 所有召回都没命中 gold。 |
| `dense_missed_bm25_hit` | 向量漏召，但 BM25 命中。 |
| `bm25_missed_dense_hit` | BM25 漏召，但向量命中。 |
| `fusion_dropped_gold` | 单路召回命中，但融合后丢失。 |
| `reranker_top1_not_gold` | reranker 未把 gold 排到前面。 |
| `context_budget_dropped_gold` | gold 进入候选但被上下文预算丢弃。 |
| `citation_not_gold` | 答案引用没有命中 gold。 |
| `answer_unsupported` | 答案含有引用不支持的陈述。 |
| `acl_filtered_gold` | 权限过滤导致 gold 被排除，需要确认是否合理。 |
| `decompose_bad_subqueries` | 子问题偏离原始问题或丢失关键约束。 |
| `decompose_partial_evidence` | 子问题召回只覆盖部分 required evidence groups。 |
| `decompose_merge_dropped_group` | 各子问题召回命中证据，但跨子问题融合或上下文预算丢掉了某组证据。 |
| `decompose_latency_regression` | decompose 指标收益不足以解释额外延迟。 |

## 10. 评测脚本建议

建议脚本分层：

```text
backend/scripts/evaluate_retrieval_baselines.py
backend/scripts/evaluate_rag_strategy_matrix.py
backend/scripts/evaluate_citation_accuracy.py
backend/scripts/analyze_rag_failures.py
backend/scripts/report_rag_eval.py
```

输出目录：

```text
backend/data/eval_outputs/YYYYMMDD-HHMMSS/
  metrics_summary.json
  per_query_results.jsonl
  failures.jsonl
  strategy_comparison.csv
  report.md
```

## 11. 报告模板

每次评测报告至少包含：

```text
1. 数据集版本
2. 策略列表
3. 关键参数：topK、reranker_candidate_k、HyDE、decompose
4. 检索指标表：Hit@K、Recall@K、MRR
5. 引用指标表：Citation Accuracy、Grounding 抽检
6. 延迟指标表：avg、p50、p95
7. 失败样例 Top N
8. 结论：保留哪个策略，为什么
9. 下一步：修复哪些 failure_type
```

## 12. 近期最小评测闭环

近期先做到以下闭环即可：固定 EnterpriseRAG-Bench 或人工样例集，跑 `chroma_only`、`bm25_only`、`dense_bm25_rrf`、`dense_bm25_rrf_reranker` 四组；输出 Hit@5、Recall@10、MRR@10、avg/p95 latency；保存失败样例；生成一份 Markdown 报告。这个闭环完成后，再扩展 Citation Accuracy 和 Answer Faithfulness。