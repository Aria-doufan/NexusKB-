# 企业 RAG 与检索模块设计

整理日期：2026-05-15

## 职责

企业 RAG 模块负责把 EnterpriseRAG-Bench 数据变成可检索、可评测、可生成回答的知识库链路。

当前目标不是一次性追求复杂 GraphRAG，而是先形成稳定闭环：

```text
数据准备 -> 分块 -> 入库 -> 召回 -> 融合 -> 重排 -> 评测 -> 策略迭代
```

## 当前数据与索引

| 项目 | 当前值 |
| --- | --- |
| 数据集 | EnterpriseRAG-Bench |
| 原始路径 | `dataset/EnterpriseRAG-Bench` |
| 问题文件 | `backend/data/enterprise_rag_bench/questions.jsonl` |
| parent chunks | `backend/data/enterprise_rag_bench/parent_chunks_parent_child.jsonl` |
| child chunks | `backend/data/enterprise_rag_bench/child_chunks_parent_child.jsonl` |
| Chroma collection | `enterprise_rag_bench_parent_child` |
| Chroma persist dir | `backend/data/chromadb_enterprise_parent_child` |
| Embedding model | `qwen3-embedding:latest` |

## 分块策略

采用 parent-child 分块：

- child chunk：用于检索，粒度更细。
- parent chunk：用于回填上下文，避免生成阶段只看到碎片。
- `parent_doc_id`：用于和 `expected_doc_ids` 计算命中指标。
- `parent_chunk_id`：用于从 child 检索结果回填 parent 文本。

## 检索策略

当前已验证三种方案：

| 方案 | 定位 |
| --- | --- |
| Chroma only | 向量召回 baseline，速度快 |
| Chroma + BM25 + RRF | 默认候选召回方案，召回提升明显 |
| Chroma + BM25 + RRF + reranker | 精排增强，Top1/MRR 提升明显，但延迟更高 |

当前工程判断：

- 默认应优先落地 Chroma + BM25 + RRF。
- reranker 适合复杂、精确、低置信度问题，不适合无条件默认。
- `source_hints` 先作为软信号，暂不硬过滤。

## 实验结论

完整结果见 [企业 RAG 检索评测实验](../../experiments/enterprise_retrieval_eval.md)。

阶段性结论：

- BM25 + RRF 能显著补足向量召回漏召。
- reranker 能显著提升 `hit@1` 和 `mrr@20`。
- reranker 不增加候选集合，所以不能替代混合召回。
- 后续重点应放在失败样例分析和策略矩阵，而不是继续盲目叠组件。

## 核心脚本

| 文件 | 说明 |
| --- | --- |
| `backend/scripts/prepare_enterprise_rag_bench.py` | 数据抽样与格式转换 |
| `backend/scripts/index_enterprise_chunks_chroma.py` | Chroma 入库 |
| `backend/scripts/evaluate_enterprise_retrieval.py` | Chroma baseline 评测 |
| `backend/scripts/evaluate_enterprise_hybrid_retrieval.py` | BM25/RRF/reranker 混合评测 |

## 下一步

- 分析失败样例，按 `question_type`、`source_type`、多文档问题拆分。
- 验证 `reranker_candidate_k=10` 与 `20` 的效果和延迟差异。
- 设计 RAG 策略矩阵，并接入 Router 的 `rag_intent`。
- 设计回答事实覆盖率评测，补齐生成质量指标。
