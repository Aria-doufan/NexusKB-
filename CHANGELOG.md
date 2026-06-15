# Changelog

记录 NexusKB 每次有意义的项目改动。每次完成代码、文档、评估或发布相关变更时，都要同步追加一条简短记录。

## 2026-06-15 - v0.3.0

- 新增 Elasticsearch enterprise retrieval backend，并将其定位为企业检索默认后端，Chroma 保留为 baseline。
- 新增 `doc_semantic_type` 元数据贯通，覆盖 chunk 准备、索引、检索、评估和 RAG context。
- 新增 whitelist-driven metadata filter planner，支持 source/doc semantic type 的 hard/soft filter 决策。
- 新增 Elasticsearch hard filter 与 soft boost 检索逻辑，并接入 RagEvidenceWorkflow 的 hard → soft fallback。
- 新增 metadata filter evaluation CLI flags，并记录 policy hard/soft filter 与 Confluence hard-filter smoke 结果。
- 新增 RAG context 中 `doc_semantic_type` 输出，提升 evidence 可解释性。
- 发布 GitHub Release `v0.3.0`。
- 更新 GitHub README 首页，重写项目定位、核心能力、评估结果、Elasticsearch 启动说明和后续方向。
