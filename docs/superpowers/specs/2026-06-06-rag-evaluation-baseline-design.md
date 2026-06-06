# RAG Evaluation Baseline Design

Date: 2026-06-06

## 1. Goal and Scope

The first version establishes a repeatable offline evaluation baseline for the NexusKB Agentic RAG system. The goal is to compare future RAG changes against a fixed dataset and fixed metrics before optimizing retrieval, routing, reranking, decomposition, or generation behavior.

### In scope

1. Build an offline evaluation baseline over the existing EnterpriseRAG-Bench dataset.
2. Split evaluation into retrieval quality and generation quality.
3. Use project-owned retrieval metrics for reproducible ranking evaluation.
4. Use RAGAS with a GPT judge for generation quality evaluation.
5. Produce archived run artifacts that can be compared with the current baseline.
6. Prepare, but not enforce, future CI regression gates.

### Out of scope for the first version

- Online production monitoring.
- Mandatory CI failure on metric regression.
- A full evaluation framework rewrite.
- Replacing Chroma, BM25, reranker, LangGraph, or EnterpriseRagGraph.
- Building a dashboard UI.

The first version should extend the existing evaluation scripts instead of introducing a large new subsystem. Once the offline baseline stabilizes, later work can extract reusable modules and connect smoke checks to CI.

## 2. Overall Architecture

The evaluation system has two offline tracks that share the same question set.

```text
EnterpriseRAG-Bench questions
  ├─ Retrieval evaluation
  │    ├─ run configured retrieval strategy
  │    ├─ compute Recall / Precision / NDCG / MRR / MAP
  │    ├─ compute evidence coverage for multi-hop/comparison questions
  │    └─ write summary, per-query details, and failures
  │
  └─ Generation evaluation
       ├─ run EnterpriseRagGraph to generate answers
       ├─ collect question, retrieved contexts, response, and reference
       ├─ evaluate with RAGAS using GPT as judge
       └─ write summary, per-query details, and failures
```

The retrieval track remains the source of truth for ranking quality because it uses deterministic gold document identifiers. RAGAS context metrics are useful as a second view of context quality, but they do not replace the project-owned retrieval metrics.

## 3. Evaluation Dataset

The first baseline uses:

```text
backend/data/enterprise_rag_bench/questions.jsonl
```

Each question should provide the fields already used by the current scripts:

```json
{
  "question_id": "q_001",
  "question_type": "fact_lookup",
  "question": "...",
  "expected_doc_ids": ["..."],
  "gold_answer": "...",
  "answer_facts": ["..."],
  "required_evidence_groups": [["..."]]
}
```

The report must group metrics by `question_type`. The first report should explicitly show coverage for at least these categories when present in the dataset:

- `fact_lookup`
- `procedure`
- `semantic_query`
- `multi_hop`
- `comparison`
- `constrained`
- `conflicting_info`
- `completeness`

If a category has too few samples to be meaningful, `report.md` should mark it as insufficient baseline coverage rather than hiding it.

## 4. Retrieval Evaluation Design

### 4.1 Script boundary

Extend the existing script first:

```text
backend/scripts/evaluate_enterprise_hybrid_retrieval.py
```

The script already supports multiple methods, including Chroma-only, BM25-only, RRF hybrid retrieval, reranking, strategy matrix, and decomposition variants. The first version should keep that CLI shape and enrich the metrics and outputs.

### 4.2 Metrics

The final retrieval metric set is:

| Metric | Purpose |
| --- | --- |
| `Recall@K` | Measures whether all gold documents are retrieved, especially for multi-document questions. |
| `Precision@K` | Measures how much TopK context is relevant, limiting noisy contexts. |
| `NDCG@K` | Measures ranked relevance quality and supports future graded relevance labels. |
| `MRR@K` | Measures where the first relevant document appears; useful for reranker validation. |
| `MAP@K` | Measures ranking quality across all relevant documents. |
| `EvidenceCoverage@K` | Measures whether required evidence groups are covered for multi-hop/comparison questions. |

The default K values remain:

```text
1, 5, 10, 20
```

The report should emphasize:

```text
Recall@10
Precision@10
NDCG@10
MRR@20
MAP@20
EvidenceCoverage@10
average_latency_ms
```

### 4.3 Ranking formulas

For binary relevance in the first version:

```text
DCG@K = sum((2^rel_i - 1) / log2(i + 1)) for i in 1..K
IDCG@K = ideal DCG for the gold set at K
NDCG@K = DCG@K / IDCG@K
```

`rel_i` is `1` when the ranked parent document is in `expected_doc_ids`, otherwise `0`.

```text
AP@K = sum(Precision@i for each relevant hit at rank i <= K) / min(number_of_gold_docs, K)
MAP@K = mean(AP@K over all questions)
```

MRR remains:

```text
MRR@K = mean(1 / first_relevant_rank_within_K)
```

### 4.4 Failure classification

Retrieval failures should be written to `retrieval_failures.jsonl` with explicit reasons:

- `no_gold_hit`
- `low_recall`
- `low_precision`
- `low_ndcg`
- `low_map`
- `evidence_group_missing`
- `reranker_top1_miss`

A single question can have multiple failure reasons.

## 5. Generation Evaluation Design

### 5.1 Script boundary

Add a generation evaluation script:

```text
backend/scripts/evaluate_enterprise_rag_generation.py
```

The script should run the current Agentic RAG path through `EnterpriseRagGraph`, collect the generated answer and selected contexts, then pass the records to RAGAS.

### 5.2 RAGAS dataset fields

Each sample should map project data into RAGAS-compatible fields:

```text
user_input          = question
retrieved_contexts  = selected source/context texts
response            = EnterpriseRagGraph answer
reference           = gold_answer
```

The script can build an `EvaluationDataset` using RAGAS `SingleTurnSample` records or an equivalent dataset accepted by `ragas.evaluate`.

### 5.3 RAGAS metrics

The first version should use:

| Metric | Purpose |
| --- | --- |
| `faithfulness` | Checks whether the answer is supported by retrieved contexts. |
| `answer_relevancy` | Checks whether the answer addresses the user question. |
| `answer_correctness` | Checks whether the answer matches the reference answer. |
| `context_precision` | Checks whether useful contexts are ranked near the top from RAGAS' perspective. |
| `context_recall` | Checks whether contexts cover information needed for the reference answer. |

### 5.4 Judge model configuration

Use GPT as the default first-version RAGAS judge:

```text
OPENAI_API_KEY=...
RAGAS_JUDGE_PROVIDER=openai
RAGAS_JUDGE_MODEL=gpt-4o
RAGAS_EVAL_TIMEOUT_SEC=...
```

The implementation should keep the judge configurable so later runs can switch to DashScope/Qwen, an OpenAI-compatible endpoint, or a local model through a RAGAS-supported wrapper.

No API keys should be committed.

### 5.5 Per-query generation record

Each generation detail should include:

```json
{
  "question_id": "...",
  "question_type": "...",
  "question": "...",
  "reference": "...",
  "answer": "...",
  "retrieved_contexts": ["..."],
  "source_doc_ids": ["..."],
  "source_chunk_ids": ["..."],
  "ragas_scores": {
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "answer_correctness": 0.0,
    "context_precision": 0.0,
    "context_recall": 0.0
  },
  "latency_ms": 1234.5
}
```

## 6. Output Artifacts

Each evaluation run writes to a timestamped run directory:

```text
backend/data/eval_outputs/<YYYYMMDD-HHMMSS>-<short_commit>/
  config.json
  retrieval_summary.json
  retrieval_details.jsonl
  retrieval_failures.jsonl
  generation_ragas_summary.json
  generation_ragas_details.jsonl
  generation_ragas_failures.jsonl
  report.md
```

`config.json` should capture:

- git commit
- dataset path
- question count
- strategy name
- K values
- embedding model
- Chroma collection and persist directory
- reranker model and device when used
- judge provider/model for RAGAS
- run timestamp
- CLI arguments

`report.md` should summarize:

- run configuration
- retrieval totals
- generation totals
- metrics by question type
- comparison with the current baseline when available
- top failure examples
- whether the run is complete enough to trust
- whether updating the baseline is recommended

## 7. Baseline Storage and Comparison

The current baseline should be stored separately from individual runs:

```text
backend/data/eval_baselines/current/
  config.json
  retrieval_summary.json
  generation_ragas_summary.json
```

The first version performs manual baseline comparison instead of failing CI automatically. The report should calculate deltas against the current baseline when the baseline directory exists.

Future CI thresholds can use rules like:

```text
Recall@10 must not drop by more than 0.02
Faithfulness must not drop by more than 0.05
Average latency must not increase by more than 30%
```

These thresholds are intentionally not enforced in the first version.

## 8. Error Handling and Incomplete Runs

The evaluation should continue when individual questions fail.

Rules:

- If answer generation fails, write `generation_error` for that question and continue.
- If RAGAS/GPT evaluation fails, write `ragas_error` for that question and continue.
- If retrieval returns no context, still write the detail row; generation can return an insufficient-evidence answer.
- If fewer than 80% of intended generation samples receive valid RAGAS scores, mark the generation run as `incomplete` in the summary and report.
- If config or dataset loading fails, stop the run because the baseline would be invalid.

## 9. Testing Strategy

Testing should focus on deterministic evaluation logic and dataset conversion rather than live LLM calls.

Add or extend tests around:

```text
backend/tests/test_rag_evaluation_metrics.py
backend/tests/test_rag_generation_eval.py
```

Coverage should include:

- `precision@K`
- `recall@K`
- `ndcg@K`
- `mrr@K`
- `map@K`
- multiple gold documents
- no-hit queries
- relevant documents at different ranks
- question type grouped summaries
- RAGAS input field assembly
- RAGAS/GPT failure recording and continuation

Tests should mock the RAGAS evaluator and must not call GPT.

## 10. Automation Boundary

The first version supports manual offline automation:

```text
run evaluation command
  -> produce run directory
  -> compare with baseline when available
  -> generate report.md
```

It should prepare flags for later CI use:

```text
--ci
--limit
--baseline-dir
--fail-on-regression
```

The flags can exist before all CI behavior is enforced, but the first version should not fail PRs based on GPT-judged RAGAS scores.

## 11. Suggested Commands

Retrieval evaluation:

```powershell
conda run -n nexuskb python backend/scripts/evaluate_enterprise_hybrid_retrieval.py --method strategy_matrix_decompose
```

Generation evaluation:

```powershell
conda run -n nexuskb python backend/scripts/evaluate_enterprise_rag_generation.py --limit 50
```

The generation limit keeps GPT cost and runtime controlled while establishing the first baseline.

## 12. Implementation Phases

### Phase 1: Metric core

- Add or extract deterministic retrieval metric functions.
- Implement NDCG and MAP.
- Add unit tests for metric edge cases.

### Phase 2: Retrieval baseline report

- Extend the existing retrieval script output.
- Add question type grouped summaries.
- Add failure classification.
- Write run config and standardized run directory outputs.

### Phase 3: Generation RAGAS evaluation

- Add the generation evaluation script.
- Run `EnterpriseRagGraph` over selected questions.
- Convert records into RAGAS samples.
- Evaluate with GPT judge through configurable RAGAS integration.
- Write generation summary, details, and failures.

### Phase 4: Unified report and baseline comparison

- Merge retrieval and generation summaries into `report.md`.
- Compare with `backend/data/eval_baselines/current` when present.
- Add recommended baseline update notes.

## 13. Completion Criteria

The first version is complete when:

1. Retrieval evaluation produces Recall, Precision, NDCG, MRR, MAP, EvidenceCoverage, latency, grouped summaries, and failures.
2. Generation evaluation produces RAGAS faithfulness, answer relevancy, answer correctness, context precision, and context recall using GPT judge.
3. Each run writes a timestamped artifact directory.
4. The report compares current results with the saved baseline when available.
5. Tests cover deterministic metrics and RAGAS input assembly without live GPT calls.
6. Commands run through the `nexuskb` conda environment.
7. No secrets are committed.
