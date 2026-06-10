# RAGAS Generation Evaluation

## 2026-06-10 limit-10 baseline

### Purpose

Record a small RAGAS generation-quality baseline for EnterpriseRAG-Bench so future retrieval, reranking, prompt, or judge changes can be compared against the same style of output.

### Command

```powershell
conda run -n NexusKB python backend/scripts/evaluate_enterprise_rag_generation.py --limit 10 --judge-provider openai --judge-model gpt-5.5 --embedding-provider ollama --embedding-model qwen3-embedding:latest --embedding-base-url http://localhost:11434
```

### Run artifacts

- Run directory: `backend/data/eval_outputs/generation/20260610-080518-683746-generation-2bcc519/`
- Config: `backend/data/eval_outputs/generation/20260610-080518-683746-generation-2bcc519/config.json`
- Summary: `backend/data/eval_outputs/generation/20260610-080518-683746-generation-2bcc519/generation_ragas_summary.json`
- Details: `backend/data/eval_outputs/generation/20260610-080518-683746-generation-2bcc519/generation_ragas_details.jsonl`
- Failures: `backend/data/eval_outputs/generation/20260610-080518-683746-generation-2bcc519/generation_ragas_failures.jsonl`

### Configuration

| Field | Value |
| --- | --- |
| Questions | 10 |
| Judge provider | `openai` |
| Judge model | `gpt-5.5` |
| Embedding provider | `ollama` |
| Embedding model | `qwen3-embedding:latest` |
| Embedding base URL | `http://localhost:11434` |
| Git commit recorded by run | `2bcc519` |

### Summary metrics

| Metric | Value |
| --- | ---: |
| Status | `complete` |
| Questions | 10 |
| Valid RAGAS scores | 9 |
| Failures | 1 |
| Average latency | 5568.1588 ms |
| Faithfulness | 0.8060 |
| Answer relevancy | 0.7222 |
| Answer correctness | 0.6178 |
| Context precision | 0.6783 |
| Context recall | 0.7000 |

### Metric coverage

| Metric | Count | Rate |
| --- | ---: | ---: |
| Faithfulness | 10 | 1.0 |
| Answer relevancy | 10 | 1.0 |
| Answer correctness | 9 | 0.9 |
| Context precision | 10 | 1.0 |
| Context recall | 10 | 1.0 |

## Per-sample score snapshot

| Question | Avg score | Faithfulness | Relevancy | Correctness | Ctx precision | Ctx recall | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qst_0001` | 0.7102 | 0.7778 | 0.9129 | 0.4438 | 0.4167 | 1.0000 | Correct answer likely present, but context contains noisy distractors. |
| `qst_0002` | 0.7738 | 0.8333 | 0.4673 | 0.5685 | 1.0000 | 1.0000 | Context is precise, but answer relevancy/correctness are only moderate. |
| `qst_0003` | 0.9211 | 0.9600 | 0.8330 | 0.8958 | 0.9167 | 1.0000 | Strong sample. |
| `qst_0004` | 0.6330 | 0.8889 | 0.6721 | 0.6539 | 0.9500 | 0.0000 | Retrieved context looked focused but did not cover the reference evidence according to RAGAS. |
| `qst_0005` | 0.2806 | 0.3171 | 0.8055 | missing | 0.0000 | 0.0000 | Failure sample; answer correctness missing and evidence was not covered. |
| `qst_0006` | 0.2555 | 0.6000 | 0.5191 | 0.1584 | 0.0000 | 0.0000 | Lowest complete-score sample; retrieval evidence did not cover reference. |
| `qst_0007` | 0.7640 | 0.8333 | 0.6981 | 0.7888 | 0.5000 | 1.0000 | Answer quality okay despite noisy context. |
| `qst_0008` | 0.8786 | 1.0000 | 0.7030 | 0.6898 | 1.0000 | 1.0000 | Faithful and well-grounded; correctness still not perfect. |
| `qst_0009` | 0.8727 | 0.8750 | 0.7827 | 0.7060 | 1.0000 | 1.0000 | Strong sample. |
| `qst_0010` | 0.8917 | 0.9750 | 0.8287 | 0.6546 | 1.0000 | 1.0000 | Strong grounding; correctness moderate. |

## Low-score analysis

### `qst_0005`: MedThink EU outage failover question

- Scores: faithfulness 0.3171, answer relevancy 0.8055, context precision 0.0, context recall 0.0, answer correctness missing.
- Failure mode: entity/source mismatch. The question asks about MedThink-specific failover sequence, RPO/RTO, and US traffic time limit. Retrieved evidence included a NordBank data-residency exception and generic controlled failover test material, which are topically similar but not the MedThink reference.
- Observed answer behavior: the answer stayed relevant to failover but used the wrong organization/evidence. This explains high-ish relevancy with zero context precision/recall and low faithfulness.
- Likely root cause: retrieval/reranking over-weighted generic regional failover semantics and under-weighted the entity `MedThink` plus the source hint `gmail`.
- Optimization candidates:
  - Apply stronger entity matching/boosting for named customers or projects.
  - Treat `source_hints` as a hard or semi-hard filter for narrow factual questions.
  - Add reranker features or prompts that penalize wrong-entity documents even if they are semantically similar.
  - Add a decomposition step for multi-constraint questions: failover sequence, RPO, RTO, and US-shift time cap.

### `qst_0006`: routing policy engine failure-signal priority order

- Scores: faithfulness 0.6000, answer relevancy 0.5191, answer correctness 0.1584, context precision 0.0, context recall 0.0.
- Failure mode: evidence miss. RAGAS judged that retrieved contexts did not contain the reference answer, so generation was forced to infer or answer from adjacent material.
- Likely root cause: lexical/semantic retrieval did not anchor on the specific phrase `priority order` and the target document type/source (`google_drive` draft spec). The query is conceptually broad and can collide with many failover documents.
- Optimization candidates:
  - Add source-type filtering/boosting for `google_drive` when source hints are present.
  - Increase BM25 weight or exact phrase boost for `priority order`, `failure signals`, and `routing policy engine`.
  - Use query decomposition into `document/topic` + `requested field` to retrieve the right spec section.
  - Consider evidence coverage checks before generation: if the selected documents do not mention the requested ordered list, trigger retry/query rewrite.

### `qst_0004`: GCP Marketplace entitlement delay question

- Scores: faithfulness 0.8889, answer relevancy 0.6721, answer correctness 0.6539, context precision 0.95, context recall 0.0.
- Failure mode: partial or mismatched evidence. RAGAS considered contexts mostly relevant but still did not find reference coverage.
- Likely root cause: retrieved materials may discuss GCP Marketplace onboarding generally, but miss the exact recommendation for handling delayed subscription entitlement availability.
- Optimization candidates:
  - Strengthen retrieval for meeting/action-item style answers in Fireflies transcripts.
  - Add phrase/query rewrite around `entitlement not immediately available`, `delay`, `onboarding flow`, and `recommendation`.
  - Increase final top-k or rerank candidates for meeting transcripts where key answers may appear in short conversational snippets.

## Interpretation

This run is useful as a small generation-quality baseline, not a final benchmark. The system has acceptable grounding on many samples: faithfulness is above 0.8 and several samples have full context recall. The main weakness is not broad hallucination; it is evidence selection for entity-specific or multi-constraint factual questions. When retrieval misses the exact source, answer correctness drops sharply even if the answer remains topically relevant.

## Next comparison points

Future optimization runs should compare against this baseline on:

1. `valid_ragas_scores` and `failures`, to ensure judge stability does not regress.
2. `context_precision` and `context_recall`, especially for entity-specific questions.
3. `answer_correctness`, because this is currently the weakest complete metric.
4. Low-score samples `qst_0005` and `qst_0006`, which should be treated as regression fixtures after retrieval improvements.
5. Average latency, because reranker/source-filter/decomposition changes may improve correctness at the cost of runtime.
