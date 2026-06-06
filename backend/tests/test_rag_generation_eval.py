import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.rag import RagMetrics, RagResponse, RagSource, RagStrategySummary
from scripts.evaluate_enterprise_hybrid_retrieval import Question
import scripts.evaluate_enterprise_rag_generation as rag_generation_eval
from scripts.evaluate_enterprise_rag_generation import (
    CapturingTraceStore,
    apply_ragas_scores,
    build_rag_state,
    build_ragas_sample_dict,
    normalize_generation_rag_intent,
    response_to_generation_record,
    summarize_generation_records,
)


FULL_RAGAS_SCORES = {
    "faithfulness": 0.9,
    "answer_relevancy": 0.8,
    "answer_correctness": 0.7,
    "context_precision": 0.6,
    "context_recall": 0.5,
}


def make_question():
    return Question(
        question_id="q1",
        question_type="fact_lookup",
        source_types=["policy"],
        question="Where is the PTO policy?",
        expected_doc_ids=["doc_a"],
        gold_answer="The PTO policy is in the HR handbook.",
        answer_facts=["HR handbook"],
        required_evidence_groups=[],
    )


def make_response():
    return RagResponse(
        request_id="req-1",
        debug_id="dbg-1",
        answer="The PTO policy is in the HR handbook.",
        sources=[
            RagSource(
                source_id="chunk_a",
                title="HR handbook",
                source_type="policy",
                parent_doc_id="doc_a",
                parent_chunk_id="chunk_a",
                section_heading="Leave",
                score=0.9,
            )
        ],
        strategy=RagStrategySummary(
            strategy_name="dense_bm25_rrf",
            retrieval_mode="hybrid",
            final_top_k=5,
        ),
        metrics=RagMetrics(total_ms=123.0),
    )


def test_normalize_generation_rag_intent_maps_benchmark_question_types():
    assert normalize_generation_rag_intent("semantic") == "semantic_query"
    assert normalize_generation_rag_intent("conflicting_info") == "comparison"
    assert normalize_generation_rag_intent("project_related") == "multi_hop"
    assert normalize_generation_rag_intent("lookup") == "fact_lookup"


def test_build_rag_state_normalizes_benchmark_question_type_for_router():
    question = make_question()
    question.question_type = "semantic"

    state = build_rag_state(question)

    assert state.rag_intent == "semantic_query"


def test_response_to_generation_record_uses_trace_context_texts():
    trace = SimpleNamespace(
        retrieval_attempts=[
            SimpleNamespace(
                attempt=SimpleNamespace(
                    selected_documents=[
                        SimpleNamespace(
                            text="The HR handbook contains the PTO policy.",
                            child_text="PTO policy child text.",
                            parent_doc_id="doc_a",
                            parent_chunk_id="chunk_a",
                            source_id="chunk_a",
                        ),
                        SimpleNamespace(
                            text="The engineering handbook contains deployment policy.",
                            child_text="Deployment policy child text.",
                            parent_doc_id="doc_b",
                            parent_chunk_id="chunk_b",
                            source_id="chunk_b",
                        ),
                    ]
                )
            )
        ]
    )

    record = response_to_generation_record(make_question(), make_response(), trace, latency_ms=123.0)

    assert record["question_id"] == "q1"
    assert record["question_type"] == "fact_lookup"
    assert record["question"] == "Where is the PTO policy?"
    assert record["reference"] == "The PTO policy is in the HR handbook."
    assert record["answer"] == "The PTO policy is in the HR handbook."
    assert record["retrieved_contexts"] == ["The HR handbook contains the PTO policy."]
    assert record["source_doc_ids"] == ["doc_a"]
    assert record["source_chunk_ids"] == ["chunk_a"]
    assert record["latency_ms"] == 123.0


def test_response_to_generation_record_matches_exact_chunk_before_parent_doc():
    trace = SimpleNamespace(
        retrieval_attempts=[
            SimpleNamespace(
                attempt=SimpleNamespace(
                    selected_documents=[
                        SimpleNamespace(
                            text="Parent document chunk one text.",
                            parent_doc_id="doc_a",
                            parent_chunk_id="chunk_a1",
                            source_id="chunk_a1",
                        ),
                        SimpleNamespace(
                            text="Parent document chunk two text.",
                            parent_doc_id="doc_a",
                            parent_chunk_id="chunk_a2",
                            source_id="chunk_a2",
                        ),
                    ]
                )
            )
        ]
    )
    response = make_response()
    response.sources = [
        RagSource(
            source_id="chunk_a2",
            title="HR handbook",
            source_type="policy",
            parent_doc_id="doc_a",
            parent_chunk_id="chunk_a2",
            section_heading="Leave",
            score=0.9,
        )
    ]

    record = response_to_generation_record(make_question(), response, trace, latency_ms=123.0)

    assert record["retrieved_contexts"] == ["Parent document chunk two text."]
    assert record["source_chunk_ids"] == ["chunk_a2"]


def test_response_to_generation_record_orders_contexts_by_final_sources():
    trace = SimpleNamespace(
        retrieval_attempts=[
            SimpleNamespace(
                attempt=SimpleNamespace(
                    selected_documents=[
                        SimpleNamespace(
                            text="Chunk A context text.",
                            parent_doc_id="doc_a",
                            parent_chunk_id="chunk_a",
                            source_id="chunk_a",
                        ),
                        SimpleNamespace(
                            text="Chunk B context text.",
                            parent_doc_id="doc_b",
                            parent_chunk_id="chunk_b",
                            source_id="chunk_b",
                        ),
                    ]
                )
            )
        ]
    )
    response = make_response()
    response.sources = [
        RagSource(
            source_id="chunk_b",
            title="Benefits handbook",
            source_type="policy",
            parent_doc_id="doc_b",
            parent_chunk_id="chunk_b",
            section_heading="Benefits",
            score=0.8,
        ),
        RagSource(
            source_id="chunk_a",
            title="HR handbook",
            source_type="policy",
            parent_doc_id="doc_a",
            parent_chunk_id="chunk_a",
            section_heading="Leave",
            score=0.9,
        ),
    ]

    record = response_to_generation_record(make_question(), response, trace, latency_ms=123.0)

    assert record["retrieved_contexts"] == ["Chunk B context text.", "Chunk A context text."]


def test_build_ragas_sample_dict_maps_expected_fields():
    record = {
        "question": "Where is the PTO policy?",
        "retrieved_contexts": ["The HR handbook contains the PTO policy."],
        "answer": "The PTO policy is in the HR handbook.",
        "reference": "The PTO policy is in the HR handbook.",
    }

    sample = build_ragas_sample_dict(record)

    assert sample == {
        "user_input": "Where is the PTO policy?",
        "retrieved_contexts": ["The HR handbook contains the PTO policy."],
        "response": "The PTO policy is in the HR handbook.",
        "reference": "The PTO policy is in the HR handbook.",
    }


def test_build_ragas_evaluation_components_configures_llm_embeddings_and_metric_instances(monkeypatch):
    calls = []

    class FakeDataset:
        @classmethod
        def from_list(cls, samples):
            return {"samples": samples}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLMWrapper:
        def __init__(self, chat_model):
            self.chat_model = chat_model

    class FakeEmbeddingsWrapper:
        def __init__(self, embedding_model):
            self.embedding_model = embedding_model

    def metric_class(name):
        class FakeMetric:
            def __init__(self, **kwargs):
                self.name = name
                self.kwargs = kwargs
                calls.append((name, kwargs))

        return FakeMetric

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(sys.modules, "langchain_openai.chat_models", SimpleNamespace(ChatOpenAI=FakeChatOpenAI))
    monkeypatch.setitem(sys.modules, "langchain_openai.embeddings", SimpleNamespace(OpenAIEmbeddings=FakeOpenAIEmbeddings))
    monkeypatch.setitem(sys.modules, "ragas", SimpleNamespace(evaluate=lambda *args, **kwargs: None))
    monkeypatch.setitem(sys.modules, "ragas.llms", SimpleNamespace(LangchainLLMWrapper=FakeLLMWrapper))
    monkeypatch.setitem(sys.modules, "ragas.embeddings", SimpleNamespace(LangchainEmbeddingsWrapper=FakeEmbeddingsWrapper))
    monkeypatch.setitem(
        sys.modules,
        "ragas.metrics",
        SimpleNamespace(
            AnswerCorrectness=metric_class("answer_correctness"),
            AnswerRelevancy=metric_class("answer_relevancy"),
            ContextPrecision=metric_class("context_precision"),
            ContextRecall=metric_class("context_recall"),
            Faithfulness=metric_class("faithfulness"),
        ),
    )

    dataset_class, evaluate, metrics, llm = rag_generation_eval._build_ragas_evaluation_components(
        "gpt-4o-mini",
        "text-embedding-3-large",
    )

    assert dataset_class is FakeDataset
    assert evaluate is sys.modules["ragas"].evaluate
    assert llm.chat_model.kwargs == {"model": "gpt-4o-mini", "temperature": 0}
    assert llm is calls[0][1]["llm"]
    answer_relevancy = next(kwargs for name, kwargs in calls if name == "answer_relevancy")
    answer_correctness = next(kwargs for name, kwargs in calls if name == "answer_correctness")
    assert answer_relevancy["embeddings"].embedding_model.kwargs == {"model": "text-embedding-3-large"}
    assert answer_correctness["embeddings"].embedding_model.kwargs == {"model": "text-embedding-3-large"}
    assert [metric.name for metric in metrics] == [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]


def test_apply_ragas_scores_attaches_scores_and_records_metric_errors():
    records = [
        {
            "question_id": "q1",
            "question": "Where is the PTO policy?",
            "retrieved_contexts": ["The HR handbook contains the PTO policy."],
            "answer": "The PTO policy is in the HR handbook.",
            "reference": "The PTO policy is in the HR handbook.",
        },
        {
            "question_id": "q2",
            "question": "Unknown?",
            "retrieved_contexts": [],
            "answer": "",
            "reference": "Reference",
        },
    ]

    def fake_evaluator(samples):
        assert samples[0]["user_input"] == "Where is the PTO policy?"
        return [
            FULL_RAGAS_SCORES,
            RuntimeError("judge failed"),
        ]

    scored = apply_ragas_scores(records, fake_evaluator)

    assert scored[0]["ragas_scores"] == FULL_RAGAS_SCORES
    assert scored[1]["ragas_error"] == "judge failed"


def test_apply_ragas_scores_marks_empty_score_result_as_error():
    records = [
        {
            "question_id": "q1",
            "question": "Where is the PTO policy?",
            "retrieved_contexts": ["The HR handbook contains the PTO policy."],
            "answer": "The PTO policy is in the HR handbook.",
            "reference": "The PTO policy is in the HR handbook.",
        }
    ]

    scored = apply_ragas_scores(records, lambda samples: [{}])

    assert scored[0]["ragas_error"] == "RAGAS returned no numeric scores"
    assert not scored[0].get("ragas_scores")


def test_apply_ragas_scores_preserves_partial_scores_and_marks_missing_metrics():
    records = [
        {
            "question_id": "q1",
            "question": "Where is the PTO policy?",
            "retrieved_contexts": ["The HR handbook contains the PTO policy."],
            "answer": "The PTO policy is in the HR handbook.",
            "reference": "The PTO policy is in the HR handbook.",
        }
    ]

    scored = apply_ragas_scores(records, lambda samples: [{"faithfulness": 0.9}])

    assert scored[0]["ragas_scores"] == {"faithfulness": 0.9}
    assert "RAGAS missing metrics" in scored[0]["ragas_error"]
    assert "answer_relevancy" in scored[0]["ragas_error"]
    assert not rag_generation_eval.has_valid_ragas_scores(scored[0])


def test_apply_ragas_scores_marks_missing_results_as_errors():
    records = [
        {
            "question_id": "q1",
            "question": "Where is the PTO policy?",
            "retrieved_contexts": ["The HR handbook contains the PTO policy."],
            "answer": "The PTO policy is in the HR handbook.",
            "reference": "The PTO policy is in the HR handbook.",
        },
        {
            "question_id": "q2",
            "question": "Unknown?",
            "retrieved_contexts": [],
            "answer": "",
            "reference": "Reference",
        },
    ]

    scored = apply_ragas_scores(records, lambda samples: [{"faithfulness": 0.9}])

    assert scored[0]["ragas_scores"] == {"faithfulness": 0.9}
    assert scored[1]["ragas_error"] == "RAGAS returned fewer results than samples"


def test_summarize_generation_records_marks_incomplete_below_valid_score_threshold():
    records = [
        {"ragas_scores": FULL_RAGAS_SCORES, "latency_ms": 100.0},
        {"ragas_scores": {"faithfulness": 0.3}, "ragas_error": "RAGAS missing metrics", "latency_ms": 200.0},
        {"ragas_scores": {}, "latency_ms": 300.0},
    ]

    summary = summarize_generation_records(records, intended_count=3)

    assert summary["questions"] == 3
    assert summary["valid_ragas_scores"] == 1
    assert summary["status"] == "incomplete"
    assert summary["failures"] == 2
    assert summary["average_latency_ms"] == 200.0
    assert summary["faithfulness"] == 0.6
    assert summary["answer_relevancy"] == 0.8
    assert summary["metric_coverage"] == {
        "faithfulness": 2,
        "answer_relevancy": 1,
        "answer_correctness": 1,
        "context_precision": 1,
        "context_recall": 1,
    }
    assert summary["metric_coverage_rate"] == {
        "faithfulness": 0.6667,
        "answer_relevancy": 0.3333,
        "answer_correctness": 0.3333,
        "context_precision": 0.3333,
        "context_recall": 0.3333,
    }


def test_write_generation_outputs_creates_standard_files(tmp_path):
    config = {
        "git_commit": "abc123",
        "dataset_path": "questions.jsonl",
        "question_count": 1,
        "judge_provider": "openai",
        "judge_model": "gpt-4o",
    }
    records = [
        {
            "question_id": "q1",
            "question_type": "fact_lookup",
            "question": "Where is the PTO policy?",
            "reference": "The PTO policy is in HR handbook.",
            "answer": "The PTO policy is in HR handbook.",
            "retrieved_contexts": ["HR handbook PTO policy."],
            "source_doc_ids": ["doc_a"],
            "source_chunk_ids": ["chunk_a"],
            "ragas_scores": FULL_RAGAS_SCORES,
            "latency_ms": 100.0,
        },
    ]
    summary = summarize_generation_records(records, intended_count=1)

    from scripts.evaluate_enterprise_rag_generation import write_generation_outputs

    run_dir = write_generation_outputs(tmp_path / "outputs", tmp_path / "baseline", config, records, summary)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "generation_ragas_summary.json").exists()
    assert (run_dir / "generation_ragas_details.jsonl").exists()
    assert (run_dir / "generation_ragas_failures.jsonl").exists()
    assert (run_dir / "report.md").exists()
    saved_summary = json.loads((run_dir / "generation_ragas_summary.json").read_text(encoding="utf-8"))
    assert saved_summary["faithfulness"] == 0.9
    assert saved_summary["status"] == "complete"
    assert saved_summary["failures"] == 0
    failures = [
        json.loads(line)
        for line in (run_dir / "generation_ragas_failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert failures == []
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "RAG Generation Evaluation Report" in report
    assert "gpt-4o" in report
    assert "| status | complete |" in report
    assert "| failures | 0 |" in report
    assert "## Metric Coverage" in report
    assert "| faithfulness | 1 | 1.0 |" in report


def test_write_generation_outputs_includes_empty_ragas_scores_in_failures(tmp_path):
    records = [
        {
            "question_id": "q-empty-scores",
            "question_type": "fact_lookup",
            "question": "Where is the PTO policy?",
            "reference": "The PTO policy is in HR handbook.",
            "answer": "The PTO policy is in HR handbook.",
            "retrieved_contexts": ["HR handbook PTO policy."],
            "source_doc_ids": ["doc_a"],
            "source_chunk_ids": ["chunk_a"],
            "ragas_scores": {},
            "latency_ms": 100.0,
        }
    ]
    summary = summarize_generation_records(records, intended_count=1)

    run_dir = rag_generation_eval.write_generation_outputs(
        tmp_path / "outputs",
        tmp_path / "baseline",
        {"judge_provider": "openai", "judge_model": "gpt-4o"},
        records,
        summary,
    )

    failures = [
        json.loads(line)
        for line in (run_dir / "generation_ragas_failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [failure["question_id"] for failure in failures] == ["q-empty-scores"]


def test_capturing_trace_store_keeps_saved_traces():
    store = CapturingTraceStore()
    trace = SimpleNamespace(debug_id="dbg-1")

    import anyio

    anyio.run(store.save, trace)

    assert store.saved == [trace]
