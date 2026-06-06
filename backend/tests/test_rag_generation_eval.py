import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.rag import RagMetrics, RagResponse, RagSource, RagStrategySummary
from scripts.evaluate_enterprise_hybrid_retrieval import Question
from scripts.evaluate_enterprise_rag_generation import (
    CapturingTraceStore,
    build_ragas_sample_dict,
    response_to_generation_record,
    summarize_generation_records,
)


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
                        )
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


def test_summarize_generation_records_marks_incomplete_below_valid_score_threshold():
    records = [
        {"ragas_scores": {"faithfulness": 0.9, "answer_relevancy": 0.8}, "latency_ms": 100.0},
        {"ragas_error": "rate limit", "latency_ms": 200.0},
    ]

    summary = summarize_generation_records(records, intended_count=2)

    assert summary["questions"] == 2
    assert summary["valid_ragas_scores"] == 1
    assert summary["status"] == "incomplete"
    assert summary["average_latency_ms"] == 150.0
    assert summary["faithfulness"] == 0.9
    assert summary["answer_relevancy"] == 0.8


def test_capturing_trace_store_keeps_saved_traces():
    store = CapturingTraceStore()
    trace = SimpleNamespace(debug_id="dbg-1")

    import anyio

    anyio.run(store.save, trace)

    assert store.saved == [trace]
