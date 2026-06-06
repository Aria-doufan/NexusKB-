import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.rag import RagMetrics, RagResponse, RagSource, RagStrategySummary
from scripts.evaluate_enterprise_hybrid_retrieval import Question
from scripts.evaluate_enterprise_rag_generation import (
    CapturingTraceStore,
    build_rag_state,
    build_ragas_sample_dict,
    normalize_generation_rag_intent,
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
