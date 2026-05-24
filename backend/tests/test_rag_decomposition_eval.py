import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_enterprise_hybrid_retrieval import (
    Candidate,
    Question,
    decompose_question_for_eval,
    evaluate_question,
    evidence_coverage_at_k,
    load_questions,
    method_needs_reranker,
    normalize_method,
    should_rerank_question,
    summarize,
    write_outputs,
)


class FakeStore:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.queries = []

    def similarity_search_with_score(self, question, k, filter=None):
        self.queries.append(question)
        return self.results_by_query.get(question, [])[:k]


class FakeBM25:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        return self.results_by_query.get(query, [])[:k]


class FakeReranker:
    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text
        self.pairs = []

    def predict(self, pairs, batch_size=1):
        self.pairs.extend(pairs)
        return [self.scores_by_text.get(document, 0.0) for _query, document in pairs]



def make_document(chunk_id, parent_doc_id="doc", source_type="policy", text="text"):
    return SimpleNamespace(
        metadata={
            "chunk_id": chunk_id,
            "parent_doc_id": parent_doc_id,
            "parent_chunk_id": f"parent_{chunk_id}",
            "source_type": source_type,
            "title": "Title",
            "section_heading": "Section",
        },
        page_content=text,
    )


def make_bm25_candidate(chunk_id, parent_doc_id="doc", source_type="policy"):
    return Candidate(
        chunk_id=chunk_id,
        parent_doc_id=parent_doc_id,
        parent_chunk_id=f"parent_{chunk_id}",
        source_type=source_type,
        title="Title",
        section_heading="Section",
        text="text",
        bm25_rank=1,
        bm25_score=1.0,
    )



def test_normalize_method_accepts_strategy_matrix_decompose():
    assert normalize_method("strategy_matrix_decompose") == "strategy_matrix_decompose"


def test_strategy_matrix_decompose_loads_and_applies_reranker_for_complex_question_types():
    assert method_needs_reranker("strategy_matrix_decompose") is True

    for question_type in ["semantic", "multi_hop", "comparison"]:
        question = Question(
            question_id="q1",
            question_type=question_type,
            source_types=[],
            question="What policy applies?",
            expected_doc_ids=[],
            gold_answer="",
            answer_facts=[],
            required_evidence_groups=[],
        )

        assert should_rerank_question("strategy_matrix_decompose", question) is True


def test_evidence_coverage_at_k_requires_each_group_to_be_covered():
    ranked_chunk_ids = ["chunk_a", "chunk_x", "chunk_c"]
    required_groups = [["chunk_a", "chunk_b"], ["chunk_c"], ["chunk_d"]]

    assert evidence_coverage_at_k(ranked_chunk_ids, required_groups, max_k=3) == 2 / 3
    assert evidence_coverage_at_k(ranked_chunk_ids, required_groups, max_k=1) == 1 / 3


def test_evidence_coverage_at_k_returns_zero_without_groups():
    assert evidence_coverage_at_k(["chunk_a"], [], max_k=3) == 0.0


def test_load_questions_reads_required_evidence_groups(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question_type": "multi_hop",
                "source_types": ["policy"],
                "question": "What policies apply?",
                "expected_doc_ids": ["doc_a"],
                "gold_answer": "Answer",
                "answer_facts": ["Fact one", "Fact two"],
                "required_evidence_groups": [["chunk_a", "chunk_b", None], [], ["chunk_c"], [None]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    question = load_questions(questions_path)[0]

    assert question.required_evidence_groups == [["chunk_a", "chunk_b"], ["chunk_c"]]


def test_decompose_question_for_eval_uses_first_four_non_empty_answer_facts_for_complex_questions():
    question = Question(
        question_id="q1",
        question_type="comparison",
        source_types=[],
        question="Compare the policies",
        expected_doc_ids=[],
        gold_answer="",
        answer_facts=["", "Fact one", "Fact two", "Fact three", "Fact four", "Fact five"],
        required_evidence_groups=[],
    )

    assert decompose_question_for_eval(question) == [
        "Fact one",
        "Fact two",
        "Fact three",
        "Fact four",
    ]


def test_decompose_question_for_eval_ignores_whitespace_answer_facts_for_complex_questions():
    question = Question(
        question_id="q1",
        question_type="comparison",
        source_types=[],
        question="Compare the policies",
        expected_doc_ids=[],
        gold_answer="",
        answer_facts=["   ", "Fact one", "\t"],
        required_evidence_groups=[],
    )

    assert decompose_question_for_eval(question) == ["Fact one"]


def test_decompose_question_for_eval_falls_back_to_question_when_complex_facts_are_empty():
    question = Question(
        question_id="q1",
        question_type="multi_hop",
        source_types=[],
        question="Which policies apply?",
        expected_doc_ids=[],
        gold_answer="",
        answer_facts=["", "", ""],
        required_evidence_groups=[],
    )

    assert decompose_question_for_eval(question) == ["Which policies apply?"]


def test_decompose_question_for_eval_keeps_simple_questions_whole():
    question = Question(
        question_id="q1",
        question_type="lookup",
        source_types=[],
        question="What is the policy?",
        expected_doc_ids=[],
        gold_answer="",
        answer_facts=["Fact one"],
        required_evidence_groups=[],
    )

    assert decompose_question_for_eval(question) == ["What is the policy?"]


def test_strategy_matrix_decompose_runs_deterministic_sub_queries_and_merges_best_scores():
    question = Question(
        question_id="q1",
        question_type="multi_hop",
        source_types=["policy"],
        question="Which policies apply?",
        expected_doc_ids=["doc_a", "doc_c"],
        gold_answer="",
        answer_facts=["Fact one", "Fact two"],
        required_evidence_groups=[["chunk_a"], ["chunk_c"]],
    )
    store = FakeStore(
        {
            "Fact one": [(make_document("chunk_a", "doc_a"), 0.1)],
            "Fact two": [(make_document("chunk_c", "doc_c"), 0.1)],
        }
    )
    bm25 = FakeBM25(
        {
            "Fact one": [make_bm25_candidate("chunk_a", "doc_a")],
            "Fact two": [make_bm25_candidate("chunk_b", "doc_b")],
        }
    )

    detail = evaluate_question(
        method="strategy_matrix_decompose",
        store=store,
        bm25=bm25,
        question=question,
        chroma_search_k=5,
        bm25_search_k=5,
        rrf_k=60,
        source_boost=0.15,
        k_values=[1, 3],
        where=None,
        reranker_model=None,
        parent_texts={},
        reranker_candidate_k=20,
        reranker_batch_size=4,
    )

    assert store.queries == ["Fact one", "Fact two"]
    assert bm25.queries == ["Fact one", "Fact two"]
    assert detail["retrieved_chunk_ids"][:3] == ["chunk_a", "chunk_c", "chunk_b"]
    assert detail["evidence_coverage"] == 1.0


def test_strategy_matrix_decompose_skips_candidates_without_chunk_ids():
    question = Question(
        question_id="q1",
        question_type="multi_hop",
        source_types=["policy"],
        question="Which policies apply?",
        expected_doc_ids=["doc_a"],
        gold_answer="",
        answer_facts=["Fact one", "Fact two"],
        required_evidence_groups=[["chunk_a"]],
    )
    store = FakeStore(
        {
            "Fact one": [(make_document("", "doc_empty"), 0.1)],
            "Fact two": [(make_document("chunk_a", "doc_a"), 0.1)],
        }
    )
    bm25 = FakeBM25({"Fact one": [], "Fact two": []})

    detail = evaluate_question(
        method="strategy_matrix_decompose",
        store=store,
        bm25=bm25,
        question=question,
        chroma_search_k=5,
        bm25_search_k=5,
        rrf_k=60,
        source_boost=0.15,
        k_values=[1],
        where=None,
        reranker_model=None,
        parent_texts={},
        reranker_candidate_k=20,
        reranker_batch_size=4,
    )

    assert detail["retrieved_parent_doc_ids"] == ["doc_a"]
    assert detail["retrieved_chunk_ids"] == ["chunk_a"]
    assert detail["evidence_coverage"] == 1.0


def test_evaluate_question_uses_post_rerank_order_for_retrieved_chunks_and_evidence_coverage():
    question = Question(
        question_id="q1",
        question_type="semantic",
        source_types=["policy"],
        question="What is the policy?",
        expected_doc_ids=["doc_b"],
        gold_answer="",
        answer_facts=[],
        required_evidence_groups=[["chunk_b"]],
    )
    store = FakeStore(
        {
            "What is the policy?": [
                (make_document("chunk_a", "doc_a", text="parent text a"), 0.1),
                (make_document("chunk_b", "doc_b", text="parent text b"), 0.2),
            ]
        }
    )
    bm25 = FakeBM25({"What is the policy?": []})
    reranker = FakeReranker({"parent text a": 0.1, "parent text b": 0.9})

    detail = evaluate_question(
        method="chroma_bm25_rrf_reranker",
        store=store,
        bm25=bm25,
        question=question,
        chroma_search_k=5,
        bm25_search_k=5,
        rrf_k=60,
        source_boost=0.15,
        k_values=[1],
        where=None,
        reranker_model=reranker,
        parent_texts={"parent_chunk_a": "parent text a", "parent_chunk_b": "parent text b"},
        reranker_candidate_k=20,
        reranker_batch_size=4,
    )

    assert detail["reranker_used"] is True
    assert detail["retrieved_parent_doc_ids"][:1] == ["doc_b"]
    assert detail["retrieved_chunk_ids"][:1] == ["chunk_b"]
    assert detail["evidence_coverage"] == 1.0


def test_post_rerank_chunk_order_keeps_parent_representatives_before_sibling_chunks():
    question = Question(
        question_id="q1",
        question_type="semantic",
        source_types=["policy"],
        question="What is the policy?",
        expected_doc_ids=["doc_b"],
        gold_answer="",
        answer_facts=[],
        required_evidence_groups=[["chunk_b"]],
    )
    store = FakeStore(
        {
            "What is the policy?": [
                (make_document("chunk_a1", "doc_a", text="parent text a"), 0.1),
                (make_document("chunk_a2", "doc_a", text="parent text a sibling"), 0.2),
                (make_document("chunk_b", "doc_b", text="parent text b"), 0.3),
            ]
        }
    )
    bm25 = FakeBM25({"What is the policy?": []})
    reranker = FakeReranker({"parent text a": 0.9, "parent text b": 0.8})

    detail = evaluate_question(
        method="chroma_bm25_rrf_reranker",
        store=store,
        bm25=bm25,
        question=question,
        chroma_search_k=5,
        bm25_search_k=5,
        rrf_k=60,
        source_boost=0.15,
        k_values=[2, 3],
        where=None,
        reranker_model=reranker,
        parent_texts={"parent_chunk_a1": "parent text a", "parent_chunk_b": "parent text b"},
        reranker_candidate_k=20,
        reranker_batch_size=4,
    )

    assert detail["retrieved_chunk_ids"] == ["chunk_a1", "chunk_b", "chunk_a2"]
    assert detail["evidence_coverage@2"] == 1.0


def test_evidence_coverage_uses_chunk_level_ranking_before_parent_deduplication():
    question = Question(
        question_id="q1",
        question_type="lookup",
        source_types=["policy"],
        question="What is the policy?",
        expected_doc_ids=["doc_a"],
        gold_answer="",
        answer_facts=[],
        required_evidence_groups=[["chunk_a"], ["chunk_b"]],
    )
    store = FakeStore(
        {
            "What is the policy?": [
                (make_document("chunk_a", "doc_a"), 0.1),
                (make_document("chunk_b", "doc_a"), 0.2),
            ]
        }
    )

    detail = evaluate_question(
        method="chroma_only",
        store=store,
        bm25=None,
        question=question,
        chroma_search_k=5,
        bm25_search_k=5,
        rrf_k=60,
        source_boost=0.15,
        k_values=[1, 2],
        where=None,
        reranker_model=None,
        parent_texts={},
        reranker_candidate_k=20,
        reranker_batch_size=4,
    )

    assert detail["retrieved_parent_doc_ids"] == ["doc_a"]
    assert detail["retrieved_chunk_ids"] == ["chunk_a", "chunk_b"]
    assert detail["evidence_coverage@1"] == 0.5
    assert detail["evidence_coverage@2"] == 1.0
    assert detail["evidence_coverage"] == 1.0


def test_strategy_matrix_decompose_preserves_single_query_behavior_for_simple_questions():
    question = Question(
        question_id="q1",
        question_type="lookup",
        source_types=["policy"],
        question="What is the policy?",
        expected_doc_ids=["doc_a"],
        gold_answer="",
        answer_facts=["Fact one"],
        required_evidence_groups=[["chunk_a"]],
    )
    store = FakeStore({"What is the policy?": [(make_document("chunk_a", "doc_a"), 0.1)]})
    bm25 = FakeBM25({"What is the policy?": [make_bm25_candidate("chunk_a", "doc_a")]})

    detail = evaluate_question(
        method="strategy_matrix_decompose",
        store=store,
        bm25=bm25,
        question=question,
        chroma_search_k=5,
        bm25_search_k=5,
        rrf_k=60,
        source_boost=0.15,
        k_values=[1],
        where=None,
        reranker_model=None,
        parent_texts={},
        reranker_candidate_k=20,
        reranker_batch_size=4,
    )

    assert store.queries == ["What is the policy?"]
    assert bm25.queries == ["What is the policy?"]
    assert detail["retrieved_chunk_ids"] == ["chunk_a"]
    assert detail["evidence_coverage"] == 1.0


def test_summarize_averages_evidence_coverage_only_for_annotated_questions():
    details = [
        {
            "latency_ms": 10.0,
            "evidence_coverage": 1.0,
            "evidence_coverage@1": 0.5,
            "evidence_coverage@2": 1.0,
            "required_evidence_groups_count": 1,
            "hit@1": 1,
            "precision@1": 1.0,
            "recall@1": 1.0,
            "f1@1": 1.0,
            "hit@2": 1,
            "precision@2": 0.5,
            "recall@2": 1.0,
            "f1@2": 2 / 3,
            "rr@2": 1.0,
        },
        {
            "latency_ms": 20.0,
            "evidence_coverage": 0.0,
            "evidence_coverage@1": 0.0,
            "evidence_coverage@2": 0.0,
            "required_evidence_groups_count": 0,
            "hit@1": 0,
            "precision@1": 0.0,
            "recall@1": 0.0,
            "f1@1": 0.0,
            "hit@2": 0,
            "precision@2": 0.0,
            "recall@2": 0.0,
            "f1@2": 0.0,
            "rr@2": 0.0,
        },
    ]

    summary = summarize(details, [1, 2])

    assert summary["questions"] == 2
    assert summary["evidence_coverage_questions"] == 1
    assert summary["evidence_coverage@1"] == 0.5
    assert summary["evidence_coverage@2"] == 1.0
    assert summary["evidence_coverage"] == 1.0


def test_write_outputs_includes_per_k_evidence_coverage_in_details_csv(tmp_path):
    details = [
        {
            "method": "strategy_matrix_decompose",
            "normalized_method": "strategy_matrix_decompose",
            "question_id": "q1",
            "question_type": "multi_hop",
            "source_types": ["policy"],
            "question": "What policies apply?",
            "expected_doc_ids": ["doc_a"],
            "retrieved_parent_doc_ids": ["doc_a"],
            "required_evidence_groups_count": 1,
            "evidence_coverage": 1.0,
            "evidence_coverage@1": 0.5,
            "evidence_coverage@2": 1.0,
            "latency_ms": 10.0,
            "vector_child_results": 1,
            "bm25_child_results": 1,
            "fused_child_results": 1,
            "dedup_parent_results": 1,
            "source_boost_applied": True,
            "reranker_used": False,
            "top_hits": [],
            "hit@1": 1,
            "precision@1": 1.0,
            "recall@1": 1.0,
            "f1@1": 1.0,
            "matched_doc_ids@1": ["doc_a"],
            "hit@2": 1,
            "precision@2": 0.5,
            "recall@2": 1.0,
            "f1@2": 2 / 3,
            "matched_doc_ids@2": ["doc_a"],
            "rr@2": 1.0,
        }
    ]

    write_outputs(
        output_dir=tmp_path,
        method="strategy_matrix_decompose",
        summary={"questions": 1},
        details=details,
        k_values=[1, 2],
    )

    csv_path = tmp_path / "strategy_matrix_decompose_details.csv"
    with csv_path.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))

    assert row["required_evidence_groups_count"] == "1"
    assert row["evidence_coverage@1"] == "0.5"
    assert row["evidence_coverage@2"] == "1.0"
