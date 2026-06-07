import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.evaluate_enterprise_hybrid_retrieval as retrieval_eval
from scripts.evaluate_enterprise_hybrid_retrieval import (
    DEFAULT_GRAPH_DIR,
    Candidate,
    Question,
    classify_failures,
    decompose_question_for_eval,
    evaluate_question,
    evidence_coverage_at_k,
    load_questions,
    maybe_load_graph_index,
    method_needs_graph,
    method_needs_reranker,
    normalize_method,
    parse_args,
    should_rerank_question,
    summarize,
    validate_graph_index_loaded,
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


class FakeGraphIndex:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def retrieve_sync(self, query, top_k, depth, source_hints=None):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "depth": depth,
                "source_hints": source_hints,
            }
        )
        return self.results_by_query.get(query, [])[:top_k]


class FakeReranker:
    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text
        self.pairs = []

    def predict(self, pairs, batch_size=1):
        self.pairs.extend(pairs)
        return [self.scores_by_text.get(document, 0.0) for _query, document in pairs]


class FakeLoadedGraphIndex:
    def __init__(self, parent_chunks=None, entities=None):
        self.parent_chunks = parent_chunks or {}
        self.entities = entities or {}



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


def make_graph_result(parent_chunk_id, parent_doc_id="doc", source_type="policy", score=2.0):
    from app.rag.graph_index_service import GraphRetrievedDocument

    return GraphRetrievedDocument(
        parent_chunk_id=parent_chunk_id,
        parent_doc_id=parent_doc_id,
        source_type=source_type,
        title="Graph Title",
        section_heading="Graph Section",
        text="graph text",
        score=score,
        matched_entities=["policy"],
        matched_relations=["policy__requires__approval"],
        reason="graph_entity_relation_match",
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


def test_graph_method_metadata_requires_graph_but_not_reranker():
    assert normalize_method("chroma_bm25_graph_rrf") == "chroma_bm25_graph_rrf"
    assert method_needs_graph("chroma_bm25_graph_rrf") is True
    assert method_needs_reranker("chroma_bm25_graph_rrf") is False



def test_parse_args_defaults_graph_configuration(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_enterprise_hybrid_retrieval.py"])

    args = parse_args()

    assert args.graph_dir == DEFAULT_GRAPH_DIR
    assert args.graph_search_k == 40
    assert args.graph_depth == 2


@pytest.mark.parametrize("graph_search_k", [0, -1])
def test_graph_method_rejects_invalid_graph_search_k_before_loading_questions(monkeypatch, graph_search_k):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_enterprise_hybrid_retrieval.py",
            "--method",
            "chroma_bm25_graph_rrf",
            "--graph-search-k",
            str(graph_search_k),
        ],
    )
    monkeypatch.setattr(
        retrieval_eval,
        "load_questions",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_questions should not run")),
    )

    with pytest.raises(ValueError, match="--graph-search-k must be greater than 0"):
        retrieval_eval.main()


@pytest.mark.parametrize("graph_depth", [-1, -2])
def test_graph_method_rejects_negative_graph_depth_before_loading_questions(monkeypatch, graph_depth):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_enterprise_hybrid_retrieval.py",
            "--method",
            "chroma_bm25_graph_rrf",
            "--graph-depth",
            str(graph_depth),
        ],
    )
    monkeypatch.setattr(
        retrieval_eval,
        "load_questions",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_questions should not run")),
    )

    with pytest.raises(ValueError, match="--graph-depth must be greater than or equal to 0"):
        retrieval_eval.main()



def test_maybe_load_graph_index_returns_none_for_non_graph_methods(tmp_path):
    assert maybe_load_graph_index("chroma_bm25_rrf", tmp_path / "missing_graph") is None



def test_maybe_load_graph_index_loads_non_empty_graph_data(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "entities.jsonl").write_text(
        json.dumps(
            {
                "entity_id": "leave_approval",
                "name": "leave approval",
                "normalized_name": "leave approval",
                "entity_type": "policy",
                "source_chunk_ids": ["parent_a"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (graph_dir / "relations.jsonl").write_text("", encoding="utf-8")
    (graph_dir / "entity_chunk_map.json").write_text(
        json.dumps({"leave_approval": ["parent_a"]}),
        encoding="utf-8",
    )
    (graph_dir / "parent_chunks.json").write_text(
        json.dumps({"parent_a": {"parent_doc_id": "doc_a", "text": "Leave approval text"}}),
        encoding="utf-8",
    )

    graph_index = maybe_load_graph_index("chroma_bm25_graph_rrf", graph_dir)

    assert graph_index is not None
    assert "leave_approval" in graph_index.entities
    assert "parent_a" in graph_index.parent_chunks


def test_validate_graph_index_loaded_fails_for_missing_graph_dir(tmp_path):
    graph_dir = tmp_path / "missing_graph"
    graph_index = FakeLoadedGraphIndex(parent_chunks={"parent_a": object()}, entities={"entity_a": object()})

    with pytest.raises(ValueError, match="Graph index directory does not exist"):
        validate_graph_index_loaded(graph_dir, graph_index)


@pytest.mark.parametrize(
    ("parent_chunks", "entities"),
    [
        ({}, {"entity_a": object()}),
        ({"parent_a": object()}, {}),
    ],
)
def test_validate_graph_index_loaded_fails_for_empty_graph_data(tmp_path, parent_chunks, entities):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    graph_index = FakeLoadedGraphIndex(parent_chunks=parent_chunks, entities=entities)

    with pytest.raises(ValueError, match="No graph data loaded"):
        validate_graph_index_loaded(graph_dir, graph_index)



def test_validate_graph_index_loaded_accepts_non_empty_graph_data(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    graph_index = FakeLoadedGraphIndex(parent_chunks={"parent_a": object()}, entities={"entity_a": object()})

    validate_graph_index_loaded(graph_dir, graph_index)



def test_graph_method_raises_when_graph_index_is_missing():
    question = Question(
        question_id="q_graph_missing",
        question_type="multi_hop",
        source_types=[],
        question="How does leave approval work?",
        expected_doc_ids=[],
        gold_answer="",
        answer_facts=[],
        required_evidence_groups=[],
    )

    try:
        evaluate_question(
            method="chroma_bm25_graph_rrf",
            store=FakeStore({"How does leave approval work?": []}),
            bm25=FakeBM25({"How does leave approval work?": []}),
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
            graph_index=None,
            graph_search_k=5,
            graph_depth=2,
        )
    except ValueError as exc:
        assert "requires graph index" in str(exc)
    else:
        raise AssertionError("Expected graph method to require graph_index")



def test_graph_rrf_fuses_vector_child_and_graph_parent_for_same_parent_chunk():
    question = Question(
        question_id="q_graph_parent_fusion",
        question_type="multi_hop",
        source_types=["policy"],
        question="How does probation leave approval work?",
        expected_doc_ids=["doc_graph"],
        gold_answer="",
        answer_facts=[],
        required_evidence_groups=[["child_graph"], ["parent_graph"]],
    )
    vector_document = SimpleNamespace(
        metadata={
            "chunk_id": "child_graph",
            "parent_doc_id": "doc_graph",
            "parent_chunk_id": "parent_graph",
            "source_type": "policy",
            "title": "Vector Title",
            "section_heading": "Vector Section",
        },
        page_content="vector child text",
    )
    store = FakeStore({"How does probation leave approval work?": [(vector_document, 0.1)]})
    bm25 = FakeBM25({"How does probation leave approval work?": []})
    graph = FakeGraphIndex(
        {
            "How does probation leave approval work?": [
                make_graph_result("parent_graph", parent_doc_id="doc_graph", score=3.0)
            ]
        }
    )

    detail = evaluate_question(
        method="chroma_bm25_graph_rrf",
        store=store,
        bm25=bm25,
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
        graph_index=graph,
        graph_search_k=5,
        graph_depth=2,
    )

    top_hit = detail["top_hits"][0]
    assert detail["fused_child_results"] == 1
    assert detail["retrieved_parent_doc_ids"] == ["doc_graph"]
    assert detail["retrieved_chunk_ids"] == ["child_graph"]
    assert top_hit["chunk_id"] == "child_graph"
    assert top_hit["parent_chunk_id"] == "parent_graph"
    assert top_hit["vector_rank"] == 1
    assert top_hit["graph_rank"] == 1
    assert top_hit["graph_score"] == 3.0
    assert detail["evidence_coverage@1"] == 1.0
    assert detail["evidence_coverage"] == 1.0



def test_graph_rrf_preserves_sibling_vector_children_when_graph_matches_parent_chunk():
    question = Question(
        question_id="q_graph_sibling_child_fusion",
        question_type="multi_hop",
        source_types=["policy"],
        question="How do related policy details fit together?",
        expected_doc_ids=["doc_graph"],
        gold_answer="",
        answer_facts=[],
        required_evidence_groups=[["child_graph_a"], ["child_graph_b"]],
    )
    vector_documents = [
        SimpleNamespace(
            metadata={
                "chunk_id": "child_graph_a",
                "parent_doc_id": "doc_graph",
                "parent_chunk_id": "parent_graph",
                "source_type": "policy",
                "title": "Vector Title",
                "section_heading": "Vector Section A",
            },
            page_content="first vector child text",
        ),
        SimpleNamespace(
            metadata={
                "chunk_id": "child_graph_b",
                "parent_doc_id": "doc_graph",
                "parent_chunk_id": "parent_graph",
                "source_type": "policy",
                "title": "Vector Title",
                "section_heading": "Vector Section B",
            },
            page_content="second vector child text",
        ),
    ]
    store = FakeStore(
        {"How do related policy details fit together?": [(vector_documents[0], 0.1), (vector_documents[1], 0.2)]}
    )
    bm25 = FakeBM25({"How do related policy details fit together?": []})
    graph = FakeGraphIndex(
        {
            "How do related policy details fit together?": [
                make_graph_result("parent_graph", parent_doc_id="doc_graph", score=3.0)
            ]
        }
    )

    detail = evaluate_question(
        method="chroma_bm25_graph_rrf",
        store=store,
        bm25=bm25,
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
        graph_index=graph,
        graph_search_k=5,
        graph_depth=2,
    )

    assert detail["retrieved_chunk_ids"] == ["child_graph_a", "child_graph_b"]
    assert detail["fused_child_results"] == 2
    assert detail["evidence_coverage@2"] == 1.0
    assert detail["evidence_coverage"] == 1.0
    assert detail["top_hits"][0]["graph_rank"] == 1
    assert detail["top_hits"][0]["graph_score"] == 3.0


def test_graph_rrf_includes_graph_candidates_in_fusion():
    question = Question(
        question_id="q_graph",
        question_type="multi_hop",
        source_types=["policy"],
        question="How does probation leave approval work?",
        expected_doc_ids=["doc_graph"],
        gold_answer="",
        answer_facts=[],
        required_evidence_groups=[["parent_graph"]],
    )
    store = FakeStore({"How does probation leave approval work?": []})
    bm25 = FakeBM25({"How does probation leave approval work?": []})
    graph = FakeGraphIndex(
        {
            "How does probation leave approval work?": [
                make_graph_result("parent_graph", parent_doc_id="doc_graph", score=3.0)
            ]
        }
    )

    detail = evaluate_question(
        method="chroma_bm25_graph_rrf",
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
        graph_index=graph,
        graph_search_k=5,
        graph_depth=2,
    )

    assert graph.calls == [
        {
            "query": "How does probation leave approval work?",
            "top_k": 5,
            "depth": 2,
            "source_hints": [],
        }
    ]
    assert detail["graph_results"] == 1
    assert detail["retrieved_parent_doc_ids"] == ["doc_graph"]
    assert detail["retrieved_chunk_ids"] == ["parent_graph"]
    assert detail["hit@1"] == 1


def test_strategy_matrix_graph_passes_source_hints_when_source_boost_applies():
    question = Question(
        question_id="q_strategy_graph",
        question_type="semantic",
        source_types=["policy"],
        question="What policy applies?",
        expected_doc_ids=["doc_graph"],
        gold_answer="",
        answer_facts=[],
        required_evidence_groups=[],
    )
    store = FakeStore({"What policy applies?": []})
    bm25 = FakeBM25({"What policy applies?": []})
    graph = FakeGraphIndex(
        {"What policy applies?": [make_graph_result("parent_graph", parent_doc_id="doc_graph", score=3.0)]}
    )

    detail = evaluate_question(
        method="strategy_matrix_graph",
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
        graph_index=graph,
        graph_search_k=5,
        graph_depth=2,
    )

    assert graph.calls[0]["source_hints"] == ["policy"]
    assert detail["source_boost_applied"] is True



def test_strategy_matrix_decompose_graph_runs_graph_for_each_sub_query():
    question = Question(
        question_id="q_graph_decompose",
        question_type="comparison",
        source_types=["policy"],
        question="Compare leave policies",
        expected_doc_ids=["doc_a", "doc_b"],
        gold_answer="",
        answer_facts=["Probation leave", "Regular leave"],
        required_evidence_groups=[["parent_a"], ["parent_b"]],
    )
    store = FakeStore({"Probation leave": [], "Regular leave": []})
    bm25 = FakeBM25({"Probation leave": [], "Regular leave": []})
    graph = FakeGraphIndex(
        {
            "Probation leave": [make_graph_result("parent_a", parent_doc_id="doc_a", score=3.0)],
            "Regular leave": [make_graph_result("parent_b", parent_doc_id="doc_b", score=2.5)],
        }
    )

    detail = evaluate_question(
        method="strategy_matrix_decompose_graph",
        store=store,
        bm25=bm25,
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
        graph_index=graph,
        graph_search_k=5,
        graph_depth=2,
    )

    assert [call["query"] for call in graph.calls] == ["Probation leave", "Regular leave"]
    assert [call["source_hints"] for call in graph.calls] == [["policy"], ["policy"]]
    assert detail["graph_results"] == 2
    assert detail["evidence_coverage@2"] == 1.0


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


def test_evaluate_question_includes_ndcg_and_average_precision_metrics():
    question = Question(
        question_id="q_metrics",
        question_type="lookup",
        source_types=["policy"],
        question="What is the policy?",
        expected_doc_ids=["doc_a", "doc_c"],
        gold_answer="",
        answer_facts=[],
        required_evidence_groups=[],
    )
    store = FakeStore(
        {
            "What is the policy?": [
                (make_document("chunk_a", "doc_a"), 0.1),
                (make_document("chunk_b", "doc_b"), 0.2),
                (make_document("chunk_c", "doc_c"), 0.3),
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
        k_values=[1, 3],
        where=None,
        reranker_model=None,
        parent_texts={},
        reranker_candidate_k=20,
        reranker_batch_size=4,
    )

    assert detail["ndcg@1"] == 1.0
    assert 0.0 < detail["ndcg@3"] < 1.0
    assert detail["ap@1"] == 1.0
    assert detail["ap@3"] == (1 / 1 + 2 / 3) / 2


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


def test_classify_failures_does_not_mark_top1_single_gold_hit_as_low_precision_at_large_k():
    row = {
        "expected_doc_ids": ["doc_a"],
        "dedup_parent_results": 20,
        "hit@1": 1,
        "hit@20": 1,
        "recall@20": 1.0,
        "precision@20": 0.05,
        "ndcg@20": 1.0,
        "ap@20": 1.0,
        "required_evidence_groups_count": 0,
        "reranker_used": False,
    }

    reasons = classify_failures(row, [1, 20])

    assert "low_precision" not in reasons


def test_classify_failures_returns_multiple_retrieval_failure_reasons():
    row = {
        "dedup_parent_results": 3,
        "hit@1": 0,
        "hit@10": 1,
        "recall@10": 0.25,
        "precision@10": 0.1,
        "ndcg@10": 0.2,
        "ap@10": 0.15,
        "evidence_coverage@10": 0.5,
        "required_evidence_groups_count": 2,
        "reranker_used": True,
    }

    reasons = classify_failures(row, [1, 10])

    assert reasons == [
        "low_recall",
        "low_precision",
        "low_ndcg",
        "low_map",
        "evidence_group_missing",
        "reranker_top1_miss",
    ]


def test_summarize_includes_question_type_summary_and_map():
    details = [
        {
            "question_type": "lookup",
            "latency_ms": 100.0,
            "hit@1": 1,
            "precision@1": 1.0,
            "recall@1": 0.5,
            "f1@1": 2 / 3,
            "ndcg@1": 1.0,
            "ap@1": 0.5,
            "evidence_coverage@1": 0.0,
            "rr@1": 1.0,
            "required_evidence_groups_count": 0,
        },
        {
            "question_type": "comparison",
            "latency_ms": 300.0,
            "hit@1": 0,
            "precision@1": 0.0,
            "recall@1": 0.0,
            "f1@1": 0.0,
            "ndcg@1": 0.0,
            "ap@1": 0.0,
            "evidence_coverage@1": 0.0,
            "rr@1": 0.0,
            "required_evidence_groups_count": 1,
        },
    ]

    summary = summarize(details, [1])

    assert summary["map@1"] == 0.25
    assert summary["question_type_summary"]["lookup"]["questions"] == 1
    assert summary["question_type_summary"]["lookup"]["map@1"] == 0.5
    assert summary["question_type_summary"]["comparison"]["questions"] == 1


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
