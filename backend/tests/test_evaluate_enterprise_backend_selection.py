import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_enterprise_hybrid_retrieval import parse_args


def test_parse_args_defaults_to_chroma_backend(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_enterprise_hybrid_retrieval.py", "--limit", "1"])

    args = parse_args()

    assert args.backend == "chroma"


def test_parse_args_accepts_elasticsearch_backend(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_enterprise_hybrid_retrieval.py", "--backend", "elasticsearch", "--method", "chroma_bm25_rrf", "--limit", "1"],
    )

    args = parse_args()

    assert args.backend == "elasticsearch"
    assert args.method == "chroma_bm25_rrf"


def test_candidate_from_backend_document_preserves_parent_ids_and_text():
    from scripts.evaluate_enterprise_hybrid_retrieval import candidate_from_backend_document

    candidate = candidate_from_backend_document(
        {
            "parent_doc_id": "doc-1",
            "parent_chunk_id": "parent-1",
            "source_type": "policy",
            "title": "Policy",
            "section_heading": "Section",
            "child_text": "child text",
            "parent_text": "parent text",
            "score": 0.7,
            "metadata": {"chunk_id": "child-1"},
        }
    )

    assert candidate.parent_doc_id == "doc-1"
    assert candidate.parent_chunk_id == "parent-1"
    assert candidate.chunk_id == "child-1"
    assert candidate.text == "child text"
    assert candidate.evidence_chunk_ids == ["child-1"]


class FakeRetrievalBackend:
    def __init__(self):
        self.calls = []

    async def retrieve_with_details(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "dense_results": [],
            "bm25_results": [],
            "fused_results": [],
            "reranked_results": [],
            "selected_documents": [
                {
                    "parent_doc_id": "doc-1",
                    "parent_chunk_id": "parent-1",
                    "source_type": "policy",
                    "title": "Policy",
                    "section_heading": "Section",
                    "child_text": "child text",
                    "parent_text": "parent text",
                    "score": 0.7,
                    "metadata": {"chunk_id": "child-1"},
                }
            ],
            "metrics": {},
        }


def make_question():
    from scripts.evaluate_enterprise_hybrid_retrieval import Question

    return Question(
        question_id="q1",
        question_type="semantic",
        source_types=["policy"],
        question="policy?",
        expected_doc_ids=["doc-1"],
        gold_answer="answer",
        answer_facts=[],
    )


def test_elasticsearch_evaluation_branch_respects_bm25_only_method():
    from scripts.evaluate_enterprise_hybrid_retrieval import evaluate_question

    backend = FakeRetrievalBackend()
    detail = evaluate_question(
        method="bm25_only",
        store=None,
        bm25=None,
        question=make_question(),
        chroma_search_k=50,
        bm25_search_k=25,
        rrf_k=60,
        source_boost=0.15,
        k_values=[1, 5],
        where=None,
        reranker_model=None,
        parent_texts={},
        reranker_candidate_k=20,
        reranker_batch_size=4,
        backend="elasticsearch",
        retrieval_backend=backend,
    )

    assert backend.calls[0]["dense_top_k"] == 0
    assert backend.calls[0]["bm25_top_k"] == 25
    assert backend.calls[0]["source_hints"] is None
    assert detail["retrieved_parent_doc_ids"] == ["doc-1"]


def test_elasticsearch_evaluation_branch_passes_source_hints_for_boost_method():
    from scripts.evaluate_enterprise_hybrid_retrieval import evaluate_question

    backend = FakeRetrievalBackend()
    evaluate_question(
        method="chroma_bm25_rrf_source_boost",
        store=None,
        bm25=None,
        question=make_question(),
        chroma_search_k=50,
        bm25_search_k=25,
        rrf_k=60,
        source_boost=0.15,
        k_values=[1, 5],
        where=None,
        reranker_model=None,
        parent_texts={},
        reranker_candidate_k=20,
        reranker_batch_size=4,
        backend="elasticsearch",
        retrieval_backend=backend,
    )

    assert backend.calls[0]["dense_top_k"] == 50
    assert backend.calls[0]["bm25_top_k"] == 25
    assert backend.calls[0]["source_hints"] == ["policy"]
