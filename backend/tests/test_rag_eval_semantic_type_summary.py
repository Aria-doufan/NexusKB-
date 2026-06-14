import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_enterprise_hybrid_retrieval import (
    Candidate,
    build_doc_semantic_type_summary,
    fuse_by_rrf,
    graph_search,
    load_doc_semantic_types_by_doc_id,
)
from scripts.rag_eval_metrics import build_group_summary


def _detail(expected_doc_semantic_types, hit):
    return {
        "expected_doc_semantic_types": expected_doc_semantic_types,
        "latency_ms": 100.0 if hit else 300.0,
        "hit@1": 1 if hit else 0,
        "precision@1": 1.0 if hit else 0.0,
        "recall@1": 1.0 if hit else 0.0,
        "f1@1": 1.0 if hit else 0.0,
        "ndcg@1": 1.0 if hit else 0.0,
        "ap@1": 1.0 if hit else 0.0,
        "evidence_coverage@1": 1.0 if hit else 0.0,
        "required_evidence_groups_count": 1,
        "rr@1": 1.0 if hit else 0.0,
    }


def test_build_group_summary_groups_multi_value_expected_semantic_types():
    summary = build_group_summary(
        [
            _detail(["policy_doc", "procedure_doc"], True),
            _detail(["policy_doc"], False),
        ],
        [1],
        "expected_doc_semantic_types",
    )

    assert summary["policy_doc"]["questions"] == 2
    assert summary["policy_doc"]["hit@1"] == 0.5
    assert summary["policy_doc"]["average_latency_ms"] == 200.0
    assert summary["procedure_doc"]["questions"] == 1
    assert summary["procedure_doc"]["hit@1"] == 1.0


def test_build_doc_semantic_type_summary_uses_type_specific_expected_sets():
    detail = {
        "expected_doc_ids": ["doc-policy", "doc-procedure"],
        "expected_doc_semantic_type_by_doc_id": {
            "doc-policy": "policy_doc",
            "doc-procedure": "procedure_doc",
        },
        "retrieved_parent_doc_ids": ["doc-policy"],
        "latency_ms": 123.0,
    }

    summary = build_doc_semantic_type_summary([detail], [1])

    assert summary["policy_doc"]["questions"] == 1
    assert summary["policy_doc"]["hit@1"] == 1.0
    assert summary["policy_doc"]["precision@1"] == 1.0
    assert summary["policy_doc"]["recall@1"] == 1.0
    assert summary["procedure_doc"]["questions"] == 1
    assert summary["procedure_doc"]["hit@1"] == 0.0
    assert summary["procedure_doc"]["precision@1"] == 0.0
    assert summary["procedure_doc"]["recall@1"] == 0.0


def test_load_doc_semantic_types_by_doc_id_maps_parent_doc_id_to_semantic_type(tmp_path):
    parent_chunks_path = tmp_path / "parent_chunks.jsonl"
    rows = [
        {"parent_doc_id": "doc-policy", "doc_semantic_type": "policy_doc"},
        {"parent_doc_id": "doc-generic"},
    ]
    parent_chunks_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    mapping = load_doc_semantic_types_by_doc_id(parent_chunks_path)

    assert mapping == {
        "doc-policy": "policy_doc",
        "doc-generic": "generic_doc",
    }


class _GraphResult:
    parent_chunk_id = "chunk-policy"
    parent_doc_id = "doc-policy"
    source_type = "hr"
    title = "Policy"
    section_heading = "Eligibility"
    text = "Policy text"
    score = 0.75
    metadata = {"doc_semantic_type": "policy_doc"}


class _GraphIndex:
    def retrieve_sync(self, **kwargs):
        return [_GraphResult()]


def test_graph_search_reads_doc_semantic_type_from_result_metadata():
    candidates = graph_search(_GraphIndex(), "policy question", 1, 2, [])

    assert candidates[0].doc_semantic_type == "policy_doc"


def test_fuse_by_rrf_preserves_graph_semantic_type_on_generic_merge():
    vector_candidate = Candidate(
        chunk_id="child-policy",
        parent_doc_id="doc-policy",
        parent_chunk_id="parent-policy",
        source_type="hr",
        doc_semantic_type="generic_doc",
        vector_rank=1,
        vector_score=0.9,
    )
    graph_candidate = Candidate(
        chunk_id="graph-evidence-policy",
        parent_doc_id="doc-policy",
        parent_chunk_id="parent-policy",
        source_type="hr",
        doc_semantic_type="policy_doc",
        graph_rank=1,
        graph_score=0.8,
    )

    fused = fuse_by_rrf(
        [vector_candidate],
        [],
        rrf_k=60,
        graph_candidates=[graph_candidate],
    )

    assert len(fused) == 1
    assert fused[0].doc_semantic_type == "policy_doc"
