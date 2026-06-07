import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.graph_index_service import (
    EntityNode,
    GraphIndexService,
    GraphRetrievedDocument,
    RelationEdge,
    normalize_entity_name,
)


def test_normalize_entity_name_collapses_case_punctuation_and_whitespace():
    assert normalize_entity_name("  Manager Approval!!! ") == "manager approval"
    assert normalize_entity_name("Probation\tLeave\nPolicy") == "probation leave policy"


def test_graph_index_save_and_load_round_trip(tmp_path):
    graph_dir = tmp_path / "graph"
    service = GraphIndexService(graph_dir=graph_dir)
    service.entities = {
        "policy": EntityNode(
            entity_id="policy",
            name="Probation Leave Policy",
            normalized_name="probation leave policy",
            entity_type="policy",
            description="Leave policy for probation employees.",
            source_chunk_ids=["parent_a"],
            aliases=["Probation Leave Policy"],
        )
    }
    service.relations = [
        RelationEdge(
            relation_id="policy__requires__approval",
            src_entity_id="policy",
            tgt_entity_id="approval",
            relation_type="requires",
            description="Policy requires manager approval.",
            keywords=["leave", "approval"],
            weight=1.0,
            source_chunk_ids=["parent_a"],
        )
    ]
    service.entity_chunk_map = {"policy": ["parent_a"]}
    service.parent_chunks = {
        "parent_a": {
            "parent_chunk_id": "parent_a",
            "parent_doc_id": "doc_a",
            "title": "Policy A",
            "source_type": "policy",
            "section_heading": "Leave",
            "text": "Employees request manager approval.",
        }
    }

    service.save_sync()

    loaded = GraphIndexService(graph_dir=graph_dir)
    loaded.load_sync()

    assert loaded.entities["policy"].name == "Probation Leave Policy"
    assert loaded.relations[0].keywords == ["leave", "approval"]
    assert loaded.entity_chunk_map == {"policy": ["parent_a"]}
    assert loaded.parent_chunks["parent_a"]["parent_doc_id"] == "doc_a"


def test_graph_retrieve_scores_entity_and_relation_matches(tmp_path):
    service = GraphIndexService(graph_dir=tmp_path / "graph")
    service.entities = {
        "policy": EntityNode(
            entity_id="policy",
            name="Probation Leave Policy",
            normalized_name="probation leave policy",
            entity_type="policy",
            description="Leave policy.",
            source_chunk_ids=["parent_a"],
            aliases=["Probation Leave Policy"],
        ),
        "approval": EntityNode(
            entity_id="approval",
            name="Manager Approval",
            normalized_name="manager approval",
            entity_type="process",
            description="Approval by manager.",
            source_chunk_ids=["parent_b"],
            aliases=["Manager Approval"],
        ),
    }
    service.relations = [
        RelationEdge(
            relation_id="policy__requires__approval",
            src_entity_id="policy",
            tgt_entity_id="approval",
            relation_type="requires",
            description="Probation leave requires manager approval.",
            keywords=["leave", "approval", "manager"],
            weight=1.0,
            source_chunk_ids=["parent_a", "parent_b"],
        )
    ]
    service.entity_chunk_map = {"policy": ["parent_a"], "approval": ["parent_b"]}
    service.parent_chunks = {
        "parent_a": {"parent_chunk_id": "parent_a", "parent_doc_id": "doc_a", "source_type": "policy"},
        "parent_b": {"parent_chunk_id": "parent_b", "parent_doc_id": "doc_b", "source_type": "policy"},
    }

    results = service.retrieve_sync(
        query="How does probation leave policy manager approval work?",
        top_k=5,
        depth=1,
        source_hints=["policy"],
    )

    assert [result.parent_chunk_id for result in results] == ["parent_a", "parent_b"]
    assert results[0].score > results[1].score
    assert "policy" in results[0].matched_entities
    assert "policy__requires__approval" in results[0].matched_relations


def test_graph_retrieve_ignores_keyword_relation_outside_entity_frontier(tmp_path):
    service = GraphIndexService(graph_dir=tmp_path / "graph")
    service.entities = {
        "alpha_policy": EntityNode(
            entity_id="alpha_policy",
            name="Alpha Policy",
            normalized_name="alpha policy",
            entity_type="policy",
            description="Alpha policy.",
            source_chunk_ids=["parent_alpha"],
            aliases=[],
        ),
        "beta": EntityNode(
            entity_id="beta",
            name="Beta",
            normalized_name="beta",
            entity_type="topic",
            description="Beta topic.",
            source_chunk_ids=["parent_beta"],
            aliases=[],
        ),
        "gamma": EntityNode(
            entity_id="gamma",
            name="Gamma",
            normalized_name="gamma",
            entity_type="topic",
            description="Gamma topic.",
            source_chunk_ids=["parent_gamma"],
            aliases=[],
        ),
    }
    service.relations = [
        RelationEdge(
            relation_id="beta__requires__gamma",
            src_entity_id="beta",
            tgt_entity_id="gamma",
            relation_type="requires",
            description="Beta requires keyword approval.",
            keywords=["approval"],
            weight=1.0,
            source_chunk_ids=["parent_beta_gamma"],
        )
    ]
    service.entity_chunk_map = {"alpha_policy": ["parent_alpha"], "beta": ["parent_beta"], "gamma": ["parent_gamma"]}
    service.parent_chunks = {
        "parent_alpha": {"parent_chunk_id": "parent_alpha", "parent_doc_id": "doc_alpha", "source_type": "policy"},
        "parent_beta": {"parent_chunk_id": "parent_beta", "parent_doc_id": "doc_beta", "source_type": "policy"},
        "parent_gamma": {"parent_chunk_id": "parent_gamma", "parent_doc_id": "doc_gamma", "source_type": "policy"},
        "parent_beta_gamma": {"parent_chunk_id": "parent_beta_gamma", "parent_doc_id": "doc_beta_gamma", "source_type": "policy"},
    }

    results = service.retrieve_sync("alpha policy keyword approval", top_k=5, depth=1)

    assert [result.parent_chunk_id for result in results] == ["parent_alpha"]


def test_depth_two_traversal_scores_each_relation_once(tmp_path):
    service = GraphIndexService(graph_dir=tmp_path / "graph")
    service.entities = {
        "a": EntityNode(
            entity_id="a",
            name="Alpha Policy",
            normalized_name="alpha policy",
            entity_type="policy",
            description="Alpha policy.",
            source_chunk_ids=["parent_a"],
            aliases=[],
        ),
        "b": EntityNode(
            entity_id="b",
            name="Beta Process",
            normalized_name="beta process",
            entity_type="process",
            description="Beta process.",
            source_chunk_ids=["parent_b"],
            aliases=[],
        ),
        "c": EntityNode(
            entity_id="c",
            name="Gamma Control",
            normalized_name="gamma control",
            entity_type="control",
            description="Gamma control.",
            source_chunk_ids=["parent_c"],
            aliases=[],
        ),
    }
    service.relations = [
        RelationEdge(
            relation_id="a__relates__b",
            src_entity_id="a",
            tgt_entity_id="b",
            relation_type="links",
            description="",
            keywords=[],
            weight=1.0,
            source_chunk_ids=["parent_ab"],
        ),
        RelationEdge(
            relation_id="b__relates__c",
            src_entity_id="b",
            tgt_entity_id="c",
            relation_type="links",
            description="",
            keywords=[],
            weight=1.0,
            source_chunk_ids=["parent_bc"],
        ),
    ]
    service.entity_chunk_map = {"a": ["parent_a"], "b": ["parent_b"], "c": ["parent_c"]}
    service.parent_chunks = {
        "parent_a": {"parent_chunk_id": "parent_a", "parent_doc_id": "doc_a", "source_type": "policy"},
        "parent_ab": {"parent_chunk_id": "parent_ab", "parent_doc_id": "doc_ab", "source_type": "policy"},
        "parent_bc": {"parent_chunk_id": "parent_bc", "parent_doc_id": "doc_bc", "source_type": "policy"},
    }

    results = service.retrieve_sync("alpha policy", top_k=5, depth=2)

    scores_by_parent = {result.parent_chunk_id: result.score for result in results}
    assert scores_by_parent["parent_ab"] == pytest.approx(0.35)
    assert scores_by_parent["parent_bc"] == pytest.approx(0.35 / 2)



def test_graph_retrieve_does_not_match_entity_on_partial_token_overlap(tmp_path):
    service = GraphIndexService(graph_dir=tmp_path / "graph")
    service.entities = {
        "employee_category": EntityNode(
            entity_id="employee_category",
            name="Employee Category",
            normalized_name="employee category",
            entity_type="policy_term",
            description="Eligibility depends on employee grouping.",
            source_chunk_ids=["parent_employee_category"],
            aliases=[],
        )
    }
    service.entity_chunk_map = {"employee_category": ["parent_employee_category"]}
    service.parent_chunks = {
        "parent_employee_category": {
            "parent_chunk_id": "parent_employee_category",
            "parent_doc_id": "doc_employee_category",
            "source_type": "policy",
        }
    }

    results = service.retrieve_sync("category eligibility", top_k=5, depth=1)

    assert results == []


def test_graph_retrieve_does_not_match_entity_inside_query_token(tmp_path):
    service = GraphIndexService(graph_dir=tmp_path / "graph")
    service.entities = {
        "age": EntityNode(
            entity_id="age",
            name="Age",
            normalized_name="age",
            entity_type="policy_term",
            description="Employee age.",
            source_chunk_ids=["parent_age"],
            aliases=[],
        )
    }
    service.entity_chunk_map = {"age": ["parent_age"]}
    service.parent_chunks = {
        "parent_age": {
            "parent_chunk_id": "parent_age",
            "parent_doc_id": "doc_age",
            "source_type": "policy",
        }
    }

    results = service.retrieve_sync("manager approval", top_k=5, depth=1)

    assert results == []


def test_relation_match_score_normalizes_multi_token_keywords():
    relation = RelationEdge(
        relation_id="policy__requires__approval",
        src_entity_id="policy",
        tgt_entity_id="approval",
        relation_type="related_to",
        description="",
        keywords=["Manager Approval"],
        weight=1.0,
        source_chunk_ids=["parent_a"],
    )

    score = GraphIndexService._relation_match_score(relation, {"manager", "approval"})

    assert score == 2.0


def test_graph_retrieve_respects_top_k(tmp_path):
    service = GraphIndexService(graph_dir=tmp_path / "graph")
    service.entities = {
        "a": EntityNode(
            entity_id="a",
            name="Alpha Policy",
            normalized_name="alpha policy",
            entity_type="policy",
            description="Alpha.",
            source_chunk_ids=["parent_a", "parent_b"],
            aliases=["Alpha Policy"],
        )
    }
    service.entity_chunk_map = {"a": ["parent_a", "parent_b"]}
    service.parent_chunks = {
        "parent_a": {"parent_chunk_id": "parent_a", "parent_doc_id": "doc_a", "source_type": "policy"},
        "parent_b": {"parent_chunk_id": "parent_b", "parent_doc_id": "doc_b", "source_type": "policy"},
    }

    results = service.retrieve_sync("alpha policy", top_k=1, depth=1)

    assert len(results) == 1
    assert isinstance(results[0], GraphRetrievedDocument)
