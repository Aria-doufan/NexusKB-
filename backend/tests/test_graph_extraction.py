import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.graph_extraction import extract_graph_from_parent_chunk


def test_extract_graph_from_parent_chunk_uses_metadata_entities():
    row = {
        "parent_chunk_id": "parent_a",
        "parent_doc_id": "doc_a",
        "title": "Probation Leave Policy",
        "section_heading": "Manager Approval",
        "source_type": "policy",
        "text": "Employees must request approval before leave.",
    }

    result = extract_graph_from_parent_chunk(row)

    names = {entity.name for entity in result.entities}
    assert "Probation Leave Policy" in names
    assert "Manager Approval" in names
    assert "policy" in names


def test_extract_graph_from_parent_chunk_creates_requires_relation_from_text():
    row = {
        "parent_chunk_id": "parent_a",
        "parent_doc_id": "doc_a",
        "title": "Probation Leave Policy",
        "section_heading": "Manager Approval",
        "source_type": "policy",
        "text": "Probation Leave Policy requires Manager Approval before vacation.",
    }

    result = extract_graph_from_parent_chunk(row)

    assert len(result.relationships) == 1
    relation = result.relationships[0]
    assert relation["source"] == "Probation Leave Policy"
    assert relation["target"] == "Manager Approval"
    assert relation["type"] == "requires"
    assert relation["relation_id"] == "probation_leave_policy__requires__manager_approval"
    assert relation["src_entity_id"] == "probation_leave_policy"
    assert relation["tgt_entity_id"] == "manager_approval"
    assert relation["relation_type"] == "requires"
    assert relation["keywords"] == ["probation", "leave", "policy", "requires", "manager", "approval"]


def test_extract_graph_from_parent_chunk_limits_entities_and_relations():
    row = {
        "parent_chunk_id": "parent_a",
        "parent_doc_id": "doc_a",
        "title": "A",
        "section_heading": "B",
        "source_type": "policy",
        "text": "A requires B. C requires D. E requires F. G requires H.",
    }

    result = extract_graph_from_parent_chunk(row, max_entities=3, max_relations=2, max_relation_keywords=3)

    entity_ids = {entity.entity_id for entity in result.entities}

    assert len(result.entities) == 3
    assert len(result.relationships) == 1
    assert all(relation["src_entity_id"] in entity_ids for relation in result.relationships)
    assert all(relation["tgt_entity_id"] in entity_ids for relation in result.relationships)
    assert all(len(relation["keywords"]) <= 3 for relation in result.relationships)


def test_extract_graph_from_parent_chunk_relationship_endpoints_are_present():
    row = {
        "parent_chunk_id": "parent_a",
        "parent_doc_id": "doc_a",
        "title": "Probation Leave Policy",
        "section_heading": "Manager Approval",
        "source_type": "policy",
        "text": "Probation Leave Policy requires Manager Approval before vacation.",
    }

    result = extract_graph_from_parent_chunk(row)
    entity_ids = {entity.entity_id for entity in result.entities}

    assert result.relationships
    for relation in result.relationships:
        assert relation["src_entity_id"] in entity_ids
        assert relation["tgt_entity_id"] in entity_ids


def test_extract_graph_from_parent_chunk_allows_zero_relation_keywords():
    row = {
        "parent_chunk_id": "parent_a",
        "parent_doc_id": "doc_a",
        "title": "Probation Leave Policy",
        "section_heading": "Manager Approval",
        "source_type": "policy",
        "text": "Probation Leave Policy requires Manager Approval before vacation.",
    }

    result = extract_graph_from_parent_chunk(row, max_relation_keywords=0)

    assert len(result.relationships) == 1
    assert result.relationships[0]["keywords"] == []
