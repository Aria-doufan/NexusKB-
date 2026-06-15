import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.index_enterprise_chunks_elasticsearch import build_mapping, chunk_to_es_document, load_parent_chunks


def test_build_mapping_contains_text_keyword_and_dense_vector_fields():
    mapping = build_mapping(vector_dims=1024)
    props = mapping["mappings"]["properties"]

    assert props["chunk_id"]["type"] == "keyword"
    assert props["parent_doc_id"]["type"] == "keyword"
    assert props["source_type"]["type"] == "keyword"
    assert props["doc_semantic_type"]["type"] == "keyword"
    assert props["child_text"]["type"] == "text"
    assert props["parent_text"]["type"] == "text"
    assert props["embedding"]["type"] == "dense_vector"
    assert props["embedding"]["dims"] == 1024
    assert props["embedding"]["index"] is True


def test_chunk_to_es_document_joins_parent_text(tmp_path):
    parents = {
        "parent-1": {
            "text": "parent text",
            "parent_doc_id": "doc-1",
            "source_type": "policy",
            "doc_semantic_type": "policy_rule",
            "title": "Policy",
            "section_heading": "Section",
        }
    }
    child = {
        "chunk_id": "child-1",
        "parent_chunk_id": "parent-1",
        "parent_doc_id": "doc-1",
        "source_type": "policy",
        "doc_semantic_type": "policy_rule",
        "title": "Policy",
        "section_heading": "Section",
        "text": "child text",
        "chunk_index": 3,
        "parent_chunk_index": 1,
        "child_chunk_index": 2,
    }

    document = chunk_to_es_document(child, parents, embedding=[0.1, 0.2])

    assert document["chunk_id"] == "child-1"
    assert document["parent_chunk_id"] == "parent-1"
    assert document["doc_semantic_type"] == "policy_rule"
    assert document["child_text"] == "child text"
    assert document["parent_text"] == "parent text"
    assert document["embedding"] == [0.1, 0.2]


def test_load_parent_chunks_by_id(tmp_path):
    path = tmp_path / "parents.jsonl"
    path.write_text(json.dumps({"parent_chunk_id": "p1", "text": "hello"}) + "\n", encoding="utf-8")

    parents = load_parent_chunks(path)

    assert parents == {"p1": {"parent_chunk_id": "p1", "text": "hello"}}
