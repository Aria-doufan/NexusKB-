import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.index_enterprise_graph import build_graph_from_parent_chunks, iter_jsonl


def test_script_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "backend/scripts/index_enterprise_graph.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Build an EnterpriseRAG-Bench offline graph index." in result.stdout


def test_iter_jsonl_reads_non_empty_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")

    assert list(iter_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_build_graph_from_parent_chunks_merges_entities_and_writes_parent_metadata(tmp_path):
    rows = [
        {
            "parent_chunk_id": "parent_a",
            "parent_doc_id": "doc_a",
            "title": "Probation Leave Policy",
            "section_heading": "Manager Approval",
            "source_type": "policy",
            "text": "Probation Leave Policy requires Manager Approval.",
        },
        {
            "parent_chunk_id": "parent_b",
            "parent_doc_id": "doc_b",
            "title": "Probation Leave Policy",
            "section_heading": "HR Review",
            "source_type": "policy",
            "text": "Probation Leave Policy requires HR Review.",
        },
    ]

    service = build_graph_from_parent_chunks(rows, graph_dir=tmp_path / "graph")

    policy = service.entities["probation_leave_policy"]
    assert policy.source_chunk_ids == ["parent_a", "parent_b"]
    assert service.parent_chunks["parent_a"]["parent_doc_id"] == "doc_a"
    assert len(service.relations) == 2


def test_build_graph_from_parent_chunks_limit_zero_builds_empty_graph(tmp_path):
    rows = [
        {"parent_chunk_id": "", "text": "This invalid row should be skipped."},
        {
            "parent_chunk_id": "parent_a",
            "parent_doc_id": "doc_a",
            "title": "Probation Leave Policy",
            "section_heading": "Manager Approval",
            "source_type": "policy",
            "text": "Probation Leave Policy requires Manager Approval.",
        },
    ]

    service = build_graph_from_parent_chunks(rows, graph_dir=tmp_path / "graph", limit=0)

    assert service.parent_chunks == {}
    assert service.entities == {}
    assert service.relations == []


def test_build_graph_from_parent_chunks_rejects_negative_limit(tmp_path):
    with pytest.raises(ValueError, match="--limit must be non-negative"):
        build_graph_from_parent_chunks([], graph_dir=tmp_path / "graph", limit=-1)


def test_build_graph_from_parent_chunks_persists_graph_files(tmp_path):
    rows = [
        {
            "parent_chunk_id": "parent_a",
            "parent_doc_id": "doc_a",
            "title": "Probation Leave Policy",
            "section_heading": "Manager Approval",
            "source_type": "policy",
            "text": "Probation Leave Policy requires Manager Approval.",
        }
    ]

    service = build_graph_from_parent_chunks(rows, graph_dir=tmp_path / "graph")
    service.save_sync()

    assert (tmp_path / "graph" / "entities.jsonl").exists()
    assert (tmp_path / "graph" / "relations.jsonl").exists()
    assert json.loads((tmp_path / "graph" / "entity_chunk_map.json").read_text(encoding="utf-8"))["probation_leave_policy"] == ["parent_a"]
