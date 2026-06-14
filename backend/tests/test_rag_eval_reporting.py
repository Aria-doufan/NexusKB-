import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rag_eval_reporting import _render_group_summary_table, utc_run_id


def test_utc_run_id_is_unique_for_consecutive_calls():
    first = utc_run_id("retrieval")
    second = utc_run_id("retrieval")

    assert first != second
    assert "-retrieval-" in first
    assert "-retrieval-" in second


def test_render_group_summary_table_escapes_group_and_metric_cells():
    table = _render_group_summary_table(
        "Document Semantic Type",
        {"policy|doc": {"questions": 1, "hit|1": 1.0}},
        ["questions", "hit|1"],
        {},
    )

    assert "policy\\|doc" in table
    assert "hit\\|1" in table
    assert "| policy|doc |" not in table
    assert "| hit|1 |" not in table
