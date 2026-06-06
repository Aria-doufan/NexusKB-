import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rag_eval_reporting import utc_run_id


def test_utc_run_id_is_unique_for_consecutive_calls():
    first = utc_run_id("retrieval")
    second = utc_run_id("retrieval")

    assert first != second
    assert "-retrieval-" in first
    assert "-retrieval-" in second
