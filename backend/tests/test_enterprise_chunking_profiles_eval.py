import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_enterprise_chunking_profiles import ChunkProfile, stage1_profiles


def test_stage1_profiles_match_design_matrix():
    profiles = stage1_profiles()

    assert [profile.name for profile in profiles] == [
        "baseline",
        "smaller_child",
        "larger_child",
        "larger_parent",
    ]
    assert profiles[0] == ChunkProfile("baseline", 3000, 300, 700, 100)
    assert profiles[1] == ChunkProfile("smaller_child", 3000, 300, 500, 80)
    assert profiles[2] == ChunkProfile("larger_child", 3000, 300, 900, 120)
    assert profiles[3] == ChunkProfile("larger_parent", 4000, 400, 700, 100)


def test_chunk_profile_slug_is_safe_for_paths():
    profile = ChunkProfile("smaller_child", 3000, 300, 500, 80)

    assert profile.slug == "smaller_child_p3000-o300_c500-o80"
