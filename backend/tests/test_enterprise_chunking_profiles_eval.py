import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_enterprise_chunking_profiles import (
    ChunkProfile,
    build_eval_command,
    build_index_command,
    build_prepare_command,
    stage1_profiles,
)


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


def test_build_prepare_command_uses_parent_child_profile_paths():
    profile = ChunkProfile("baseline", 3000, 300, 700, 100)
    output_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/prepared")

    command = build_prepare_command(profile=profile, output_dir=output_dir, sample_size=10000)

    assert command[:4] == ["python", "backend/scripts/prepare_enterprise_rag_bench.py", "--strategy", "parent_child"]
    assert "--output-dir" in command
    assert str(output_dir) in command
    assert command[command.index("--parent-chunk-size") + 1] == "3000"
    assert command[command.index("--parent-chunk-overlap") + 1] == "300"
    assert command[command.index("--child-chunk-size") + 1] == "700"
    assert command[command.index("--child-chunk-overlap") + 1] == "100"


def test_build_index_command_uses_profile_specific_collection_and_paths():
    chunks_path = Path("backend/data/chunking_eval_outputs/stage1/baseline/prepared/child_chunks_parent_child.jsonl")
    persist_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/chroma")

    command = build_index_command(
        chunks_path=chunks_path,
        persist_dir=persist_dir,
        collection_name="enterprise_chunking_baseline",
        embedding_model="qwen3-embedding:latest",
        reset=True,
    )

    assert command[:2] == ["python", "backend/scripts/index_enterprise_chunks_chroma.py"]
    assert command[command.index("--chunks-path") + 1] == str(chunks_path)
    assert command[command.index("--persist-dir") + 1] == str(persist_dir)
    assert command[command.index("--collection-name") + 1] == "enterprise_chunking_baseline"
    assert "--reset" in command


def test_build_eval_command_points_at_profile_index_and_chunks():
    prepared_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/prepared")
    persist_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/chroma")
    output_dir = Path("backend/data/chunking_eval_outputs/stage1/baseline/eval")

    command = build_eval_command(
        prepared_dir=prepared_dir,
        persist_dir=persist_dir,
        collection_name="enterprise_chunking_baseline",
        output_dir=output_dir,
        method="chroma_bm25_rrf_reranker",
        k_values="1,5,10,20",
        limit=25,
    )

    assert command[:2] == ["python", "backend/scripts/evaluate_enterprise_hybrid_retrieval.py"]
    assert command[command.index("--questions-path") + 1] == str(prepared_dir / "questions.jsonl")
    assert command[command.index("--child-chunks-path") + 1] == str(prepared_dir / "child_chunks_parent_child.jsonl")
    assert command[command.index("--parent-chunks-path") + 1] == str(prepared_dir / "parent_chunks_parent_child.jsonl")
    assert command[command.index("--persist-dir") + 1] == str(persist_dir)
    assert command[command.index("--output-dir") + 1] == str(output_dir)
    assert command[command.index("--limit") + 1] == "25"
