"""Run recall-focused enterprise chunking profile evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = BACKEND_DIR / "data" / "chunking_eval_outputs"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_METHOD = "chroma_bm25_rrf_reranker"
DEFAULT_K_VALUES = "1,5,10,20"


@dataclass(frozen=True, slots=True)
class ChunkProfile:
    name: str
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int

    @property
    def slug(self) -> str:
        return (
            f"{self.name}_"
            f"p{self.parent_chunk_size}-o{self.parent_chunk_overlap}_"
            f"c{self.child_chunk_size}-o{self.child_chunk_overlap}"
        )

    def to_record(self) -> dict[str, int | str]:
        return asdict(self)


def stage1_profiles() -> list[ChunkProfile]:
    return [
        ChunkProfile("baseline", 3000, 300, 700, 100),
        ChunkProfile("smaller_child", 3000, 300, 500, 80),
        ChunkProfile("larger_child", 3000, 300, 900, 120),
        ChunkProfile("larger_parent", 4000, 400, 700, 100),
    ]


def build_prepare_command(
    profile: ChunkProfile,
    output_dir: Path,
    sample_size: int,
) -> list[str]:
    return [
        "python",
        "backend/scripts/prepare_enterprise_rag_bench.py",
        "--strategy",
        "parent_child",
        "--output-dir",
        str(output_dir),
        "--sample-size",
        str(sample_size),
        "--parent-chunk-size",
        str(profile.parent_chunk_size),
        "--parent-chunk-overlap",
        str(profile.parent_chunk_overlap),
        "--child-chunk-size",
        str(profile.child_chunk_size),
        "--child-chunk-overlap",
        str(profile.child_chunk_overlap),
    ]


def build_index_command(
    chunks_path: Path,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    reset: bool,
) -> list[str]:
    command = [
        "python",
        "backend/scripts/index_enterprise_chunks_chroma.py",
        "--chunks-path",
        str(chunks_path),
        "--persist-dir",
        str(persist_dir),
        "--collection-name",
        collection_name,
        "--embedding-model",
        embedding_model,
    ]
    if reset:
        command.append("--reset")
    return command


def build_eval_command(
    prepared_dir: Path,
    persist_dir: Path,
    collection_name: str,
    output_dir: Path,
    method: str,
    k_values: str,
    limit: int | None,
) -> list[str]:
    command = [
        "python",
        "backend/scripts/evaluate_enterprise_hybrid_retrieval.py",
        "--method",
        method,
        "--questions-path",
        str(prepared_dir / "questions.jsonl"),
        "--child-chunks-path",
        str(prepared_dir / "child_chunks_parent_child.jsonl"),
        "--parent-chunks-path",
        str(prepared_dir / "parent_chunks_parent_child.jsonl"),
        "--persist-dir",
        str(persist_dir),
        "--collection-name",
        collection_name,
        "--output-dir",
        str(output_dir),
        "--k-values",
        k_values,
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command
