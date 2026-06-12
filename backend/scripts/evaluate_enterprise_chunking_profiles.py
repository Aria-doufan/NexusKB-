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
