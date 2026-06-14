"""Index prepared EnterpriseRAG-Bench parent-child chunks into Chroma.

The script intentionally uses a separate default collection/persist directory
from the app's original `rag_collection`, so benchmark experiments do not
pollute the existing development knowledge base.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "child_chunks_parent_child.jsonl"
DEFAULT_PERSIST_DIR = BACKEND_DIR / "data" / "chromadb_enterprise_parent_child"
DEFAULT_COLLECTION_NAME = "enterprise_rag_bench_parent_child"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index EnterpriseRAG-Bench chunks into Chroma.")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N chunks. Useful for smoke tests.",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=1,
        help="Split the chunks file into N contiguous parts.",
    )
    parser.add_argument(
        "--part-index",
        type=int,
        default=1,
        help="1-based part index to index when --num-parts is greater than 1.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only report how many chunks belong to the selected part.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the persist directory before indexing.",
    )
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count


def get_part_bounds(total: int, num_parts: int, part_index: int) -> tuple[int, int]:
    if num_parts <= 0:
        raise ValueError("--num-parts must be positive")
    if part_index <= 0 or part_index > num_parts:
        raise ValueError("--part-index must be in the range 1..--num-parts")
    part_size = math.ceil(total / num_parts)
    start = min((part_index - 1) * part_size, total)
    end = min(start + part_size, total)
    return start, end


def iter_jsonl(
    path: Path,
    start: int = 0,
    end: int | None = None,
    limit: int | None = None,
) -> Iterable[dict[str, Any]]:
    yielded = 0
    non_empty_index = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if non_empty_index < start:
                non_empty_index += 1
                continue
            if end is not None and non_empty_index >= end:
                break
            if limit is not None and yielded >= limit:
                break
            yield json.loads(line)
            yielded += 1
            non_empty_index += 1


def chunk_to_document(row: dict[str, Any]) -> Document:
    metadata = {
        "chunk_id": row["chunk_id"],
        "parent_doc_id": row["parent_doc_id"],
        "parent_chunk_id": row.get("parent_chunk_id", ""),
        "source_type": row.get("source_type", ""),
        "doc_semantic_type": row.get("doc_semantic_type") or "generic_doc",
        "title": row.get("title", ""),
        "chunk_index": int(row.get("chunk_index", 0)),
        "parent_chunk_index": int(row.get("parent_chunk_index", 0)),
        "child_chunk_index": int(row.get("child_chunk_index", 0)),
        "section_heading": row.get("section_heading", ""),
        "text_chars": int(row.get("text_chars", 0)),
        "parent_text_chars": int(row.get("parent_text_chars", 0)),
        "is_gold_parent": bool(row.get("is_gold_parent", False)),
    }
    return Document(page_content=row["text"], metadata=metadata)


def batched(items: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def reset_persist_dir(path: Path) -> None:
    resolved_backend = BACKEND_DIR.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_backend or resolved_backend not in resolved_path.parents:
        raise ValueError(f"Refusing to reset path outside backend directory: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def main() -> None:
    args = parse_args()
    chunks_path = args.chunks_path.resolve()
    persist_dir = args.persist_dir.resolve()

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks JSONL not found: {chunks_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    total_chunks = count_jsonl(chunks_path)
    part_start, part_end = get_part_bounds(total_chunks, args.num_parts, args.part_index)
    selected_count = max(part_end - part_start, 0)
    if args.limit is not None:
        selected_count = min(selected_count, args.limit)

    part_info = {
        "chunks_path": str(chunks_path),
        "total_chunks": total_chunks,
        "num_parts": args.num_parts,
        "part_index": args.part_index,
        "part_start_zero_based": part_start,
        "part_end_zero_based_exclusive": part_end,
        "selected_count": selected_count,
    }
    print(json.dumps(part_info, ensure_ascii=False, indent=2), flush=True)

    if args.count_only:
        return

    if args.reset:
        reset_persist_dir(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = OllamaEmbeddings(
        model=args.embedding_model,
        base_url=args.ollama_base_url,
    )
    vector_store = Chroma(
        collection_name=args.collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    started = time.perf_counter()
    indexed = 0
    for batch in batched(
        iter_jsonl(chunks_path, start=part_start, end=part_end, limit=args.limit),
        args.batch_size,
    ):
        documents = [chunk_to_document(row) for row in batch]
        ids = [row["chunk_id"] for row in batch]
        vector_store.add_documents(documents=documents, ids=ids)
        indexed += len(batch)
        elapsed = time.perf_counter() - started
        print(f"indexed={indexed} elapsed_sec={elapsed:.1f}", flush=True)

    collection_count = vector_store._collection.count()
    elapsed = time.perf_counter() - started
    result = {
        "collection_name": args.collection_name,
        "persist_dir": str(persist_dir),
        "embedding_model": args.embedding_model,
        "chunks_path": str(chunks_path),
        "total_chunks": total_chunks,
        "num_parts": args.num_parts,
        "part_index": args.part_index,
        "part_start_zero_based": part_start,
        "part_end_zero_based_exclusive": part_end,
        "indexed_this_run": indexed,
        "collection_count": collection_count,
        "elapsed_sec": round(elapsed, 2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
