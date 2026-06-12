from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

from langchain_ollama import OllamaEmbeddings


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHILD_CHUNKS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "child_chunks_parent_child.jsonl"
DEFAULT_PARENT_CHUNKS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "parent_chunks_parent_child.jsonl"
DEFAULT_URL = "http://localhost:9200"
DEFAULT_INDEX_NAME = "nexuskb_enterprise_chunks"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def build_mapping(vector_dims: int) -> dict[str, Any]:
    return {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "parent_chunk_id": {"type": "keyword"},
                "parent_doc_id": {"type": "keyword"},
                "source_type": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "section_heading": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "child_text": {"type": "text"},
                "parent_text": {"type": "text"},
                "chunk_index": {"type": "integer"},
                "parent_chunk_index": {"type": "integer"},
                "child_chunk_index": {"type": "integer"},
                "embedding": {"type": "dense_vector", "dims": vector_dims, "index": True, "similarity": "cosine"},
            }
        }
    }


def load_parent_chunks(path: Path) -> dict[str, dict[str, Any]]:
    parents: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            parents[row["parent_chunk_id"]] = row
    return parents


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if limit is not None and count >= limit:
                break
            yield json.loads(line)
            count += 1


def chunk_to_es_document(child: dict[str, Any], parents: dict[str, dict[str, Any]], embedding: list[float]) -> dict[str, Any]:
    parent_chunk_id = child.get("parent_chunk_id", "")
    parent = parents.get(parent_chunk_id, {})
    return {
        "chunk_id": child["chunk_id"],
        "parent_chunk_id": parent_chunk_id,
        "parent_doc_id": child.get("parent_doc_id") or parent.get("parent_doc_id", ""),
        "source_type": child.get("source_type") or parent.get("source_type", ""),
        "title": child.get("title") or parent.get("title", ""),
        "section_heading": child.get("section_heading") or parent.get("section_heading", ""),
        "child_text": child.get("text", ""),
        "parent_text": parent.get("text", child.get("text", "")),
        "chunk_index": int(child.get("chunk_index", 0)),
        "parent_chunk_index": int(child.get("parent_chunk_index", 0)),
        "child_chunk_index": int(child.get("child_chunk_index", 0)),
        "embedding": embedding,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index EnterpriseRAG-Bench chunks into Elasticsearch.")
    parser.add_argument("--child-chunks-path", type=Path, default=DEFAULT_CHILD_CHUNKS_PATH)
    parser.add_argument("--parent-chunks-path", type=Path, default=DEFAULT_PARENT_CHUNKS_PATH)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from elasticsearch import Elasticsearch, helpers

    parents = load_parent_chunks(args.parent_chunks_path.resolve())
    embeddings = OllamaEmbeddings(model=args.embedding_model, base_url=args.ollama_base_url)
    client = Elasticsearch(args.url)

    first_child = next(iter_jsonl(args.child_chunks_path.resolve(), limit=1), None)
    if first_child is None:
        raise ValueError(f"No child chunks found: {args.child_chunks_path}")
    first_embedding = embeddings.embed_query(first_child.get("text", ""))
    mapping = build_mapping(vector_dims=len(first_embedding))

    if args.reset and client.indices.exists(index=args.index_name):
        client.indices.delete(index=args.index_name)
    if not client.indices.exists(index=args.index_name):
        client.indices.create(index=args.index_name, **mapping)

    started = time.perf_counter()
    indexed = 0

    def actions() -> Iterable[dict[str, Any]]:
        nonlocal indexed
        for child in iter_jsonl(args.child_chunks_path.resolve(), limit=args.limit):
            embedding = first_embedding if indexed == 0 and child.get("chunk_id") == first_child.get("chunk_id") else embeddings.embed_query(child.get("text", ""))
            document = chunk_to_es_document(child, parents, embedding)
            indexed += 1
            yield {"_index": args.index_name, "_id": document["chunk_id"], "_source": document}

    helpers.bulk(client, actions(), chunk_size=args.batch_size)
    elapsed = time.perf_counter() - started
    print(json.dumps({"index_name": args.index_name, "indexed": indexed, "embedding_model": args.embedding_model, "elapsed_sec": round(elapsed, 2)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
