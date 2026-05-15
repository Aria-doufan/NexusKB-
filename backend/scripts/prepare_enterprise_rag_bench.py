"""Prepare EnterpriseRAG-Bench samples and chunks for backend RAG experiments.

This script keeps all gold documents referenced by the benchmark questions,
adds a deterministic random background sample, and emits JSONL files that can
be used by later indexing and evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset" / "EnterpriseRAG-Bench"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "enterprise_rag_bench"

STRUCTURAL_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\nsummary:",
    "\ntranscript:",
    "\ndescription:",
    "\nreview_comments:",
    "\nnotes:",
    "\nPurpose",
    "\nScope",
    "\nFrom:",
    "\nSubject:",
    "\n\n",
    "\n",
    ". ",
    "; ",
    ", ",
    " ",
    "",
]

SECTION_PATTERN = re.compile(
    r"^(#{1,6}\s+.+|[A-Za-z][A-Za-z0-9 _/-]{1,80}:|Purpose|Scope|Summary|Architecture notes.*)$"
)


@dataclass(slots=True)
class ChunkConfig:
    chunk_size: int
    chunk_overlap: int


@dataclass(slots=True)
class ParentChildConfig:
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and chunk EnterpriseRAG-Bench for RAG indexing."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to dataset/EnterpriseRAG-Bench.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for prepared JSONL/stat files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Target number of sampled documents, including all gold documents.",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--strategy",
        choices=["structured_recursive", "parent_child", "both"],
        default="structured_recursive",
        help="Chunking output strategy.",
    )
    parser.add_argument("--parent-chunk-size", type=int, default=3000)
    parser.add_argument("--parent-chunk-overlap", type=int, default=300)
    parser.add_argument("--child-chunk-size", type=int, default=700)
    parser.add_argument("--child-chunk-overlap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Parquet batch size used while scanning documents.",
    )
    return parser.parse_args()


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def build_document_text(row: dict[str, Any]) -> str:
    title = clean_text(row.get("title"))
    source_type = clean_text(row.get("source_type"))
    content = clean_text(row.get("content"))
    header = f"Title: {title}\nSource type: {source_type}"
    return f"{header}\n\n{content}".strip()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_questions(questions_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    table = pq.read_table(questions_path)
    questions: list[dict[str, Any]] = []
    expected_doc_ids: set[str] = set()

    for row in table.to_pylist():
        question = {
            "question_id": row.get("question_id"),
            "question_type": row.get("question_type"),
            "source_types": normalize_list(row.get("source_types")),
            "question": clean_text(row.get("question")),
            "expected_doc_ids": normalize_list(row.get("expected_doc_ids")),
            "gold_answer": clean_text(row.get("gold_answer")),
            "answer_facts": normalize_list(row.get("answer_facts")),
        }
        expected_doc_ids.update(question["expected_doc_ids"])
        questions.append(question)

    return questions, expected_doc_ids


def normalize_document(row: dict[str, Any], is_gold: bool) -> dict[str, Any]:
    doc_id = clean_text(row.get("doc_id"))
    source_type = clean_text(row.get("source_type"))
    title = clean_text(row.get("title"))
    content = clean_text(row.get("content"))
    text = build_document_text(
        {
            "title": title,
            "source_type": source_type,
            "content": content,
        }
    )

    return {
        "doc_id": doc_id,
        "source_type": source_type,
        "title": title,
        "content": content,
        "text": text,
        "is_gold": is_gold,
        "content_chars": len(content),
        "text_chars": len(text),
    }


def sample_documents(
    documents_path: Path,
    expected_doc_ids: set[str],
    sample_size: int,
    seed: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    target_background = max(sample_size - len(expected_doc_ids), 0)
    gold_docs: dict[str, dict[str, Any]] = {}
    background_docs: list[dict[str, Any]] = []
    seen_background = 0
    total_docs = 0
    source_type_counter: Counter[str] = Counter()

    parquet_file = pq.ParquetFile(documents_path)
    columns = ["doc_id", "source_type", "title", "content"]

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            total_docs += 1
            doc_id = clean_text(row.get("doc_id"))
            source_type_counter[clean_text(row.get("source_type"))] += 1

            if doc_id in expected_doc_ids:
                gold_docs[doc_id] = normalize_document(row, is_gold=True)
                continue

            if target_background <= 0:
                continue

            seen_background += 1
            doc = normalize_document(row, is_gold=False)
            if len(background_docs) < target_background:
                background_docs.append(doc)
                continue

            replacement_index = rng.randrange(seen_background)
            if replacement_index < target_background:
                background_docs[replacement_index] = doc

    missing_gold_doc_ids = sorted(expected_doc_ids.difference(gold_docs.keys()))
    documents = list(gold_docs.values()) + background_docs
    documents.sort(key=lambda item: (not item["is_gold"], item["source_type"], item["doc_id"]))

    stats = {
        "total_dataset_documents": total_docs,
        "target_sample_size": sample_size,
        "sampled_documents": len(documents),
        "gold_doc_ids_from_questions": len(expected_doc_ids),
        "gold_documents_found": len(gold_docs),
        "missing_gold_documents": len(missing_gold_doc_ids),
        "missing_gold_doc_ids": missing_gold_doc_ids[:50],
        "background_documents": len(background_docs),
        "source_type_distribution_full": dict(source_type_counter.most_common()),
    }
    return documents, stats


def split_by_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    parts = text.split(separator)
    return [parts[0]] + [separator + part for part in parts[1:]]


def recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    pieces = split_by_separator(text, separator)
    if len(pieces) == 1:
        return recursive_split(text, chunk_size, separators[1:])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        candidate = piece if not current else f"{current}{piece}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.extend(recursive_split(current, chunk_size, separators[1:]))
        current = piece

    if current:
        chunks.extend(recursive_split(current, chunk_size, separators[1:]))
    return chunks


def add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        prefix = previous[-overlap:].strip()
        if prefix:
            overlapped.append(f"{prefix}\n\n{current}".strip())
        else:
            overlapped.append(current)
    return overlapped


def extract_section_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if SECTION_PATTERN.match(stripped):
            return stripped[:120]
    return ""


def chunk_document(document: dict[str, Any], config: ChunkConfig) -> list[dict[str, Any]]:
    raw_chunks = recursive_split(document["text"], config.chunk_size, STRUCTURAL_SEPARATORS)
    raw_chunks = add_overlap(raw_chunks, config.chunk_overlap)
    chunks: list[dict[str, Any]] = []

    for index, text in enumerate(raw_chunks):
        text = clean_text(text)
        if not text:
            continue
        chunk_id = f"{document['doc_id']}::chunk_{index:04d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "parent_doc_id": document["doc_id"],
                "source_type": document["source_type"],
                "title": document["title"],
                "chunk_index": index,
                "section_heading": extract_section_heading(text),
                "text": text,
                "text_chars": len(text),
                "is_gold_parent": document["is_gold"],
            }
        )

    return chunks


def build_chunks(
    documents: Iterable[dict[str, Any]],
    config: ChunkConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunks_per_doc: Counter[str] = Counter()
    chunks_per_source_type: Counter[str] = Counter()
    chunk_lengths: list[int] = []

    for document in documents:
        document_chunks = chunk_document(document, config)
        chunks.extend(document_chunks)
        chunks_per_doc[document["doc_id"]] = len(document_chunks)
        chunks_per_source_type[document["source_type"]] += len(document_chunks)
        chunk_lengths.extend(chunk["text_chars"] for chunk in document_chunks)

    if chunk_lengths:
        average_chunk_chars = round(sum(chunk_lengths) / len(chunk_lengths), 2)
        max_chunk_chars = max(chunk_lengths)
        min_chunk_chars = min(chunk_lengths)
    else:
        average_chunk_chars = 0
        max_chunk_chars = 0
        min_chunk_chars = 0

    stats = {
        "chunk_strategy": "structured_recursive",
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "total_chunks": len(chunks),
        "average_chunk_chars": average_chunk_chars,
        "min_chunk_chars": min_chunk_chars,
        "max_chunk_chars": max_chunk_chars,
        "chunks_per_source_type": dict(chunks_per_source_type.most_common()),
        "average_chunks_per_doc": round(len(chunks) / max(len(chunks_per_doc), 1), 2),
    }
    return chunks, stats


def chunk_document_parent_child(
    document: dict[str, Any],
    config: ParentChildConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parent_texts = recursive_split(
        document["text"],
        config.parent_chunk_size,
        STRUCTURAL_SEPARATORS,
    )
    parent_texts = add_overlap(parent_texts, config.parent_chunk_overlap)

    parent_chunks: list[dict[str, Any]] = []
    child_chunks: list[dict[str, Any]] = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_text = clean_text(parent_text)
        if not parent_text:
            continue

        parent_chunk_id = f"{document['doc_id']}::parent_{parent_index:04d}"
        section_heading = extract_section_heading(parent_text)
        parent_chunks.append(
            {
                "parent_chunk_id": parent_chunk_id,
                "parent_doc_id": document["doc_id"],
                "source_type": document["source_type"],
                "title": document["title"],
                "parent_chunk_index": parent_index,
                "section_heading": section_heading,
                "text": parent_text,
                "text_chars": len(parent_text),
                "is_gold_parent": document["is_gold"],
            }
        )

        child_texts = recursive_split(
            parent_text,
            config.child_chunk_size,
            STRUCTURAL_SEPARATORS,
        )
        child_texts = add_overlap(child_texts, config.child_chunk_overlap)
        for child_index, child_text in enumerate(child_texts):
            child_text = clean_text(child_text)
            if not child_text:
                continue
            child_chunk_id = f"{parent_chunk_id}::child_{child_index:04d}"
            child_chunks.append(
                {
                    "chunk_id": child_chunk_id,
                    "parent_chunk_id": parent_chunk_id,
                    "parent_doc_id": document["doc_id"],
                    "source_type": document["source_type"],
                    "title": document["title"],
                    "parent_chunk_index": parent_index,
                    "child_chunk_index": child_index,
                    "section_heading": extract_section_heading(child_text) or section_heading,
                    "text": child_text,
                    "text_chars": len(child_text),
                    "parent_text_chars": len(parent_text),
                    "is_gold_parent": document["is_gold"],
                }
            )

    return parent_chunks, child_chunks


def build_parent_child_chunks(
    documents: Iterable[dict[str, Any]],
    config: ParentChildConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    parent_chunks: list[dict[str, Any]] = []
    child_chunks: list[dict[str, Any]] = []
    parent_lengths: list[int] = []
    child_lengths: list[int] = []
    parent_chunks_per_source_type: Counter[str] = Counter()
    child_chunks_per_source_type: Counter[str] = Counter()
    parent_chunks_per_doc: Counter[str] = Counter()
    child_chunks_per_doc: Counter[str] = Counter()

    for document in documents:
        doc_parent_chunks, doc_child_chunks = chunk_document_parent_child(document, config)
        parent_chunks.extend(doc_parent_chunks)
        child_chunks.extend(doc_child_chunks)
        parent_chunks_per_doc[document["doc_id"]] = len(doc_parent_chunks)
        child_chunks_per_doc[document["doc_id"]] = len(doc_child_chunks)

        for chunk in doc_parent_chunks:
            parent_lengths.append(chunk["text_chars"])
            parent_chunks_per_source_type[chunk["source_type"]] += 1
        for chunk in doc_child_chunks:
            child_lengths.append(chunk["text_chars"])
            child_chunks_per_source_type[chunk["source_type"]] += 1

    stats = {
        "chunk_strategy": "parent_child",
        "parent_chunk_size": config.parent_chunk_size,
        "parent_chunk_overlap": config.parent_chunk_overlap,
        "child_chunk_size": config.child_chunk_size,
        "child_chunk_overlap": config.child_chunk_overlap,
        "total_parent_chunks": len(parent_chunks),
        "total_child_chunks": len(child_chunks),
        "average_parent_chunk_chars": round(sum(parent_lengths) / max(len(parent_lengths), 1), 2),
        "average_child_chunk_chars": round(sum(child_lengths) / max(len(child_lengths), 1), 2),
        "min_parent_chunk_chars": min(parent_lengths) if parent_lengths else 0,
        "max_parent_chunk_chars": max(parent_lengths) if parent_lengths else 0,
        "min_child_chunk_chars": min(child_lengths) if child_lengths else 0,
        "max_child_chunk_chars": max(child_lengths) if child_lengths else 0,
        "average_parent_chunks_per_doc": round(
            len(parent_chunks) / max(len(parent_chunks_per_doc), 1), 2
        ),
        "average_child_chunks_per_doc": round(
            len(child_chunks) / max(len(child_chunks_per_doc), 1), 2
        ),
        "parent_chunks_per_source_type": dict(parent_chunks_per_source_type.most_common()),
        "child_chunks_per_source_type": dict(child_chunks_per_source_type.most_common()),
    }
    return parent_chunks, child_chunks, stats


def main() -> None:
    args = parse_args()
    documents_path = args.dataset_dir / "data" / "documents" / "test.parquet"
    questions_path = args.dataset_dir / "data" / "questions" / "test.parquet"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not documents_path.exists():
        raise FileNotFoundError(f"Documents parquet not found: {documents_path}")
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions parquet not found: {questions_path}")

    questions, expected_doc_ids = read_questions(questions_path)
    documents, sample_stats = sample_documents(
        documents_path=documents_path,
        expected_doc_ids=expected_doc_ids,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    documents_count = write_jsonl(output_dir / "documents_sample.jsonl", documents)
    questions_count = write_jsonl(output_dir / "questions.jsonl", questions)
    output_files = {
        "documents": str(output_dir / "documents_sample.jsonl"),
        "questions": str(output_dir / "questions.jsonl"),
    }

    chunk_stats: dict[str, Any] = {"chunk_strategy": args.strategy}

    if args.strategy in {"structured_recursive", "both"}:
        chunks, structured_stats = build_chunks(
            documents=documents,
            config=ChunkConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        )
        chunks_count = write_jsonl(output_dir / "chunks_structured_recursive.jsonl", chunks)
        chunk_stats["structured_recursive"] = {
            **structured_stats,
            "chunks_written": chunks_count,
        }
        output_files["structured_recursive_chunks"] = str(
            output_dir / "chunks_structured_recursive.jsonl"
        )

    if args.strategy in {"parent_child", "both"}:
        parent_chunks, child_chunks, parent_child_stats = build_parent_child_chunks(
            documents=documents,
            config=ParentChildConfig(
                parent_chunk_size=args.parent_chunk_size,
                parent_chunk_overlap=args.parent_chunk_overlap,
                child_chunk_size=args.child_chunk_size,
                child_chunk_overlap=args.child_chunk_overlap,
            ),
        )
        parent_chunks_count = write_jsonl(
            output_dir / "parent_chunks_parent_child.jsonl",
            parent_chunks,
        )
        child_chunks_count = write_jsonl(
            output_dir / "child_chunks_parent_child.jsonl",
            child_chunks,
        )
        chunk_stats["parent_child"] = {
            **parent_child_stats,
            "parent_chunks_written": parent_chunks_count,
            "child_chunks_written": child_chunks_count,
        }
        output_files["parent_chunks"] = str(output_dir / "parent_chunks_parent_child.jsonl")
        output_files["child_chunks"] = str(output_dir / "child_chunks_parent_child.jsonl")

    source_type_distribution_sample = Counter(doc["source_type"] for doc in documents)
    stats = {
        **sample_stats,
        **chunk_stats,
        "questions": questions_count,
        "documents_written": documents_count,
        "source_type_distribution_sample": dict(source_type_distribution_sample.most_common()),
        "output_files": output_files,
    }

    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
