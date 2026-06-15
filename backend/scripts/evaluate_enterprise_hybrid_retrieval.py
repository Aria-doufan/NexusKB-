"""Evaluate EnterpriseRAG-Bench retrieval variants.

This script compares:

1. Chroma child vector search.
2. BM25 child lexical search.
3. Chroma + BM25 fused by RRF.
4. Optional source hint soft boost and reranking over fused parent candidates.
5. The current service strategy matrix used by EnterpriseRagService.

The evaluation target remains parent_doc_id relevance against expected_doc_ids.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from scripts.rag_eval_metrics import (
    average_precision_at_k,
    build_group_summary,
    build_question_type_summary,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from scripts.rag_eval_reporting import load_json_if_exists, render_retrieval_report, short_git_commit, utc_run_id
from app.rag.metadata_filter_planner import ALLOWED_DOC_SEMANTIC_TYPES, ALLOWED_SOURCE_TYPES
from app.schemas.rag import MetadataFilterDecision


DEFAULT_QUESTIONS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "questions.jsonl"
DEFAULT_CHILD_CHUNKS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "child_chunks_parent_child.jsonl"
DEFAULT_PARENT_CHUNKS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "parent_chunks_parent_child.jsonl"
DEFAULT_GRAPH_DIR = BACKEND_DIR / "data" / "enterprise_rag_bench" / "graph"
DEFAULT_PERSIST_DIR = BACKEND_DIR / "data" / "chromadb_enterprise_parent_child"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "enterprise_rag_bench" / "eval"
DEFAULT_COLLECTION_NAME = "enterprise_rag_bench_parent_child"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_RERANKER_MODEL_PATH = r"D:\Hugging_Face\models\Qwen3-Reranker-0.6B"
TOKEN_PATTERN = re.compile(r"[a-z0-9_.$%/-]+")
SOURCE_HINT_SOFT_BOOST = 0.15
COMPLEX_QUESTION_TYPES = {
    "semantic",
    "semantic_query",
    "multi_hop",
    "comparison",
    "intra_document_reasoning",
    "project_related",
    "constrained",
    "conflicting_info",
    "completeness",
    "high_level",
}
METHOD_ALIASES = {
    "hybrid_bm25_rrf": "chroma_bm25_rrf",
    "hybrid_bm25_rrf_reranker": "chroma_bm25_rrf_reranker",
}
METHODS = [
    "chroma_only",
    "bm25_only",
    "chroma_bm25_rrf",
    "chroma_bm25_rrf_source_boost",
    "chroma_bm25_rrf_reranker",
    "chroma_bm25_graph_rrf",
    "chroma_bm25_graph_rrf_reranker",
    "strategy_matrix",
    "strategy_matrix_decompose",
    "strategy_matrix_graph",
    "strategy_matrix_decompose_graph",
    "hybrid_bm25_rrf",
    "hybrid_bm25_rrf_reranker",
]


@dataclass(slots=True)
class Question:
    question_id: str
    question_type: str
    source_types: list[str]
    question: str
    expected_doc_ids: list[str]
    gold_answer: str
    answer_facts: list[str]
    required_evidence_groups: list[list[str]] = field(default_factory=list)


@dataclass(slots=True)
class Candidate:
    chunk_id: str
    parent_doc_id: str
    parent_chunk_id: str
    source_type: str
    doc_semantic_type: str = "generic_doc"
    title: str = ""
    section_heading: str = ""
    text: str = ""
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    graph_rank: int | None = None
    graph_score: float | None = None
    evidence_chunk_ids: list[str] = field(default_factory=list)
    fused_score: float = 0.0
    reranker_score: float | None = None


class Qwen3CausalReranker:
    """Qwen3 reranker scorer using the official yes/no causal-LM logit method."""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_length: int = 2048,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(self.device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.instruction = "Given a web search query, retrieve relevant passages that answer the query"

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 1) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            inputs = self._process_inputs(batch_pairs)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
                true_vector = logits[:, self.token_true_id]
                false_vector = logits[:, self.token_false_id]
                batch_scores = self.torch.stack([false_vector, true_vector], dim=1)
                batch_scores = self.torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores.extend(batch_scores[:, 1].exp().tolist())
        return scores

    def _process_inputs(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        formatted_pairs = [
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
            for query, document in pairs
        ]
        token_budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = self.tokenizer(
            formatted_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=token_budget,
        )
        for index, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = self.prefix_tokens + input_ids + self.suffix_tokens
        padded_inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        return {key: value.to(self.model.device) for key, value in padded_inputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EnterpriseRAG-Bench hybrid retrieval.")
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="chroma_bm25_rrf",
    )
    parser.add_argument(
        "--backend",
        choices=["chroma", "elasticsearch"],
        default="chroma",
        help="Retrieval backend used for candidate generation. Chroma preserves the existing evaluator path; Elasticsearch uses the enterprise ES backend.",
    )
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--child-chunks-path", type=Path, default=DEFAULT_CHILD_CHUNKS_PATH)
    parser.add_argument("--parent-chunks-path", type=Path, default=DEFAULT_PARENT_CHUNKS_PATH)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chroma-search-k", type=int, default=50)
    parser.add_argument("--bm25-search-k", type=int, default=50)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--graph-search-k", type=int, default=40)
    parser.add_argument("--graph-depth", type=int, default=2)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--source-boost", type=float, default=SOURCE_HINT_SOFT_BOOST)
    parser.add_argument("--k-values", default="1,5,10,20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--where-source-type",
        default=None,
        help="Optional single source_type metadata filter. Keep empty for comparable experiments.",
    )
    parser.add_argument("--metadata-filter-mode", choices=["none", "soft", "hard"], default="none")
    parser.add_argument("--filter-source-types", default="")
    parser.add_argument("--filter-doc-semantic-types", default="")
    parser.add_argument("--reranker-model-path", default=os.getenv("RERANKER_MODEL_PATH", DEFAULT_RERANKER_MODEL_PATH))
    parser.add_argument("--reranker-device", default=None, help="Optional device override, for example cpu or cuda.")
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-candidate-k", type=int, default=20)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--run-output-root", type=Path, default=BACKEND_DIR / "data" / "eval_outputs")
    parser.add_argument("--baseline-dir", type=Path, default=BACKEND_DIR / "data" / "eval_baselines" / "current")
    parser.add_argument("--standard-output", action="store_true", help="Also write standardized run artifacts and report.md.")
    parser.add_argument("--ci", action="store_true", help="Record CI mode in config without enforcing regressions in this version.")
    parser.add_argument("--fail-on-regression", action="store_true", help="Reserved for future threshold enforcement; no-op in this version.")
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_questions(path: Path, limit: int | None = None) -> list[Question]:
    questions: list[Question] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if limit is not None and len(questions) >= limit:
                break
            row = json.loads(line)
            required_evidence_groups = []
            for group in row.get("required_evidence_groups") or []:
                if not isinstance(group, list):
                    continue
                normalized_group = [
                    str(chunk_id).strip()
                    for chunk_id in group
                    if chunk_id is not None and str(chunk_id).strip()
                ]
                if normalized_group:
                    required_evidence_groups.append(normalized_group)
            questions.append(
                Question(
                    question_id=row["question_id"],
                    question_type=row.get("question_type", ""),
                    source_types=list(row.get("source_types") or []),
                    question=row["question"],
                    expected_doc_ids=list(row.get("expected_doc_ids") or []),
                    gold_answer=row.get("gold_answer", ""),
                    answer_facts=list(row.get("answer_facts") or []),
                    required_evidence_groups=required_evidence_groups,
                )
            )
    return questions


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--k-values must contain positive integers")
    return values


def _split_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def build_metadata_filter_decision(filter_mode: str, source_types: str, doc_semantic_types: str) -> MetadataFilterDecision:
    sources = _split_csv(source_types)
    semantic_types = _split_csv(doc_semantic_types)
    for source_type in sources:
        if source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"Unsupported source_type for metadata filter: {source_type}")
    for semantic_type in semantic_types:
        if semantic_type not in ALLOWED_DOC_SEMANTIC_TYPES:
            raise ValueError(f"Unsupported doc_semantic_type for metadata filter: {semantic_type}")
    if filter_mode == "none":
        return MetadataFilterDecision(mode="none")
    if not sources and not semantic_types:
        raise ValueError("--metadata-filter-mode requires --filter-source-types or --filter-doc-semantic-types when mode is soft or hard")
    return MetadataFilterDecision(
        mode=filter_mode,
        source_types=sources,
        doc_semantic_types=semantic_types,
        confidence=1.0,
        reason="Manual evaluation metadata filter.",
    )


def normalize_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def method_needs_chroma(method: str) -> bool:
    return normalize_method(method) != "bm25_only"


def method_needs_bm25(method: str) -> bool:
    return normalize_method(method) != "chroma_only"


def method_needs_graph(method: str) -> bool:
    return normalize_method(method) in {
        "chroma_bm25_graph_rrf",
        "chroma_bm25_graph_rrf_reranker",
        "strategy_matrix_graph",
        "strategy_matrix_decompose_graph",
    }


def validate_graph_cli_args(args: argparse.Namespace, normalized_method: str) -> None:
    if not method_needs_graph(normalized_method):
        return
    if args.graph_search_k <= 0:
        raise ValueError("--graph-search-k must be greater than 0 for graph methods")
    if args.graph_depth < 0:
        raise ValueError("--graph-depth must be greater than or equal to 0 for graph methods")


def method_needs_reranker(method: str) -> bool:
    return normalize_method(method) in {
        "chroma_bm25_rrf_reranker",
        "chroma_bm25_graph_rrf_reranker",
        "strategy_matrix",
        "strategy_matrix_decompose",
        "strategy_matrix_graph",
        "strategy_matrix_decompose_graph",
    }


def method_uses_decompose(method: str, question: Question) -> bool:
    return normalize_method(method) in {
        "strategy_matrix_decompose",
        "strategy_matrix_decompose_graph",
    } and question.question_type in {"multi_hop", "comparison"}


def should_rerank_question(method: str, question: Question) -> bool:
    normalized = normalize_method(method)
    if normalized in {"chroma_bm25_rrf_reranker", "chroma_bm25_graph_rrf_reranker"}:
        return True
    if normalized in {
        "strategy_matrix",
        "strategy_matrix_decompose",
        "strategy_matrix_graph",
        "strategy_matrix_decompose_graph",
    }:
        return question.question_type in COMPLEX_QUESTION_TYPES
    return False


def should_apply_source_boost(method: str) -> bool:
    return normalize_method(method) in {
        "chroma_bm25_rrf_source_boost",
        "strategy_matrix",
        "strategy_matrix_decompose",
        "strategy_matrix_graph",
        "strategy_matrix_decompose_graph",
    }


class ChildChunkBM25:
    def __init__(
        self,
        records: list[Candidate],
        doc_lengths: list[int],
        inverted_index: dict[str, list[tuple[int, int]]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.records = records
        self.doc_lengths = doc_lengths
        self.inverted_index = inverted_index
        self.k1 = k1
        self.b = b
        self.total_docs = len(records)
        self.average_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)

    @classmethod
    def from_jsonl(cls, path: Path) -> "ChildChunkBM25":
        records: list[Candidate] = []
        doc_lengths: list[int] = []
        inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)

        with path.open("r", encoding="utf-8") as file:
            for doc_index, line in enumerate(file):
                row = json.loads(line)
                text = row.get("text", "")
                tokens = tokenize(text)
                token_counts = Counter(tokens)
                records.append(
                    Candidate(
                        chunk_id=row.get("chunk_id", ""),
                        parent_doc_id=row.get("parent_doc_id", ""),
                        parent_chunk_id=row.get("parent_chunk_id", ""),
                        source_type=row.get("source_type", ""),
                        doc_semantic_type=row.get("doc_semantic_type") or "generic_doc",
                        title=row.get("title", ""),
                        section_heading=row.get("section_heading", ""),
                        text=text,
                    )
                )
                doc_lengths.append(len(tokens))
                for token, term_frequency in token_counts.items():
                    inverted_index[token].append((doc_index, term_frequency))

        return cls(records=records, doc_lengths=doc_lengths, inverted_index=dict(inverted_index))

    def search(self, query: str, k: int) -> list[Candidate]:
        query_terms = Counter(tokenize(query))
        scores: dict[int, float] = defaultdict(float)

        for term, query_frequency in query_terms.items():
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            document_frequency = len(postings)
            idf = math.log(1 + (self.total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
            for doc_index, term_frequency in postings:
                doc_length = self.doc_lengths[doc_index]
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.average_doc_length, 1e-9)
                )
                scores[doc_index] += query_frequency * idf * (term_frequency * (self.k1 + 1)) / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        candidates: list[Candidate] = []
        for rank, (doc_index, score) in enumerate(ranked, start=1):
            record = self.records[doc_index]
            candidates.append(
                Candidate(
                    chunk_id=record.chunk_id,
                    parent_doc_id=record.parent_doc_id,
                    parent_chunk_id=record.parent_chunk_id,
                    source_type=record.source_type,
                    doc_semantic_type=record.doc_semantic_type or "generic_doc",
                    title=record.title,
                    section_heading=record.section_heading,
                    text=record.text,
                    bm25_rank=rank,
                    bm25_score=score,
                )
            )
        return candidates


def load_parent_texts(path: Path) -> dict[str, str]:
    parent_texts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            parent_texts[row["parent_chunk_id"]] = row.get("text", "")
    return parent_texts


def load_doc_semantic_types_by_doc_id(path: Path) -> dict[str, str]:
    doc_semantic_types: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            parent_doc_id = row.get("parent_doc_id")
            if parent_doc_id:
                doc_semantic_types[str(parent_doc_id)] = row.get("doc_semantic_type") or "generic_doc"
    return doc_semantic_types


def chroma_search(
    store: Chroma,
    question: str,
    search_k: int,
    where: dict[str, str] | None,
) -> list[Candidate]:
    results = store.similarity_search_with_score(question, k=search_k, filter=where)
    candidates: list[Candidate] = []
    for rank, (document, score) in enumerate(results, start=1):
        metadata = dict(document.metadata or {})
        candidates.append(
            Candidate(
                chunk_id=metadata.get("chunk_id", ""),
                parent_doc_id=metadata.get("parent_doc_id", ""),
                parent_chunk_id=metadata.get("parent_chunk_id", ""),
                source_type=metadata.get("source_type", ""),
                doc_semantic_type=metadata.get("doc_semantic_type") or "generic_doc",
                title=metadata.get("title", ""),
                section_heading=metadata.get("section_heading", ""),
                text=document.page_content,
                vector_rank=rank,
                vector_score=float(score),
            )
        )
    return candidates


def run_coroutine_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


def candidate_from_backend_document(row: dict[str, Any]) -> Candidate:
    metadata = dict(row.get("metadata") or {})
    chunk_id = metadata.get("chunk_id") or row.get("parent_chunk_id", "")
    evidence_chunk_ids = [chunk_id] if chunk_id else []
    return Candidate(
        chunk_id=chunk_id,
        parent_doc_id=row.get("parent_doc_id", ""),
        parent_chunk_id=row.get("parent_chunk_id", ""),
        source_type=row.get("source_type", metadata.get("source_type", "")),
        title=row.get("title", metadata.get("title", "")),
        section_heading=row.get("section_heading", metadata.get("section_heading", "")),
        text=row.get("child_text") or row.get("parent_text") or "",
        evidence_chunk_ids=evidence_chunk_ids,
        fused_score=float(row.get("score", 0.0) or 0.0),
    )


def graph_search(
    graph_index: Any,
    query: str,
    top_k: int,
    depth: int,
    source_hints: list[str],
) -> list[Candidate]:
    results = graph_index.retrieve_sync(
        query=query,
        top_k=top_k,
        depth=depth,
        source_hints=source_hints,
    )
    candidates: list[Candidate] = []
    for rank, result in enumerate(results, start=1):
        metadata = getattr(result, "metadata", None) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        candidates.append(
            Candidate(
                chunk_id=result.parent_chunk_id,
                parent_doc_id=result.parent_doc_id or "",
                parent_chunk_id=result.parent_chunk_id,
                source_type=result.source_type or "",
                doc_semantic_type=metadata.get("doc_semantic_type")
                or getattr(result, "doc_semantic_type", None)
                or "generic_doc",
                title=result.title or "",
                section_heading=result.section_heading or "",
                text=result.text or "",
                graph_rank=rank,
                graph_score=float(result.score),
                evidence_chunk_ids=[result.parent_chunk_id] if result.parent_chunk_id else [],
            )
        )
    return candidates


def reciprocal_rank(ranked_doc_ids: list[str], expected_doc_ids: set[str], max_k: int) -> float:
    return reciprocal_rank_at_k(ranked_doc_ids, expected_doc_ids, max_k)


def evidence_coverage_at_k(
    ranked_chunk_ids: list[str] | list[set[str]],
    required_evidence_groups: list[list[str]],
    max_k: int,
) -> float:
    if not required_evidence_groups:
        return 0.0
    top_ids: set[str] = set()
    for evidence_ids in ranked_chunk_ids[:max_k]:
        if isinstance(evidence_ids, set):
            top_ids.update(evidence_ids)
        elif evidence_ids:
            top_ids.add(evidence_ids)
    covered = 0
    for group in required_evidence_groups:
        if top_ids.intersection(group):
            covered += 1
    return covered / len(required_evidence_groups)


def decompose_question_for_eval(question: Question) -> list[str]:
    if question.question_type not in {"multi_hop", "comparison"}:
        return [question.question]
    answer_facts = [fact.strip() for fact in question.answer_facts if fact.strip()]
    if answer_facts:
        return answer_facts[:4]
    return [question.question]



def fuse_by_rrf(
    vector_candidates: Iterable[Candidate],
    bm25_candidates: Iterable[Candidate],
    rrf_k: int,
    source_hints: list[str] | None = None,
    source_boost: float = SOURCE_HINT_SOFT_BOOST,
    graph_candidates: Iterable[Candidate] | None = None,
) -> list[Candidate]:
    graph_candidates = list(graph_candidates or [])
    by_candidate_key: dict[str, Candidate] = {}
    keys_by_parent_chunk_id: dict[str, list[str]] = defaultdict(list)

    def create_candidate(candidate: Candidate) -> Candidate:
        return Candidate(
            chunk_id=candidate.chunk_id,
            parent_doc_id=candidate.parent_doc_id,
            parent_chunk_id=candidate.parent_chunk_id,
            source_type=candidate.source_type,
            doc_semantic_type=candidate.doc_semantic_type or "generic_doc",
            title=candidate.title,
            section_heading=candidate.section_heading,
            text=candidate.text,
            evidence_chunk_ids=list(candidate.evidence_chunk_ids),
        )

    def add_parent_key(candidate: Candidate, key: str) -> None:
        if candidate.parent_chunk_id and key not in keys_by_parent_chunk_id[candidate.parent_chunk_id]:
            keys_by_parent_chunk_id[candidate.parent_chunk_id].append(key)

    def merge_child(candidate: Candidate) -> Candidate:
        key = candidate.chunk_id
        if key not in by_candidate_key:
            by_candidate_key[key] = create_candidate(candidate)
            add_parent_key(candidate, key)
        existing = by_candidate_key[key]
        if candidate.vector_rank is not None:
            existing.vector_rank = candidate.vector_rank
            existing.vector_score = candidate.vector_score
            existing.fused_score += 1.0 / (rrf_k + candidate.vector_rank)
        if candidate.bm25_rank is not None:
            existing.bm25_rank = candidate.bm25_rank
            existing.bm25_score = candidate.bm25_score
            existing.fused_score += 1.0 / (rrf_k + candidate.bm25_rank)
        return existing

    def best_existing_rank(candidate: Candidate) -> int:
        ranks = [
            rank
            for rank in (candidate.vector_rank, candidate.bm25_rank, candidate.graph_rank)
            if rank is not None
        ]
        return min(ranks, default=sys.maxsize)

    def representative_for_graph(candidate: Candidate) -> Candidate:
        parent_keys = keys_by_parent_chunk_id.get(candidate.parent_chunk_id, [])
        parent_candidates = [by_candidate_key[key] for key in parent_keys if key in by_candidate_key]
        if parent_candidates:
            return max(
                parent_candidates,
                key=lambda item: (item.fused_score, -best_existing_rank(item)),
            )

        key = candidate.parent_chunk_id or candidate.chunk_id
        if key not in by_candidate_key:
            by_candidate_key[key] = create_candidate(candidate)
            add_parent_key(candidate, key)
        return by_candidate_key[key]

    def merge_graph(candidate: Candidate) -> Candidate:
        existing = representative_for_graph(candidate)
        graph_semantic_type = candidate.doc_semantic_type or "generic_doc"
        if (
            existing.doc_semantic_type in {"", "generic_doc"}
            and graph_semantic_type != "generic_doc"
        ):
            existing.doc_semantic_type = graph_semantic_type
        graph_evidence_ids = [candidate.chunk_id, candidate.parent_chunk_id, *candidate.evidence_chunk_ids]
        for evidence_id in graph_evidence_ids:
            if evidence_id and evidence_id not in existing.evidence_chunk_ids:
                existing.evidence_chunk_ids.append(evidence_id)
        if candidate.graph_rank is not None:
            existing.graph_rank = candidate.graph_rank
            existing.graph_score = candidate.graph_score
            existing.fused_score += 1.0 / (rrf_k + candidate.graph_rank)
        return existing

    for candidate in vector_candidates:
        merge_child(candidate)
    for candidate in bm25_candidates:
        merge_child(candidate)
    for candidate in graph_candidates:
        merge_graph(candidate)

    source_hint_set = {source for source in (source_hints or []) if source}
    if source_hint_set:
        for candidate in by_candidate_key.values():
            if candidate.source_type in source_hint_set:
                candidate.fused_score *= 1.0 + source_boost

    return sorted(by_candidate_key.values(), key=lambda item: item.fused_score, reverse=True)


def dedup_parent_doc_ids(candidates: list[Candidate]) -> tuple[list[str], list[Candidate]]:
    seen: set[str] = set()
    ranked_doc_ids: list[str] = []
    hits: list[Candidate] = []
    for candidate in candidates:
        if not candidate.parent_doc_id or candidate.parent_doc_id in seen:
            continue
        seen.add(candidate.parent_doc_id)
        ranked_doc_ids.append(candidate.parent_doc_id)
        hits.append(candidate)
    return ranked_doc_ids, hits


def ranked_chunk_ids_for_coverage(
    ranked_candidates: list[Candidate],
    parent_candidates: list[Candidate],
    reranker_used: bool,
) -> list[str]:
    if not reranker_used:
        return [candidate.chunk_id for candidate in ranked_candidates if candidate.chunk_id]

    ranked_parent_doc_ids = {candidate.parent_doc_id for candidate in parent_candidates}
    seen_chunk_ids: set[str] = set()
    ranked_chunk_ids: list[str] = []

    for candidate in parent_candidates:
        if candidate.chunk_id:
            ranked_chunk_ids.append(candidate.chunk_id)
            seen_chunk_ids.add(candidate.chunk_id)

    for candidate in ranked_candidates:
        if (
            not candidate.parent_doc_id
            or candidate.parent_doc_id not in ranked_parent_doc_ids
            or not candidate.chunk_id
            or candidate.chunk_id in seen_chunk_ids
        ):
            continue
        ranked_chunk_ids.append(candidate.chunk_id)
        seen_chunk_ids.add(candidate.chunk_id)

    return ranked_chunk_ids


def candidate_evidence_ids(candidate: Candidate) -> set[str]:
    evidence_ids = {candidate.chunk_id, *candidate.evidence_chunk_ids}
    return {evidence_id for evidence_id in evidence_ids if evidence_id}


def ranked_evidence_ids_for_coverage(
    ranked_candidates: list[Candidate],
    parent_candidates: list[Candidate],
    reranker_used: bool,
) -> list[set[str]]:
    if not reranker_used:
        return [candidate_evidence_ids(candidate) for candidate in ranked_candidates if candidate.chunk_id]

    ranked_parent_doc_ids = {candidate.parent_doc_id for candidate in parent_candidates}
    seen_chunk_ids: set[str] = set()
    ranked_evidence_ids: list[set[str]] = []

    for candidate in parent_candidates:
        if candidate.chunk_id:
            ranked_evidence_ids.append(candidate_evidence_ids(candidate))
            seen_chunk_ids.add(candidate.chunk_id)

    for candidate in ranked_candidates:
        if (
            not candidate.parent_doc_id
            or candidate.parent_doc_id not in ranked_parent_doc_ids
            or not candidate.chunk_id
            or candidate.chunk_id in seen_chunk_ids
        ):
            continue
        ranked_evidence_ids.append(candidate_evidence_ids(candidate))
        seen_chunk_ids.add(candidate.chunk_id)

    return ranked_evidence_ids


def maybe_load_reranker(model_path: str, device: str | None, max_length: int) -> Any:
    return Qwen3CausalReranker(
        model_path=model_path,
        device=device,
        max_length=max_length,
    )



def maybe_load_graph_index(method: str, graph_dir: Path) -> Any | None:
    if not method_needs_graph(normalize_method(method)):
        return None

    from app.rag.graph_index_service import GraphIndexService

    graph_index = GraphIndexService(graph_dir)
    graph_index.load_sync()
    validate_graph_index_loaded(graph_dir, graph_index)
    return graph_index


def validate_graph_index_loaded(graph_dir: Path, graph_index: Any) -> None:
    if not graph_dir.exists():
        raise ValueError(f"Graph index directory does not exist: {graph_dir}")

    parent_chunks = getattr(graph_index, "parent_chunks", None) or {}
    entities = getattr(graph_index, "entities", None) or {}
    if not parent_chunks or not entities:
        raise ValueError(
            f"No graph data loaded from {graph_dir}: expected non-empty parent_chunks and entities"
        )


def rerank_parent_candidates(
    model: Any,
    query: str,
    parent_candidates: list[Candidate],
    parent_texts: dict[str, str],
    candidate_k: int,
    batch_size: int,
) -> list[Candidate]:
    rerank_window = parent_candidates[:candidate_k]
    if not rerank_window:
        return parent_candidates

    pairs = []
    for candidate in rerank_window:
        parent_text = parent_texts.get(candidate.parent_chunk_id, candidate.text)
        pairs.append((query, parent_text[:4000]))

    scores = model.predict(pairs, batch_size=batch_size)
    for candidate, score in zip(rerank_window, scores):
        candidate.reranker_score = float(score)

    reranked_window = sorted(rerank_window, key=lambda item: item.reranker_score or 0.0, reverse=True)
    return reranked_window + parent_candidates[candidate_k:]


def candidate_to_hit(candidate: Candidate, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "parent_doc_id": candidate.parent_doc_id,
        "parent_chunk_id": candidate.parent_chunk_id,
        "chunk_id": candidate.chunk_id,
        "source_type": candidate.source_type,
        "doc_semantic_type": candidate.doc_semantic_type or "generic_doc",
        "title": candidate.title,
        "section_heading": candidate.section_heading,
        "vector_rank": candidate.vector_rank,
        "vector_score": candidate.vector_score,
        "bm25_rank": candidate.bm25_rank,
        "bm25_score": candidate.bm25_score,
        "graph_rank": candidate.graph_rank,
        "graph_score": candidate.graph_score,
        "evidence_chunk_ids": candidate.evidence_chunk_ids,
        "fused_score": candidate.fused_score,
        "reranker_score": candidate.reranker_score,
        "preview": candidate.text[:240].replace("\n", " "),
    }


def evaluate_question(
    method: str,
    store: Chroma | None,
    bm25: ChildChunkBM25 | None,
    question: Question,
    chroma_search_k: int,
    bm25_search_k: int,
    rrf_k: int,
    source_boost: float,
    k_values: list[int],
    where: dict[str, str] | None,
    reranker_model: Any | None,
    parent_texts: dict[str, str],
    reranker_candidate_k: int,
    reranker_batch_size: int,
    graph_index: Any | None = None,
    graph_search_k: int = 0,
    graph_depth: int = 1,
    backend: str = "chroma",
    retrieval_backend: Any | None = None,
    metadata_filter: MetadataFilterDecision | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    normalized_method = normalize_method(method)
    vector_candidates: list[Candidate] = []
    bm25_candidates: list[Candidate] = []
    graph_candidates: list[Candidate] = []
    graph_source_hints = question.source_types if should_apply_source_boost(normalized_method) else []

    if backend == "elasticsearch":
        if retrieval_backend is None:
            raise ValueError("Elasticsearch backend is selected, but retrieval_backend is not initialized")
        dense_top_k = chroma_search_k if method_needs_chroma(normalized_method) else 0
        lexical_top_k = bm25_search_k if method_needs_bm25(normalized_method) else 0
        raw = run_coroutine_sync(
            retrieval_backend.retrieve_with_details(
                query=question.question,
                final_top_k=max(k_values),
                dense_top_k=dense_top_k,
                bm25_top_k=lexical_top_k,
                fusion_top_k=max(dense_top_k, lexical_top_k, max(k_values)),
                source_hints=question.source_types if should_apply_source_boost(normalized_method) else None,
                use_reranker=False,
                metadata_filter=metadata_filter,
            )
        )
        ranked_candidates = [candidate_from_backend_document(row) for row in raw["selected_documents"]]
        reranker_model = None
    elif method_needs_graph(normalized_method) and graph_index is None:
        raise ValueError(f"Method {method} requires graph index, but graph_index is not initialized")

    if backend == "chroma" and method_uses_decompose(normalized_method, question):
        if store is None:
            raise ValueError(f"Method {method} requires Chroma, but store is not initialized")
        if bm25 is None:
            raise ValueError(f"Method {method} requires BM25, but index is not initialized")

        merged_by_candidate_key: dict[str, Candidate] = {}
        for sub_query in decompose_question_for_eval(question):
            sub_vector_candidates = chroma_search(store, sub_query, chroma_search_k, where)
            sub_bm25_candidates = bm25.search(sub_query, bm25_search_k)
            sub_graph_candidates = []
            if method_needs_graph(normalized_method):
                sub_graph_candidates = graph_search(
                    graph_index=graph_index,
                    query=sub_query,
                    top_k=graph_search_k,
                    depth=graph_depth,
                    source_hints=graph_source_hints,
                )
            vector_candidates.extend(sub_vector_candidates)
            bm25_candidates.extend(sub_bm25_candidates)
            graph_candidates.extend(sub_graph_candidates)
            sub_ranked_candidates = fuse_by_rrf(
                vector_candidates=sub_vector_candidates,
                bm25_candidates=sub_bm25_candidates,
                rrf_k=rrf_k,
                source_hints=graph_source_hints,
                source_boost=source_boost,
                graph_candidates=sub_graph_candidates,
            )
            for candidate in sub_ranked_candidates:
                if not candidate.chunk_id:
                    continue
                existing = merged_by_candidate_key.get(candidate.chunk_id)
                if existing is None or candidate.fused_score > existing.fused_score:
                    merged_by_candidate_key[candidate.chunk_id] = candidate
        ranked_candidates = sorted(
            merged_by_candidate_key.values(),
            key=lambda item: item.fused_score,
            reverse=True,
        )
    elif backend == "chroma":
        if method_needs_chroma(normalized_method):
            if store is None:
                raise ValueError(f"Method {method} requires Chroma, but store is not initialized")
            vector_candidates = chroma_search(store, question.question, chroma_search_k, where)

        if method_needs_bm25(normalized_method):
            if bm25 is None:
                raise ValueError(f"Method {method} requires BM25, but index is not initialized")
            bm25_candidates = bm25.search(question.question, bm25_search_k)

        if method_needs_graph(normalized_method):
            graph_candidates = graph_search(
                graph_index=graph_index,
                query=question.question,
                top_k=graph_search_k,
                depth=graph_depth,
                source_hints=graph_source_hints,
            )

        if normalized_method == "chroma_only":
            ranked_candidates = vector_candidates
        elif normalized_method == "bm25_only":
            ranked_candidates = bm25_candidates
        else:
            ranked_candidates = fuse_by_rrf(
                vector_candidates=vector_candidates,
                bm25_candidates=bm25_candidates,
                rrf_k=rrf_k,
                source_hints=question.source_types if should_apply_source_boost(normalized_method) else None,
                source_boost=source_boost,
                graph_candidates=graph_candidates,
            )

    ranked_doc_ids, parent_candidates = dedup_parent_doc_ids(ranked_candidates)
    reranker_used = reranker_model is not None and should_rerank_question(normalized_method, question)

    if reranker_used:
        parent_candidates = rerank_parent_candidates(
            model=reranker_model,
            query=question.question,
            parent_candidates=parent_candidates,
            parent_texts=parent_texts,
            candidate_k=reranker_candidate_k,
            batch_size=reranker_batch_size,
        )
        ranked_doc_ids = [candidate.parent_doc_id for candidate in parent_candidates]

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    expected = set(question.expected_doc_ids)
    max_k = max(k_values)
    ranked_chunk_ids = ranked_chunk_ids_for_coverage(
        ranked_candidates=ranked_candidates,
        parent_candidates=parent_candidates,
        reranker_used=reranker_used,
    )
    ranked_evidence_ids = ranked_evidence_ids_for_coverage(
        ranked_candidates=ranked_candidates,
        parent_candidates=parent_candidates,
        reranker_used=reranker_used,
    )
    required_evidence_groups_count = len(question.required_evidence_groups)

    detail: dict[str, Any] = {
        "method": method,
        "normalized_method": normalized_method,
        "question_id": question.question_id,
        "question_type": question.question_type,
        "source_types": question.source_types,
        "question": question.question,
        "expected_doc_ids": question.expected_doc_ids,
        "retrieved_parent_doc_ids": ranked_doc_ids,
        "retrieved_chunk_ids": ranked_chunk_ids[:max_k],
        "required_evidence_groups_count": required_evidence_groups_count,
        "evidence_coverage": evidence_coverage_at_k(
            ranked_evidence_ids,
            question.required_evidence_groups,
            max_k,
        ),
        "top_hits": [
            candidate_to_hit(candidate, rank)
            for rank, candidate in enumerate(parent_candidates[: max(k_values)], start=1)
        ],
        "latency_ms": elapsed_ms,
        "vector_child_results": len(vector_candidates),
        "bm25_child_results": len(bm25_candidates),
        "graph_results": len(graph_candidates),
        "fused_child_results": len(ranked_candidates),
        "dedup_parent_results": len(ranked_doc_ids),
        "source_boost_applied": should_apply_source_boost(normalized_method),
        "reranker_used": reranker_used,
    }

    for k in k_values:
        retrieved_at_k = set(ranked_doc_ids[:k])
        matched = sorted(expected.intersection(retrieved_at_k))
        precision = precision_at_k(ranked_doc_ids, expected, k)
        recall = recall_at_k(ranked_doc_ids, expected, k)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        detail[f"hit@{k}"] = 1 if matched else 0
        detail[f"precision@{k}"] = precision
        detail[f"recall@{k}"] = recall
        detail[f"f1@{k}"] = f1
        detail[f"ndcg@{k}"] = ndcg_at_k(ranked_doc_ids, expected, k)
        detail[f"ap@{k}"] = average_precision_at_k(ranked_doc_ids, expected, k)
        detail[f"matched_doc_ids@{k}"] = matched
        detail[f"evidence_coverage@{k}"] = evidence_coverage_at_k(
            ranked_evidence_ids,
            question.required_evidence_groups,
            k,
        )

    detail[f"rr@{max_k}"] = reciprocal_rank(ranked_doc_ids, expected, max_k)
    return detail


def build_doc_semantic_type_summary(
    details: Iterable[dict[str, Any]],
    k_values: Iterable[int],
) -> dict[str, dict[str, float | int]]:
    """Average retrieval metrics against type-specific expected document sets."""
    type_specific_rows: list[dict[str, Any]] = []
    k_list = list(k_values)
    max_k = max(k_list) if k_list else 0

    for detail in details:
        expected_type_by_doc_id = detail.get("expected_doc_semantic_type_by_doc_id") or {}
        if not isinstance(expected_type_by_doc_id, dict):
            expected_type_by_doc_id = {}
        if not expected_type_by_doc_id:
            expected_types = detail.get("expected_doc_semantic_types") or []
            if len(expected_types) == 1:
                expected_type_by_doc_id = {
                    expected_doc_id: expected_types[0]
                    for expected_doc_id in detail.get("expected_doc_ids", [])
                }

        expected_doc_ids_by_type: dict[str, set[str]] = defaultdict(set)
        for expected_doc_id in detail.get("expected_doc_ids", []):
            semantic_type = expected_type_by_doc_id.get(expected_doc_id, "generic_doc")
            expected_doc_ids_by_type[str(semantic_type or "generic_doc")].add(expected_doc_id)

        ranked_doc_ids = list(detail.get("retrieved_parent_doc_ids") or [])
        for semantic_type, expected_doc_ids in expected_doc_ids_by_type.items():
            row: dict[str, Any] = {
                "doc_semantic_type": semantic_type,
                "latency_ms": detail.get("latency_ms", 0.0),
                "required_evidence_groups_count": 0,
            }
            for k in k_list:
                retrieved_at_k = set(ranked_doc_ids[:k])
                matched = expected_doc_ids.intersection(retrieved_at_k)
                precision = precision_at_k(ranked_doc_ids, expected_doc_ids, k)
                recall = recall_at_k(ranked_doc_ids, expected_doc_ids, k)
                row[f"hit@{k}"] = 1 if matched else 0
                row[f"precision@{k}"] = precision
                row[f"recall@{k}"] = recall
                row[f"f1@{k}"] = (
                    0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                )
                row[f"ndcg@{k}"] = ndcg_at_k(ranked_doc_ids, expected_doc_ids, k)
                row[f"ap@{k}"] = average_precision_at_k(ranked_doc_ids, expected_doc_ids, k)
                row[f"evidence_coverage@{k}"] = 0.0
            if max_k:
                row[f"rr@{max_k}"] = reciprocal_rank_at_k(ranked_doc_ids, expected_doc_ids, max_k)
            type_specific_rows.append(row)

    return build_group_summary(type_specific_rows, k_list, "doc_semantic_type")


def summarize(details: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    total = len(details)
    evidence_coverage_details = [
        item for item in details if item.get("required_evidence_groups_count", 0) > 0
    ]
    evidence_coverage_questions = len(evidence_coverage_details)
    summary: dict[str, Any] = {
        "questions": total,
        "k_values": k_values,
        "average_latency_ms": round(
            sum(item["latency_ms"] for item in details) / max(total, 1),
            2,
        ),
        "evidence_coverage": round(
            sum(item.get("evidence_coverage", 0.0) for item in evidence_coverage_details)
            / max(evidence_coverage_questions, 1),
            4,
        ),
        "evidence_coverage_questions": evidence_coverage_questions,
    }

    for k in k_values:
        summary[f"hit@{k}"] = round(
            sum(item[f"hit@{k}"] for item in details) / max(total, 1),
            4,
        )
        summary[f"precision@{k}"] = round(
            sum(item[f"precision@{k}"] for item in details) / max(total, 1),
            4,
        )
        summary[f"recall@{k}"] = round(
            sum(item[f"recall@{k}"] for item in details) / max(total, 1),
            4,
        )
        summary[f"f1@{k}"] = round(
            sum(item[f"f1@{k}"] for item in details) / max(total, 1),
            4,
        )
        summary[f"ndcg@{k}"] = round(
            sum(item.get(f"ndcg@{k}", 0.0) for item in details) / max(total, 1),
            4,
        )
        summary[f"map@{k}"] = round(
            sum(item.get(f"ap@{k}", 0.0) for item in details) / max(total, 1),
            4,
        )
        summary[f"evidence_coverage@{k}"] = round(
            sum(item.get(f"evidence_coverage@{k}", 0.0) for item in evidence_coverage_details)
            / max(evidence_coverage_questions, 1),
            4,
        )

    max_mrr_k = max(k_values)
    summary[f"mrr@{max_mrr_k}"] = round(
        sum(item[f"rr@{max_mrr_k}"] for item in details) / max(total, 1),
        4,
    )
    summary["question_type_summary"] = build_question_type_summary(details, k_values)
    summary["doc_semantic_type_summary"] = build_doc_semantic_type_summary(details, k_values)
    return summary


def is_low_precision_failure(row: dict[str, Any], max_k: int) -> bool:
    precision = row.get(f"precision@{max_k}", 0.0)
    expected_count = len(row.get("expected_doc_ids") or [])
    if expected_count > 0:
        max_possible = min(expected_count, max_k) / max_k
        return max_possible > 0 and precision / max_possible < 0.5
    return precision < 0.2


def classify_failures(row: dict[str, Any], k_values: list[int]) -> list[str]:
    max_k = max(k_values)
    reasons: list[str] = []
    if row["dedup_parent_results"] == 0:
        return ["no_candidates"]
    if row[f"hit@{max_k}"] == 0:
        reasons.append("no_gold_hit")
    if row.get(f"recall@{max_k}", 0.0) < 0.5:
        reasons.append("low_recall")
    if is_low_precision_failure(row, max_k):
        reasons.append("low_precision")
    if row.get(f"ndcg@{max_k}", 0.0) < 0.5:
        reasons.append("low_ndcg")
    if row.get(f"ap@{max_k}", 0.0) < 0.5:
        reasons.append("low_map")
    if row.get("required_evidence_groups_count", 0) > 0 and row.get(f"evidence_coverage@{max_k}", 0.0) < 1.0:
        reasons.append("evidence_group_missing")
    if row.get("reranker_used") and row.get("hit@1") == 0:
        reasons.append("reranker_top1_miss")
    return reasons


def build_failure_rows(details: list[dict[str, Any]], k_values: list[int]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    max_k = max(k_values)
    for row in details:
        failure_reasons = classify_failures(row, k_values)
        if not failure_reasons:
            continue
        failures.append(
            {
                "method": row["method"],
                "normalized_method": row["normalized_method"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "source_types": row["source_types"],
                "question": row["question"],
                "expected_doc_ids": row["expected_doc_ids"],
                "retrieved_parent_doc_ids": row["retrieved_parent_doc_ids"][:max_k],
                "matched_doc_ids": row[f"matched_doc_ids@{max_k}"],
                "failure_reasons": failure_reasons,
                "reranker_used": row.get("reranker_used", False),
                "source_boost_applied": row.get("source_boost_applied", False),
                "latency_ms": row["latency_ms"],
                "top_hits": row["top_hits"],
            }
        )
    return failures


def write_outputs(
    output_dir: Path,
    method: str,
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    k_values: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{method}_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    details_path = output_dir / f"{method}_details.jsonl"
    with details_path.open("w", encoding="utf-8") as file:
        for row in details:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    failures = build_failure_rows(details, k_values)
    failures_path = output_dir / f"{method}_failures.jsonl"
    with failures_path.open("w", encoding="utf-8") as file:
        for row in failures:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / f"{method}_details.csv"
    fieldnames = [
        "method",
        "normalized_method",
        "question_id",
        "question_type",
        "source_types",
        "expected_doc_ids",
        "top_parent_doc_ids",
        "latency_ms",
        "dedup_parent_results",
        "required_evidence_groups_count",
        "evidence_coverage",
        "reranker_used",
        "source_boost_applied",
    ]
    for k in k_values:
        fieldnames.extend(
            [
                f"hit@{k}",
                f"precision@{k}",
                f"recall@{k}",
                f"f1@{k}",
                f"ndcg@{k}",
                f"ap@{k}",
                f"evidence_coverage@{k}",
                f"matched_doc_ids@{k}",
            ]
        )
    fieldnames.append(f"rr@{max(k_values)}")

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            csv_row = {
                "method": row["method"],
                "normalized_method": row["normalized_method"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "source_types": "|".join(row["source_types"]),
                "expected_doc_ids": "|".join(row["expected_doc_ids"]),
                "top_parent_doc_ids": "|".join(row["retrieved_parent_doc_ids"][: max(k_values)]),
                "latency_ms": row["latency_ms"],
                "dedup_parent_results": row["dedup_parent_results"],
                "required_evidence_groups_count": row.get("required_evidence_groups_count", 0),
                "evidence_coverage": row.get("evidence_coverage", 0.0),
                "reranker_used": row["reranker_used"],
                "source_boost_applied": row["source_boost_applied"],
                f"rr@{max(k_values)}": row[f"rr@{max(k_values)}"],
            }
            for k in k_values:
                csv_row[f"hit@{k}"] = row[f"hit@{k}"]
                csv_row[f"precision@{k}"] = row[f"precision@{k}"]
                csv_row[f"recall@{k}"] = row[f"recall@{k}"]
                csv_row[f"f1@{k}"] = row[f"f1@{k}"]
                csv_row[f"ndcg@{k}"] = row.get(f"ndcg@{k}", 0.0)
                csv_row[f"ap@{k}"] = row.get(f"ap@{k}", 0.0)
                csv_row[f"evidence_coverage@{k}"] = row.get(f"evidence_coverage@{k}", 0.0)
                csv_row[f"matched_doc_ids@{k}"] = "|".join(row[f"matched_doc_ids@{k}"])
            writer.writerow(csv_row)


def write_standard_retrieval_outputs(
    args: argparse.Namespace,
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    k_values: list[int],
    metadata_filter: MetadataFilterDecision,
) -> Path:
    run_dir = args.run_output_root / utc_run_id("retrieval")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "git_commit": short_git_commit(),
        "dataset_path": str(args.questions_path),
        "question_count": len(details),
        "method": args.method,
        "backend": args.backend,
        "k_values": k_values,
        "embedding_model": args.embedding_model,
        "collection_name": args.collection_name,
        "persist_dir": str(args.persist_dir),
        "reranker_model_path": args.reranker_model_path if args.backend == "chroma" and method_needs_reranker(normalize_method(args.method)) else None,
        "reranker_candidate_k": args.reranker_candidate_k if args.backend == "chroma" and method_needs_reranker(normalize_method(args.method)) else None,
        "graph_dir": str(args.graph_dir) if args.backend == "chroma" and method_needs_graph(normalize_method(args.method)) else None,
        "graph_search_k": args.graph_search_k if args.backend == "chroma" and method_needs_graph(normalize_method(args.method)) else None,
        "graph_depth": args.graph_depth if args.backend == "chroma" and method_needs_graph(normalize_method(args.method)) else None,
        "metadata_filter": metadata_filter.model_dump(),
        "judge_provider": None,
        "judge_model": None,
        "ci": args.ci,
        "fail_on_regression": args.fail_on_regression,
    }

    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    with (run_dir / "retrieval_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with (run_dir / "retrieval_details.jsonl").open("w", encoding="utf-8") as file:
        for row in details:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (run_dir / "retrieval_failures.jsonl").open("w", encoding="utf-8") as file:
        for row in failures:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    baseline_summary = load_json_if_exists(args.baseline_dir / "retrieval_summary.json")
    report = render_retrieval_report(config, summary, failures, baseline_summary)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return run_dir


def main() -> None:
    args = parse_args()
    normalized_method = normalize_method(args.method)
    validate_graph_cli_args(args, normalized_method)
    k_values = parse_k_values(args.k_values)
    metadata_filter = build_metadata_filter_decision(
        args.metadata_filter_mode,
        args.filter_source_types,
        args.filter_doc_semantic_types,
    )
    if args.backend != "elasticsearch" and args.metadata_filter_mode != "none":
        raise ValueError("Metadata filter evaluation flags are only supported with --backend elasticsearch")
    questions = load_questions(args.questions_path, args.limit)
    doc_semantic_types_by_doc_id = load_doc_semantic_types_by_doc_id(args.parent_chunks_path)
    if not questions:
        raise ValueError(f"No questions loaded from {args.questions_path}")
    if args.backend == "elasticsearch":
        if args.where_source_type:
            raise ValueError("--where-source-type is not supported for Elasticsearch evaluation yet")
        if method_needs_graph(normalized_method):
            raise ValueError("Graph evaluation methods are not supported for Elasticsearch evaluation yet")
        if method_needs_reranker(normalized_method):
            raise ValueError("Reranker evaluation methods are not supported for Elasticsearch evaluation yet")
        if any(method_uses_decompose(normalized_method, question) for question in questions):
            raise ValueError("Decomposition evaluation methods are not supported for Elasticsearch evaluation yet")

    retrieval_backend = None
    if args.backend == "elasticsearch":
        from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend
        from app.rag.retrieval_backends.factory import load_rag_config

        rag_config = load_rag_config()
        rag_config["text_embedding_model_name"] = args.embedding_model
        rag_config.setdefault("elasticsearch", {})["ollama_base_url"] = args.ollama_base_url
        retrieval_backend = ElasticsearchEnterpriseRetrievalBackend.from_config(rag_config)
        retrieval_backend.rrf_k = args.rrf_k
        retrieval_backend.source_hint_soft_boost = args.source_boost

    bm25 = None
    if args.backend == "chroma" and method_needs_bm25(normalized_method):
        print(f"Loading BM25 child chunks from {args.child_chunks_path} ...", flush=True)
        bm25 = ChildChunkBM25.from_jsonl(args.child_chunks_path)
        print(f"Loaded {len(bm25.records)} child chunks for BM25.", flush=True)

    store = None
    if args.backend == "chroma" and method_needs_chroma(normalized_method):
        embeddings = OllamaEmbeddings(
            model=args.embedding_model,
            base_url=args.ollama_base_url,
        )
        store = Chroma(
            collection_name=args.collection_name,
            embedding_function=embeddings,
            persist_directory=str(args.persist_dir.resolve()),
        )
    where = {"source_type": args.where_source_type} if args.where_source_type else None

    reranker_model = None
    parent_texts: dict[str, str] = {}
    if args.backend == "chroma" and method_needs_reranker(normalized_method):
        print(f"Loading parent chunks from {args.parent_chunks_path} ...", flush=True)
        parent_texts = load_parent_texts(args.parent_chunks_path)
        print(f"Loading reranker from {args.reranker_model_path} ...", flush=True)
        reranker_model = maybe_load_reranker(
            model_path=args.reranker_model_path,
            device=args.reranker_device,
            max_length=args.reranker_max_length,
        )

    graph_index = None
    if args.backend == "chroma" and method_needs_graph(normalized_method):
        print(f"Loading graph index from {args.graph_dir} ...", flush=True)
        graph_index = maybe_load_graph_index(normalized_method, args.graph_dir)
        print(
            f"Loaded graph index with {len(graph_index.entities)} entities and {len(graph_index.relations)} relations.",
            flush=True,
        )

    started = time.perf_counter()
    details: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        detail = evaluate_question(
            method=args.method,
            store=store,
            bm25=bm25,
            question=question,
            chroma_search_k=args.chroma_search_k,
            bm25_search_k=args.bm25_search_k,
            rrf_k=args.rrf_k,
            source_boost=args.source_boost,
            k_values=k_values,
            where=where,
            reranker_model=reranker_model,
            parent_texts=parent_texts,
            reranker_candidate_k=args.reranker_candidate_k,
            reranker_batch_size=args.reranker_batch_size,
            graph_index=graph_index,
            graph_search_k=args.graph_search_k,
            graph_depth=args.graph_depth,
            backend=args.backend,
            retrieval_backend=retrieval_backend,
            metadata_filter=metadata_filter,
        )
        detail["expected_doc_semantic_type_by_doc_id"] = {
            expected_doc_id: doc_semantic_types_by_doc_id.get(expected_doc_id, "generic_doc")
            for expected_doc_id in question.expected_doc_ids
        }
        detail["expected_doc_semantic_types"] = sorted(
            set(detail["expected_doc_semantic_type_by_doc_id"].values())
        )
        details.append(detail)
        print(
            f"[{index}/{len(questions)}] {question.question_id} "
            f"hit@{max(k_values)}={detail[f'hit@{max(k_values)}']} "
            f"latency_ms={detail['latency_ms']}",
            flush=True,
        )

    summary = summarize(details, k_values)
    failures = build_failure_rows(details, k_values)
    summary.update(
        {
            "method": args.method,
            "backend": args.backend,
            "normalized_method": normalized_method,
            "collection_name": args.collection_name,
            "persist_dir": str(args.persist_dir.resolve()),
            "embedding_model": args.embedding_model,
            "chroma_search_k_child_chunks": args.chroma_search_k,
            "bm25_search_k_child_chunks": args.bm25_search_k,
            "graph_dir": str(args.graph_dir.resolve()) if graph_index is not None else None,
            "graph_search_k": args.graph_search_k if graph_index is not None else None,
            "graph_depth": args.graph_depth if graph_index is not None else None,
            "rrf_k": args.rrf_k,
            "source_boost": args.source_boost if should_apply_source_boost(normalized_method) else None,
            "where_source_type": args.where_source_type,
            "reranker_model_path": args.reranker_model_path if reranker_model is not None else None,
            "reranker_candidate_k": args.reranker_candidate_k if reranker_model is not None else None,
            "reranker_batch_size": args.reranker_batch_size if reranker_model is not None else None,
            "reranker_complex_question_types": sorted(COMPLEX_QUESTION_TYPES)
            if normalized_method in {
                "strategy_matrix",
                "strategy_matrix_decompose",
                "strategy_matrix_graph",
                "strategy_matrix_decompose_graph",
            }
            else None,
            "failures": len(failures),
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }
    )
    write_outputs(args.output_dir, args.method, summary, details, k_values)
    if args.standard_output:
        run_dir = write_standard_retrieval_outputs(args, summary, details, failures, k_values, metadata_filter)
        print(f"Wrote standard retrieval outputs to {run_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
