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
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "questions.jsonl"
DEFAULT_CHILD_CHUNKS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "child_chunks_parent_child.jsonl"
DEFAULT_PARENT_CHUNKS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "parent_chunks_parent_child.jsonl"
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
    "strategy_matrix",
    "strategy_matrix_decompose",
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
    title: str
    section_heading: str
    text: str
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
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
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--source-boost", type=float, default=SOURCE_HINT_SOFT_BOOST)
    parser.add_argument("--k-values", default="1,5,10,20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--where-source-type",
        default=None,
        help="Optional single source_type metadata filter. Keep empty for comparable experiments.",
    )
    parser.add_argument("--reranker-model-path", default=os.getenv("RERANKER_MODEL_PATH", DEFAULT_RERANKER_MODEL_PATH))
    parser.add_argument("--reranker-device", default=None, help="Optional device override, for example cpu or cuda.")
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-candidate-k", type=int, default=20)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
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


def normalize_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def method_needs_chroma(method: str) -> bool:
    return normalize_method(method) != "bm25_only"


def method_needs_bm25(method: str) -> bool:
    return normalize_method(method) != "chroma_only"


def method_needs_reranker(method: str) -> bool:
    return normalize_method(method) in {
        "chroma_bm25_rrf_reranker",
        "strategy_matrix",
        "strategy_matrix_decompose",
    }


def should_rerank_question(method: str, question: Question) -> bool:
    normalized = normalize_method(method)
    if normalized == "chroma_bm25_rrf_reranker":
        return True
    if normalized in {"strategy_matrix", "strategy_matrix_decompose"}:
        return question.question_type in COMPLEX_QUESTION_TYPES
    return False


def should_apply_source_boost(method: str) -> bool:
    return normalize_method(method) in {
        "chroma_bm25_rrf_source_boost",
        "strategy_matrix",
        "strategy_matrix_decompose",
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
                title=metadata.get("title", ""),
                section_heading=metadata.get("section_heading", ""),
                text=document.page_content,
                vector_rank=rank,
                vector_score=float(score),
            )
        )
    return candidates


def reciprocal_rank(ranked_doc_ids: list[str], expected_doc_ids: set[str], max_k: int) -> float:
    for index, doc_id in enumerate(ranked_doc_ids[:max_k], start=1):
        if doc_id in expected_doc_ids:
            return 1.0 / index
    return 0.0


def evidence_coverage_at_k(
    ranked_chunk_ids: list[str],
    required_evidence_groups: list[list[str]],
    max_k: int,
) -> float:
    if not required_evidence_groups:
        return 0.0
    top_ids = set(ranked_chunk_ids[:max_k])
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
) -> list[Candidate]:
    by_chunk_id: dict[str, Candidate] = {}

    def merge(candidate: Candidate) -> Candidate:
        key = candidate.chunk_id
        if key not in by_chunk_id:
            by_chunk_id[key] = Candidate(
                chunk_id=candidate.chunk_id,
                parent_doc_id=candidate.parent_doc_id,
                parent_chunk_id=candidate.parent_chunk_id,
                source_type=candidate.source_type,
                title=candidate.title,
                section_heading=candidate.section_heading,
                text=candidate.text,
            )
        existing = by_chunk_id[key]
        if candidate.vector_rank is not None:
            existing.vector_rank = candidate.vector_rank
            existing.vector_score = candidate.vector_score
            existing.fused_score += 1.0 / (rrf_k + candidate.vector_rank)
        if candidate.bm25_rank is not None:
            existing.bm25_rank = candidate.bm25_rank
            existing.bm25_score = candidate.bm25_score
            existing.fused_score += 1.0 / (rrf_k + candidate.bm25_rank)
        return existing

    for candidate in vector_candidates:
        merge(candidate)
    for candidate in bm25_candidates:
        merge(candidate)

    source_hint_set = {source for source in (source_hints or []) if source}
    if source_hint_set:
        for candidate in by_chunk_id.values():
            if candidate.source_type in source_hint_set:
                candidate.fused_score *= 1.0 + source_boost

    return sorted(by_chunk_id.values(), key=lambda item: item.fused_score, reverse=True)


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


def maybe_load_reranker(model_path: str, device: str | None, max_length: int) -> Any:
    return Qwen3CausalReranker(
        model_path=model_path,
        device=device,
        max_length=max_length,
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
        "title": candidate.title,
        "section_heading": candidate.section_heading,
        "vector_rank": candidate.vector_rank,
        "vector_score": candidate.vector_score,
        "bm25_rank": candidate.bm25_rank,
        "bm25_score": candidate.bm25_score,
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
) -> dict[str, Any]:
    started = time.perf_counter()
    normalized_method = normalize_method(method)
    vector_candidates: list[Candidate] = []
    bm25_candidates: list[Candidate] = []

    if normalized_method == "strategy_matrix_decompose" and question.question_type in {"multi_hop", "comparison"}:
        if store is None:
            raise ValueError(f"Method {method} requires Chroma, but store is not initialized")
        if bm25 is None:
            raise ValueError(f"Method {method} requires BM25, but index is not initialized")

        merged_by_chunk_id: dict[str, Candidate] = {}
        for sub_query in decompose_question_for_eval(question):
            sub_vector_candidates = chroma_search(store, sub_query, chroma_search_k, where)
            sub_bm25_candidates = bm25.search(sub_query, bm25_search_k)
            vector_candidates.extend(sub_vector_candidates)
            bm25_candidates.extend(sub_bm25_candidates)
            sub_ranked_candidates = fuse_by_rrf(
                vector_candidates=sub_vector_candidates,
                bm25_candidates=sub_bm25_candidates,
                rrf_k=rrf_k,
                source_hints=question.source_types,
                source_boost=source_boost,
            )
            for candidate in sub_ranked_candidates:
                if not candidate.chunk_id:
                    continue
                existing = merged_by_chunk_id.get(candidate.chunk_id)
                if existing is None or candidate.fused_score > existing.fused_score:
                    merged_by_chunk_id[candidate.chunk_id] = candidate
        ranked_candidates = sorted(
            merged_by_chunk_id.values(),
            key=lambda item: item.fused_score,
            reverse=True,
        )
    else:
        if method_needs_chroma(normalized_method):
            if store is None:
                raise ValueError(f"Method {method} requires Chroma, but store is not initialized")
            vector_candidates = chroma_search(store, question.question, chroma_search_k, where)

        if method_needs_bm25(normalized_method):
            if bm25 is None:
                raise ValueError(f"Method {method} requires BM25, but index is not initialized")
            bm25_candidates = bm25.search(question.question, bm25_search_k)

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
            ranked_chunk_ids,
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
        "fused_child_results": len(ranked_candidates),
        "dedup_parent_results": len(ranked_doc_ids),
        "source_boost_applied": should_apply_source_boost(normalized_method),
        "reranker_used": reranker_used,
    }

    for k in k_values:
        retrieved_at_k = set(ranked_doc_ids[:k])
        matched = sorted(expected.intersection(retrieved_at_k))
        precision = len(matched) / k
        recall = len(matched) / max(len(expected), 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        detail[f"hit@{k}"] = 1 if matched else 0
        detail[f"precision@{k}"] = precision
        detail[f"recall@{k}"] = recall
        detail[f"f1@{k}"] = f1
        detail[f"matched_doc_ids@{k}"] = matched
        detail[f"evidence_coverage@{k}"] = evidence_coverage_at_k(
            ranked_chunk_ids,
            question.required_evidence_groups,
            k,
        )

    detail[f"rr@{max_k}"] = reciprocal_rank(ranked_doc_ids, expected, max_k)
    return detail


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
    return summary


def classify_failure(row: dict[str, Any], k_values: list[int]) -> str | None:
    max_k = max(k_values)
    if row["dedup_parent_results"] == 0:
        return "no_candidates"
    if row[f"hit@{max_k}"] == 0:
        return "missed_all_gold"
    if row.get("reranker_used") and row.get(f"hit@1") == 0:
        return "reranker_top1_not_gold"
    if row.get("source_boost_applied") and row.get(f"hit@1") == 0:
        return "source_boost_top1_not_gold"
    if row.get(f"hit@1") == 0:
        return "gold_rank_too_low"
    return None


def build_failure_rows(details: list[dict[str, Any]], k_values: list[int]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    max_k = max(k_values)
    for row in details:
        failure_type = classify_failure(row, k_values)
        if failure_type is None:
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
                "failure_type": failure_type,
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
                csv_row[f"evidence_coverage@{k}"] = row.get(f"evidence_coverage@{k}", 0.0)
                csv_row[f"matched_doc_ids@{k}"] = "|".join(row[f"matched_doc_ids@{k}"])
            writer.writerow(csv_row)


def main() -> None:
    args = parse_args()
    normalized_method = normalize_method(args.method)
    k_values = parse_k_values(args.k_values)
    questions = load_questions(args.questions_path, args.limit)
    if not questions:
        raise ValueError(f"No questions loaded from {args.questions_path}")

    bm25 = None
    if method_needs_bm25(normalized_method):
        print(f"Loading BM25 child chunks from {args.child_chunks_path} ...", flush=True)
        bm25 = ChildChunkBM25.from_jsonl(args.child_chunks_path)
        print(f"Loaded {len(bm25.records)} child chunks for BM25.", flush=True)

    store = None
    if method_needs_chroma(normalized_method):
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
    if method_needs_reranker(normalized_method):
        print(f"Loading parent chunks from {args.parent_chunks_path} ...", flush=True)
        parent_texts = load_parent_texts(args.parent_chunks_path)
        print(f"Loading reranker from {args.reranker_model_path} ...", flush=True)
        reranker_model = maybe_load_reranker(
            model_path=args.reranker_model_path,
            device=args.reranker_device,
            max_length=args.reranker_max_length,
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
            "normalized_method": normalized_method,
            "collection_name": args.collection_name,
            "persist_dir": str(args.persist_dir.resolve()),
            "embedding_model": args.embedding_model,
            "chroma_search_k_child_chunks": args.chroma_search_k,
            "bm25_search_k_child_chunks": args.bm25_search_k,
            "rrf_k": args.rrf_k,
            "source_boost": args.source_boost if should_apply_source_boost(normalized_method) else None,
            "where_source_type": args.where_source_type,
            "reranker_model_path": args.reranker_model_path if reranker_model is not None else None,
            "reranker_candidate_k": args.reranker_candidate_k if reranker_model is not None else None,
            "reranker_batch_size": args.reranker_batch_size if reranker_model is not None else None,
            "reranker_complex_question_types": sorted(COMPLEX_QUESTION_TYPES)
            if normalized_method in {"strategy_matrix", "strategy_matrix_decompose"}
            else None,
            "failures": len(failures),
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }
    )
    write_outputs(args.output_dir, args.method, summary, details, k_values)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
