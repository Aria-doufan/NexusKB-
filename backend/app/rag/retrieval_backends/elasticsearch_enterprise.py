from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace
from time import perf_counter
from typing import Any

from langchain_ollama import OllamaEmbeddings

from app.rag.enterprise_rag_service import EnterpriseRetrievedDocument, RRF_K, SOURCE_HINT_SOFT_BOOST
from app.rag.reorder_service import reorder_service
from app.schemas.rag import MetadataFilterDecision


class ElasticsearchEnterpriseRetrievalBackend:
    def __init__(
        self,
        client: Any,
        embeddings: Any,
        index_name: str,
        rrf_k: int = RRF_K,
        source_hint_soft_boost: float = SOURCE_HINT_SOFT_BOOST,
    ):
        self.client = client
        self.embeddings = embeddings
        self.index_name = index_name
        self.rrf_k = rrf_k
        self.source_hint_soft_boost = source_hint_soft_boost

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        from elasticsearch import Elasticsearch

        es_config = config.get("elasticsearch", {}) or {}
        url = es_config.get("url") or "http://localhost:9298"
        index_name = es_config.get("index_name") or "nexuskb_enterprise_chunks"
        username = es_config.get("username")
        password = es_config.get("password")
        client_kwargs: dict[str, Any] = {"hosts": [url]}
        if username and password:
            client_kwargs["basic_auth"] = (username, password)
        client = Elasticsearch(**client_kwargs)
        embeddings = OllamaEmbeddings(
            model=config.get("text_embedding_model_name", "qwen3-embedding:latest"),
            base_url=es_config.get("ollama_base_url", "http://localhost:11434"),
        )
        return cls(client=client, embeddings=embeddings, index_name=index_name)

    async def retrieve_with_details(
        self,
        query: str,
        final_top_k: int,
        dense_top_k: int,
        bm25_top_k: int,
        fusion_top_k: int,
        source_hints: list[str] | None = None,
        use_reranker: bool = False,
        metadata_filter: MetadataFilterDecision | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        metrics: dict[str, float] = {}

        step = perf_counter()
        dense_documents = await self._dense_search(query, dense_top_k, metadata_filter)
        metrics["dense_ms"] = (perf_counter() - step) * 1000

        step = perf_counter()
        bm25_documents = await self._bm25_search(query, bm25_top_k, metadata_filter)
        metrics["bm25_ms"] = (perf_counter() - step) * 1000

        step = perf_counter()
        candidates: dict[str, EnterpriseRetrievedDocument] = {}
        for document in [*dense_documents, *bm25_documents]:
            if document.parent_chunk_id and document.parent_chunk_id not in candidates:
                candidates[document.parent_chunk_id] = document
        fused_scores = self._rrf_fuse(
            ranked_lists=[
                [document.parent_chunk_id for document in dense_documents],
                [document.parent_chunk_id for document in bm25_documents],
            ],
            source_hints=source_hints,
            candidates=candidates,
            rrf_k=self.rrf_k,
            source_hint_soft_boost=self.source_hint_soft_boost,
            metadata_filter=metadata_filter,
        )
        fused_documents = [
            replace(candidates[parent_chunk_id], score=score)
            for parent_chunk_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
            if parent_chunk_id in candidates
        ][:fusion_top_k]
        metrics["rrf_ms"] = (perf_counter() - step) * 1000

        reranked_documents = fused_documents
        if use_reranker:
            step = perf_counter()
            reranked_documents = await self._rerank_documents(query, fused_documents)
            metrics["rerank_ms"] = (perf_counter() - step) * 1000
        else:
            metrics["rerank_ms"] = 0.0

        selected_documents = reranked_documents[:final_top_k]
        metrics["total_ms"] = (perf_counter() - started) * 1000
        return {
            "dense_results": [document.to_dict() for document in dense_documents],
            "bm25_results": [document.to_dict() for document in bm25_documents],
            "fused_results": [document.to_dict() for document in fused_documents],
            "reranked_results": [document.to_dict() for document in reranked_documents] if use_reranker else [],
            "selected_documents": [document.to_dict() for document in selected_documents],
            "metadata_filter": (metadata_filter or MetadataFilterDecision()).model_dump(),
            "metrics": metrics,
        }

    async def _dense_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: MetadataFilterDecision | None = None,
    ) -> list[EnterpriseRetrievedDocument]:
        if top_k <= 0:
            return []
        query_vector = await asyncio.to_thread(self.embeddings.embed_query, query)
        knn = {"field": "embedding", "query_vector": query_vector, "k": top_k, "num_candidates": max(top_k * 4, top_k)}
        es_filter = self._metadata_filter_to_es_filter(metadata_filter)
        if es_filter:
            knn["filter"] = es_filter
        body = {"knn": knn}
        response = await asyncio.to_thread(self.client.search, index=self.index_name, body=body)
        return [self._document_from_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: MetadataFilterDecision | None = None,
    ) -> list[EnterpriseRetrievedDocument]:
        if top_k <= 0:
            return []
        multi_match = {
            "multi_match": {
                "query": query,
                "fields": ["child_text^2", "parent_text", "title^2", "section_heading", "source_type", "doc_semantic_type"],
            }
        }
        es_filter = self._metadata_filter_to_es_filter(metadata_filter)
        query_body = multi_match
        if es_filter:
            query_body = {"bool": {"must": [multi_match], "filter": es_filter["bool"]["filter"]}}
        body = {"query": query_body, "size": top_k}
        response = await asyncio.to_thread(self.client.search, index=self.index_name, body=body)
        return [self._document_from_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    @staticmethod
    def _metadata_filter_to_es_filter(metadata_filter: MetadataFilterDecision | None) -> dict[str, Any] | None:
        if not metadata_filter or metadata_filter.mode != "hard":
            return None
        clauses: list[dict[str, Any]] = []
        field_filters = [
            ("source_type", metadata_filter.source_types),
            ("doc_semantic_type", metadata_filter.doc_semantic_types),
        ]
        for field, values in field_filters:
            if values:
                clauses.append({"terms": {field: values}})
        title_clause = ElasticsearchEnterpriseRetrievalBackend._phrase_filter_clause("title", metadata_filter.title_keywords)
        if title_clause:
            clauses.append(title_clause)
        section_clause = ElasticsearchEnterpriseRetrievalBackend._phrase_filter_clause("section_heading", metadata_filter.section_keywords)
        if section_clause:
            clauses.append(section_clause)
        if not clauses:
            return None
        return {"bool": {"filter": clauses}}

    @staticmethod
    def _phrase_filter_clause(field: str, phrases: list[str]) -> dict[str, Any] | None:
        match_phrases = [{"match_phrase": {field: phrase}} for phrase in phrases if phrase]
        if not match_phrases:
            return None
        if len(match_phrases) == 1:
            return match_phrases[0]
        return {"bool": {"should": match_phrases, "minimum_should_match": 1}}

    @staticmethod
    def _document_from_hit(hit: dict[str, Any]) -> EnterpriseRetrievedDocument:
        source = hit.get("_source", {}) or {}
        return EnterpriseRetrievedDocument(
            parent_doc_id=source.get("parent_doc_id", ""),
            parent_chunk_id=source.get("parent_chunk_id", ""),
            source_type=source.get("source_type", ""),
            title=source.get("title", ""),
            section_heading=source.get("section_heading", ""),
            score=float(hit.get("_score", 0.0) or 0.0),
            child_text=source.get("child_text", ""),
            parent_text=source.get("parent_text", source.get("child_text", "")),
            metadata={key: value for key, value in source.items() if key != "embedding"},
        )

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[str]],
        source_hints: list[str] | None,
        candidates: dict[str, EnterpriseRetrievedDocument],
        rrf_k: int,
        source_hint_soft_boost: float,
        metadata_filter: MetadataFilterDecision | None = None,
    ) -> dict[str, float]:
        scores: defaultdict[str, float] = defaultdict(float)
        for ranked_ids in ranked_lists:
            seen: set[str] = set()
            for rank, doc_id in enumerate(ranked_ids, start=1):
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                scores[doc_id] += 1.0 / (rrf_k + rank)
        source_hint_set = {source for source in (source_hints or []) if source}
        soft_source_type_set: set[str] = set()
        soft_doc_semantic_type_set: set[str] = set()
        soft_title_keywords: list[str] = []
        soft_section_keywords: list[str] = []
        if metadata_filter and metadata_filter.mode == "soft":
            soft_source_type_set = {source_type for source_type in metadata_filter.source_types if source_type}
            soft_doc_semantic_type_set = {doc_type for doc_type in metadata_filter.doc_semantic_types if doc_type}
            soft_title_keywords = [keyword.lower() for keyword in metadata_filter.title_keywords if keyword]
            soft_section_keywords = [keyword.lower() for keyword in metadata_filter.section_keywords if keyword]
        if source_hint_set or soft_source_type_set or soft_doc_semantic_type_set or soft_title_keywords or soft_section_keywords:
            for doc_id, score in list(scores.items()):
                document = candidates.get(doc_id)
                if not document:
                    continue
                title = document.title.lower()
                section_heading = document.section_heading.lower()
                matches_soft_filter = (
                    document.source_type in source_hint_set
                    or document.source_type in soft_source_type_set
                    or document.metadata.get("doc_semantic_type") in soft_doc_semantic_type_set
                    or any(keyword in title for keyword in soft_title_keywords)
                    or any(keyword in section_heading for keyword in soft_section_keywords)
                )
                if matches_soft_filter:
                    scores[doc_id] = score * (1.0 + source_hint_soft_boost)
        return dict(scores)

    async def _rerank_documents(self, query: str, documents: list[EnterpriseRetrievedDocument]) -> list[EnterpriseRetrievedDocument]:
        if not documents:
            return []
        document_texts = [document.parent_text for document in documents]
        reranked = await asyncio.to_thread(lambda: asyncio.run(reorder_service.reorder_documents(query, document_texts)))
        if not reranked.get("success"):
            return documents
        remaining = list(documents)
        ordered: list[EnterpriseRetrievedDocument] = []
        for item in reranked.get("documents", []):
            text = item.get("document", "")
            for index, document in enumerate(remaining):
                if document.parent_text == text:
                    ordered.append(document)
                    remaining.pop(index)
                    break
        return ordered or documents
