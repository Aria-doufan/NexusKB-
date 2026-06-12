from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace
from time import perf_counter
from typing import Any

from langchain_ollama import OllamaEmbeddings

from app.rag.enterprise_rag_service import EnterpriseRetrievedDocument, RRF_K, SOURCE_HINT_SOFT_BOOST
from app.rag.reorder_service import reorder_service


class ElasticsearchEnterpriseRetrievalBackend:
    def __init__(self, client: Any, embeddings: Any, index_name: str):
        self.client = client
        self.embeddings = embeddings
        self.index_name = index_name

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        from elasticsearch import Elasticsearch

        es_config = config.get("elasticsearch", {}) or {}
        url = es_config.get("url") or "http://localhost:9200"
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
    ) -> dict[str, Any]:
        started = perf_counter()
        metrics: dict[str, float] = {}

        step = perf_counter()
        dense_documents = await self._dense_search(query, dense_top_k)
        metrics["dense_ms"] = (perf_counter() - step) * 1000

        step = perf_counter()
        bm25_documents = await self._bm25_search(query, bm25_top_k)
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
            "metrics": metrics,
        }

    async def _dense_search(self, query: str, top_k: int) -> list[EnterpriseRetrievedDocument]:
        if top_k <= 0:
            return []
        query_vector = await asyncio.to_thread(self.embeddings.embed_query, query)
        body = {"knn": {"field": "embedding", "query_vector": query_vector, "k": top_k, "num_candidates": max(top_k * 4, top_k)}}
        response = await asyncio.to_thread(self.client.search, index=self.index_name, body=body)
        return [self._document_from_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    async def _bm25_search(self, query: str, top_k: int) -> list[EnterpriseRetrievedDocument]:
        if top_k <= 0:
            return []
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["child_text^2", "parent_text", "title^2", "section_heading", "source_type"],
                }
            },
            "size": top_k,
        }
        response = await asyncio.to_thread(self.client.search, index=self.index_name, body=body)
        return [self._document_from_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

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
    ) -> dict[str, float]:
        scores: defaultdict[str, float] = defaultdict(float)
        for ranked_ids in ranked_lists:
            seen: set[str] = set()
            for rank, doc_id in enumerate(ranked_ids, start=1):
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                scores[doc_id] += 1.0 / (RRF_K + rank)
        source_hint_set = {source for source in (source_hints or []) if source}
        if source_hint_set:
            for doc_id, score in list(scores.items()):
                document = candidates.get(doc_id)
                if document and document.source_type in source_hint_set:
                    scores[doc_id] = score * (1.0 + SOURCE_HINT_SOFT_BOOST)
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
