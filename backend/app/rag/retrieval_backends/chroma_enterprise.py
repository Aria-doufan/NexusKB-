from __future__ import annotations

from typing import Any

from app.schemas.rag import MetadataFilterDecision


class ChromaEnterpriseRetrievalBackend:
    def __init__(self, service=None):
        if service is None:
            from app.rag.enterprise_rag_service import enterprise_rag_service

            service = enterprise_rag_service
        self.service = service

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
        result = await self.service.retrieve_with_details(
            query=query,
            final_top_k=final_top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            fusion_top_k=fusion_top_k,
            source_hints=source_hints,
            use_reranker=use_reranker,
        )
        if "metadata_filter" not in result:
            result["metadata_filter"] = (metadata_filter or MetadataFilterDecision()).model_dump()
        return result
