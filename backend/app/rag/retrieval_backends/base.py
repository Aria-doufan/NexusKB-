from __future__ import annotations

from typing import Any, Protocol


class EnterpriseRetrievalBackend(Protocol):
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
        ...
