import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEnterpriseService:
    def __init__(self):
        self.calls = []

    async def retrieve_with_details(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "dense_results": [],
            "bm25_results": [],
            "fused_results": [],
            "reranked_results": [],
            "selected_documents": [],
            "metrics": {"dense_ms": 1.0, "bm25_ms": 2.0, "rrf_ms": 3.0, "rerank_ms": 0.0},
        }


def test_chroma_backend_delegates_to_enterprise_service():
    from app.rag.retrieval_backends.chroma_enterprise import ChromaEnterpriseRetrievalBackend

    service = FakeEnterpriseService()
    backend = ChromaEnterpriseRetrievalBackend(service=service)

    import asyncio

    result = asyncio.run(
        backend.retrieve_with_details(
            query="policy",
            final_top_k=5,
            dense_top_k=40,
            bm25_top_k=40,
            fusion_top_k=40,
            source_hints=["policy"],
            use_reranker=False,
        )
    )

    assert result["metrics"]["dense_ms"] == 1.0
    assert service.calls == [
        {
            "query": "policy",
            "final_top_k": 5,
            "dense_top_k": 40,
            "bm25_top_k": 40,
            "fusion_top_k": 40,
            "source_hints": ["policy"],
            "use_reranker": False,
        }
    ]
