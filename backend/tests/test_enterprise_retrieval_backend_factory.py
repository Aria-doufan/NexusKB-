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


def test_chroma_backend_accepts_metadata_filter_for_pipeline_compatibility():
    from app.rag.retrieval_backends.chroma_enterprise import ChromaEnterpriseRetrievalBackend
    from app.schemas.rag import MetadataFilterDecision

    class FakeService:
        async def retrieve_with_details(self, **kwargs):
            return {
                "dense_results": [],
                "bm25_results": [],
                "fused_results": [],
                "reranked_results": [],
                "selected_documents": [],
                "metrics": {"dense_ms": 0.0, "bm25_ms": 0.0, "rrf_ms": 0.0, "rerank_ms": 0.0},
            }

    import asyncio

    backend = ChromaEnterpriseRetrievalBackend(FakeService())
    result = asyncio.run(
        backend.retrieve_with_details(
            query="Find Confluence policy",
            final_top_k=5,
            dense_top_k=40,
            bm25_top_k=40,
            fusion_top_k=40,
            source_hints=[],
            use_reranker=False,
            metadata_filter=MetadataFilterDecision(mode="hard", source_types=["confluence"]),
        )
    )

    assert result["selected_documents"] == []


def test_factory_defaults_to_chroma_backend(monkeypatch):
    monkeypatch.delenv("NEXUSKB_RETRIEVAL_BACKEND", raising=False)

    from app.rag.retrieval_backends.chroma_enterprise import ChromaEnterpriseRetrievalBackend
    from app.rag.retrieval_backends.factory import build_enterprise_retrieval_backend

    backend = build_enterprise_retrieval_backend(service=FakeEnterpriseService())

    assert isinstance(backend, ChromaEnterpriseRetrievalBackend)


def test_factory_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("NEXUSKB_RETRIEVAL_BACKEND", "unknown")

    from app.rag.retrieval_backends.factory import build_enterprise_retrieval_backend

    import pytest

    with pytest.raises(ValueError, match="Unsupported enterprise retrieval backend"):
        build_enterprise_retrieval_backend(service=FakeEnterpriseService())


def test_rag_evidence_workflow_default_pipeline_uses_backend_factory(monkeypatch):
    from app.rag.rag_evidence_workflow import RagEvidenceWorkflow
    from app.rag.retrieval_pipeline import RetrievalPipeline

    service = FakeEnterpriseService()
    workflow = RagEvidenceWorkflow(service=service)

    assert isinstance(workflow.retrieval_pipeline, RetrievalPipeline)
    assert hasattr(workflow.retrieval_pipeline.service, "retrieve_with_details")
