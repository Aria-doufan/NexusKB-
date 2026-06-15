import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RecordingRetrievalService:
    def __init__(self):
        self.calls = []

    async def retrieve_with_details(self, **kwargs):
        self.calls.append(kwargs)
        metadata_filter = kwargs.get("metadata_filter")
        return {
            "dense_results": [],
            "bm25_results": [],
            "fused_results": [],
            "reranked_results": [],
            "selected_documents": [
                {
                    "parent_chunk_id": "p1",
                    "parent_doc_id": "d1",
                    "source_type": "confluence",
                    "title": "PTO Policy",
                    "section_heading": "Eligibility",
                    "score": 1.0,
                    "parent_text": "PTO policy text",
                    "child_text": "PTO child text",
                    "metadata": {"doc_semantic_type": "policy_rule"},
                }
            ],
            "metrics": {"dense_ms": 1.0, "bm25_ms": 2.0, "rrf_ms": 3.0, "rerank_ms": 0.0},
            "metadata_filter": metadata_filter.model_dump() if metadata_filter else None,
        }


@pytest.mark.anyio
async def test_retrieval_pipeline_passes_metadata_filter_and_records_it():
    from app.rag.retrieval_pipeline import RetrievalPipeline
    from app.schemas.rag import MetadataFilterDecision, RagStrategyConfig

    service = RecordingRetrievalService()
    pipeline = RetrievalPipeline(service)
    decision = MetadataFilterDecision(mode="hard", source_types=["confluence"], doc_semantic_types=["policy_rule"], confidence=0.9)

    result = await pipeline.run(
        query="Find Confluence PTO policy",
        strategy=RagStrategyConfig(final_top_k=1),
        source_hints=[],
        rag_intent="constrained",
        router_confidence=0.9,
        attempt_id=1,
        metadata_filter=decision,
        reason="Initial hard metadata filter retrieval.",
    )

    assert service.calls[0]["metadata_filter"] == decision
    assert result.attempt.metadata_filter.mode == "hard"
    assert result.attempt.metadata_filter.source_types == ["confluence"]
    assert result.selected_documents[0].metadata["doc_semantic_type"] == "policy_rule"
