import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEmbeddings:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


class FakeElasticsearchClient:
    def __init__(self):
        self.search_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        body = kwargs.get("body", {})
        if "knn" in body:
            return {
                "hits": {
                    "hits": [
                        {"_score": 0.9, "_source": {"parent_chunk_id": "p1", "parent_doc_id": "d1", "source_type": "policy", "title": "Policy", "section_heading": "A", "child_text": "child", "parent_text": "parent"}}
                    ]
                }
            }
        return {
            "hits": {
                "hits": [
                    {"_score": 3.0, "_source": {"parent_chunk_id": "p2", "parent_doc_id": "d2", "source_type": "faq", "title": "FAQ", "section_heading": "B", "child_text": "child 2", "parent_text": "parent 2"}}
                ]
            }
        }


def test_elasticsearch_backend_returns_standard_result_shape():
    from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend

    backend = ElasticsearchEnterpriseRetrievalBackend(
        client=FakeElasticsearchClient(),
        embeddings=FakeEmbeddings(),
        index_name="idx",
    )

    import asyncio

    result = asyncio.run(
        backend.retrieve_with_details(
            query="policy",
            final_top_k=2,
            dense_top_k=5,
            bm25_top_k=5,
            fusion_top_k=5,
            source_hints=["policy"],
            use_reranker=False,
        )
    )

    assert set(result) == {"dense_results", "bm25_results", "fused_results", "reranked_results", "selected_documents", "metrics"}
    assert result["dense_results"][0]["parent_chunk_id"] == "p1"
    assert result["dense_results"][0]["score"] == 0.9
    assert result["bm25_results"][0]["parent_chunk_id"] == "p2"
    assert result["bm25_results"][0]["score"] == 3.0
    assert result["selected_documents"]
    assert result["metrics"]["dense_ms"] >= 0.0


def test_elasticsearch_backend_builds_vector_and_bm25_queries():
    from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend

    client = FakeElasticsearchClient()
    backend = ElasticsearchEnterpriseRetrievalBackend(client=client, embeddings=FakeEmbeddings(), index_name="idx")

    import asyncio

    asyncio.run(backend.retrieve_with_details("policy", 2, 5, 5, 5, ["policy"], False))

    assert client.search_calls[0]["index"] == "idx"
    assert "knn" in client.search_calls[0]["body"]
    assert client.search_calls[1]["index"] == "idx"
    assert "query" in client.search_calls[1]["body"]


def test_elasticsearch_backend_skips_disabled_dense_search():
    from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend

    client = FakeElasticsearchClient()
    backend = ElasticsearchEnterpriseRetrievalBackend(client=client, embeddings=FakeEmbeddings(), index_name="idx")

    import asyncio

    result = asyncio.run(backend.retrieve_with_details("policy", 2, 0, 5, 5, ["policy"], False))

    assert result["dense_results"] == []
    assert len(client.search_calls) == 1
    assert "query" in client.search_calls[0]["body"]


def test_elasticsearch_backend_uses_configurable_rrf_parameters():
    from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend

    backend = ElasticsearchEnterpriseRetrievalBackend(
        client=FakeElasticsearchClient(),
        embeddings=FakeEmbeddings(),
        index_name="idx",
        rrf_k=10,
        source_hint_soft_boost=0.0,
    )

    import asyncio

    result = asyncio.run(backend.retrieve_with_details("policy", 2, 5, 5, 5, ["policy"], False))

    assert round(result["fused_results"][0]["score"], 6) == round(1 / 11, 6)
