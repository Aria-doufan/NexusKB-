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
                        {"_score": 0.9, "_source": {"parent_chunk_id": "p1", "parent_doc_id": "d1", "source_type": "policy", "doc_semantic_type": "policy_rule", "title": "Policy", "section_heading": "A", "child_text": "child", "parent_text": "parent"}}
                    ]
                }
            }
        return {
            "hits": {
                "hits": [
                    {"_score": 3.0, "_source": {"parent_chunk_id": "p2", "parent_doc_id": "d2", "source_type": "faq", "doc_semantic_type": "generic_doc", "title": "FAQ", "section_heading": "B", "child_text": "child 2", "parent_text": "parent 2"}}
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

    assert set(result) == {"dense_results", "bm25_results", "fused_results", "reranked_results", "selected_documents", "metadata_filter", "metrics"}
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


def test_elasticsearch_backend_applies_hard_metadata_filter_to_knn_and_bm25_queries():
    from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend
    from app.schemas.rag import MetadataFilterDecision

    client = FakeElasticsearchClient()
    backend = ElasticsearchEnterpriseRetrievalBackend(client=client, embeddings=FakeEmbeddings(), index_name="idx")

    import asyncio

    asyncio.run(
        backend.retrieve_with_details(
            query="Confluence PTO policy",
            final_top_k=2,
            dense_top_k=5,
            bm25_top_k=5,
            fusion_top_k=5,
            source_hints=[],
            use_reranker=False,
            metadata_filter=MetadataFilterDecision(
                mode="hard",
                source_types=["confluence"],
                doc_semantic_types=["policy_rule"],
                confidence=0.9,
            ),
        )
    )

    knn = client.search_calls[0]["body"]["knn"]
    bm25_query = client.search_calls[1]["body"]["query"]
    assert knn["filter"] == {
        "bool": {
            "filter": [
                {"terms": {"source_type": ["confluence"]}},
                {"terms": {"doc_semantic_type": ["policy_rule"]}},
            ]
        }
    }
    assert bm25_query["bool"]["filter"] == [
        {"terms": {"source_type": ["confluence"]}},
        {"terms": {"doc_semantic_type": ["policy_rule"]}},
    ]


def test_elasticsearch_backend_soft_filter_boosts_matching_metadata_without_query_filter():
    from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend
    from app.schemas.rag import MetadataFilterDecision

    client = FakeElasticsearchClient()
    backend = ElasticsearchEnterpriseRetrievalBackend(
        client=client,
        embeddings=FakeEmbeddings(),
        index_name="idx",
        source_hint_soft_boost=0.2,
    )

    import asyncio

    result = asyncio.run(
        backend.retrieve_with_details(
            query="PTO policy",
            final_top_k=2,
            dense_top_k=5,
            bm25_top_k=5,
            fusion_top_k=5,
            source_hints=[],
            use_reranker=False,
            metadata_filter=MetadataFilterDecision(
                mode="soft",
                doc_semantic_types=["policy_rule"],
                confidence=0.65,
            ),
        )
    )

    assert "filter" not in client.search_calls[0]["body"]["knn"]
    assert "bool" not in client.search_calls[1]["body"]["query"]
    assert result["fused_results"][0]["parent_chunk_id"] == "p1"
    assert result["metadata_filter"] == {
        "mode": "soft",
        "source_types": [],
        "doc_semantic_types": ["policy_rule"],
        "title_keywords": [],
        "section_keywords": [],
        "confidence": 0.65,
        "reason": "",
    }


def test_elasticsearch_backend_does_not_double_boost_source_hint_and_soft_filter_overlap():
    from app.rag.enterprise_rag_service import EnterpriseRetrievedDocument
    from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend
    from app.schemas.rag import MetadataFilterDecision

    document = EnterpriseRetrievedDocument(
        parent_doc_id="d1",
        parent_chunk_id="p1",
        source_type="policy",
        title="Policy",
        section_heading="Eligibility",
        score=0.0,
        child_text="child",
        parent_text="parent",
        metadata={"doc_semantic_type": "policy_rule"},
    )

    scores = ElasticsearchEnterpriseRetrievalBackend._rrf_fuse(
        ranked_lists=[["p1"]],
        source_hints=["policy"],
        candidates={"p1": document},
        rrf_k=60,
        source_hint_soft_boost=0.2,
        metadata_filter=MetadataFilterDecision(mode="soft", source_types=["policy"], confidence=0.7),
    )

    assert scores["p1"] == (1 / 61) * 1.2


def test_elasticsearch_backend_hard_filter_uses_phrase_matching_for_title_and_section_keywords():
    from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend
    from app.schemas.rag import MetadataFilterDecision

    client = FakeElasticsearchClient()
    backend = ElasticsearchEnterpriseRetrievalBackend(client=client, embeddings=FakeEmbeddings(), index_name="idx")

    import asyncio

    asyncio.run(
        backend.retrieve_with_details(
            query="PTO eligibility",
            final_top_k=2,
            dense_top_k=5,
            bm25_top_k=5,
            fusion_top_k=5,
            source_hints=[],
            use_reranker=False,
            metadata_filter=MetadataFilterDecision(
                mode="hard",
                title_keywords=["pto", "paid time off"],
                section_keywords=["eligibility"],
                confidence=0.9,
            ),
        )
    )

    expected_filter = {
        "bool": {
            "filter": [
                {
                    "bool": {
                        "should": [
                            {"match_phrase": {"title": "pto"}},
                            {"match_phrase": {"title": "paid time off"}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                {"match_phrase": {"section_heading": "eligibility"}},
            ]
        }
    }
    knn_filter = client.search_calls[0]["body"]["knn"]["filter"]
    bm25_filter = client.search_calls[1]["body"]["query"]["bool"]["filter"]

    assert knn_filter == expected_filter
    assert bm25_filter == expected_filter["bool"]["filter"]
    assert {"terms": {"title.keyword": ["pto", "paid time off"]}} not in bm25_filter
    assert {"terms": {"section_heading.keyword": ["eligibility"]}} not in bm25_filter


def test_elasticsearch_backend_soft_filter_boosts_title_and_section_keyword_matches():
    from app.rag.enterprise_rag_service import EnterpriseRetrievedDocument
    from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend
    from app.schemas.rag import MetadataFilterDecision

    title_document = EnterpriseRetrievedDocument(
        parent_doc_id="d1",
        parent_chunk_id="p1",
        source_type="faq",
        title="PTO Policy",
        section_heading="Overview",
        score=0.0,
        child_text="child",
        parent_text="parent",
        metadata={"doc_semantic_type": "generic_doc"},
    )
    section_document = EnterpriseRetrievedDocument(
        parent_doc_id="d2",
        parent_chunk_id="p2",
        source_type="faq",
        title="Benefits",
        section_heading="Eligibility",
        score=0.0,
        child_text="child",
        parent_text="parent",
        metadata={"doc_semantic_type": "generic_doc"},
    )

    scores = ElasticsearchEnterpriseRetrievalBackend._rrf_fuse(
        ranked_lists=[["p1", "p2"]],
        source_hints=[],
        candidates={"p1": title_document, "p2": section_document},
        rrf_k=60,
        source_hint_soft_boost=0.2,
        metadata_filter=MetadataFilterDecision(
            mode="soft",
            title_keywords=["pto"],
            section_keywords=["eligibility"],
            confidence=0.7,
        ),
    )

    assert scores["p1"] == (1 / 61) * 1.2
    assert scores["p2"] == (1 / 62) * 1.2
