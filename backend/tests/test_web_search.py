import sys
from pathlib import Path

import pytest
import requests


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.anyio
async def test_disabled_web_search_returns_no_results(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)

    from app.rag.web_search import WebSearchService

    service = WebSearchService()
    results = await service.search("通用报销流程", max_results=3)

    assert results == []


@pytest.mark.anyio
async def test_http_web_search_client_posts_through_async_thread(monkeypatch):
    from app.rag import web_search

    calls = []

    class FakeResponse:
        def raise_for_status(self):
            calls.append(("raise_for_status",))

        def json(self):
            return {
                "results": [
                    {
                        "title": "企业报销流程参考",
                        "url": "https://example.test/expense",
                        "snippet": "提交申请、主管审批、财务复核。",
                    }
                ]
            }

    def fake_post(endpoint, *, json, headers, timeout):
        calls.append(("post", endpoint, json, headers, timeout))
        return FakeResponse()

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(("to_thread", func))
        return func(*args, **kwargs)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(web_search.asyncio, "to_thread", fake_to_thread)

    client = web_search.HttpWebSearchClient(endpoint="https://search.example.test", api_key="test-key")
    results = await client.search("通用报销流程", max_results=1)

    assert results == [
        {
            "title": "企业报销流程参考",
            "url": "https://example.test/expense",
            "snippet": "提交申请、主管审批、财务复核。",
        }
    ]
    assert calls == [
        ("to_thread", fake_post),
        (
            "post",
            "https://search.example.test",
            {"query": "通用报销流程", "max_results": 1},
            {"Authorization": "Bearer test-key", "Content-Type": "application/json"},
            8,
        ),
        ("raise_for_status",),
    ]


@pytest.mark.anyio
async def test_fake_web_search_client_normalizes_results():
    from app.rag.web_search import WebSearchService

    class FakeClient:
        async def search(self, query, max_results):
            assert query == "通用报销流程"
            assert max_results == 2
            return [
                {
                    "title": "企业报销流程参考",
                    "url": "https://example.test/expense",
                    "snippet": "提交申请、主管审批、财务复核。",
                    "score": 0.8,
                }
            ]

    service = WebSearchService(client=FakeClient(), enabled=True)
    results = await service.search("通用报销流程", max_results=2)

    assert len(results) == 1
    assert results[0].title == "企业报销流程参考"
    assert results[0].url == "https://example.test/expense"
    assert results[0].snippet == "提交申请、主管审批、财务复核。"
    assert results[0].score == 0.8
    assert results[0].provider == "web"


@pytest.mark.anyio
async def test_web_search_returns_empty_results_for_malformed_provider_payload():
    from app.rag.web_search import WebSearchService

    class FakeClient:
        async def search(self, query, max_results):
            return [{"score": "not-a-number"}]

    service = WebSearchService(client=FakeClient(), enabled=True)
    results = await service.search("通用报销流程", max_results=1)

    assert results == []


@pytest.mark.anyio
async def test_web_search_skips_malformed_items_but_keeps_valid_results():
    from app.rag.web_search import WebSearchService

    class FakeClient:
        async def search(self, query, max_results):
            return [
                {
                    "title": "企业报销流程参考",
                    "url": "https://example.test/expense-1",
                    "snippet": "提交申请、主管审批、财务复核。",
                    "score": 0.8,
                },
                {"title": "Bad score", "url": "https://example.test/bad", "score": "not-a-number"},
                {
                    "title": "备用报销参考",
                    "url": "https://example.test/expense-2",
                    "snippet": "保留票据并提交审批。",
                    "score": 0.7,
                },
            ]

    service = WebSearchService(client=FakeClient(), enabled=True)
    results = await service.search("通用报销流程", max_results=3)

    assert [result.title for result in results] == ["企业报销流程参考", "备用报销参考"]
    assert [result.url for result in results] == [
        "https://example.test/expense-1",
        "https://example.test/expense-2",
    ]


@pytest.mark.anyio
async def test_web_search_skips_contentless_items_when_capping():
    from app.rag.web_search import WebSearchService

    class FakeClient:
        async def search(self, query, max_results):
            assert max_results == 1
            return [
                {"score": 0.8},
                {
                    "title": "Valid reference",
                    "url": "https://example.test/valid",
                    "snippet": "Useful",
                    "score": 0.7,
                },
            ]

    service = WebSearchService(client=FakeClient(), enabled=True)
    results = await service.search("通用报销流程", max_results=1)

    assert len(results) == 1
    assert results[0].url == "https://example.test/valid"


@pytest.mark.anyio
async def test_web_search_considers_valid_results_after_malformed_items_when_capping():
    from app.rag.web_search import WebSearchService

    class FakeClient:
        async def search(self, query, max_results):
            assert max_results == 2
            return [
                {"title": "Bad score", "url": "https://example.test/bad", "score": "not-a-number"},
                {
                    "title": "企业报销流程参考",
                    "url": "https://example.test/expense-1",
                    "snippet": "提交申请、主管审批、财务复核。",
                    "score": 0.8,
                },
                {
                    "title": "备用报销参考",
                    "url": "https://example.test/expense-2",
                    "snippet": "保留票据并提交审批。",
                    "score": 0.7,
                },
            ]

    service = WebSearchService(client=FakeClient(), enabled=True)
    results = await service.search("通用报销流程", max_results=2)

    assert [result.title for result in results] == ["企业报销流程参考", "备用报销参考"]
    assert [result.url for result in results] == [
        "https://example.test/expense-1",
        "https://example.test/expense-2",
    ]


def test_normalize_web_search_results_accepts_existing_models():
    from app.rag.web_search import normalize_web_search_results
    from app.schemas.rag import WebSearchResult

    existing = WebSearchResult(
        title="通用参考",
        url="https://example.test/reference",
        snippet="公开资料。",
        score=0.6,
    )

    results = normalize_web_search_results([existing], max_results=3)

    assert results == [existing]
