import asyncio
import os
from typing import Any, Protocol

import requests
from pydantic import ValidationError

from app.core.logger_handler import logger
from app.schemas.rag import WebSearchResult


class WebSearchClient(Protocol):
    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        ...


class HttpWebSearchClient:
    def __init__(self, endpoint: str | None = None, api_key: str | None = None):
        self.endpoint = endpoint or os.getenv("WEB_SEARCH_ENDPOINT", "")
        self.api_key = api_key or os.getenv("WEB_SEARCH_API_KEY", "")

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if not self.endpoint or not self.api_key:
            return []

        payload = {"query": query, "max_results": max_results}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = await asyncio.to_thread(requests.post, self.endpoint, json=payload, headers=headers, timeout=8)
        response.raise_for_status()
        data = response.json()
        raw_results = data.get("results", []) if isinstance(data, dict) else []
        return raw_results if isinstance(raw_results, list) else []


def _normalize_web_search_result(item: dict[str, Any]) -> WebSearchResult:
    title = str(item.get("title") or item.get("name") or "").strip()
    url = str(item.get("url") or item.get("link") or "").strip()
    snippet = str(item.get("snippet") or item.get("content") or item.get("description") or "").strip()
    if not (title or url or snippet):
        raise ValueError("search result has no usable title, url, or snippet")
    if not (url or snippet):
        raise ValueError("search result must include a usable url or snippet")
    return WebSearchResult(
        title=title,
        url=url,
        snippet=snippet,
        score=float(item.get("score", 0.0) or 0.0),
        provider=str(item.get("provider") or "web"),
        metadata={key: value for key, value in item.items() if key not in {"title", "name", "url", "link", "snippet", "content", "description", "score", "provider"}},
    )


def normalize_web_search_results(raw_results: list[Any], max_results: int) -> list[WebSearchResult]:
    results: list[WebSearchResult] = []
    for item in raw_results:
        if len(results) >= max_results:
            break
        if isinstance(item, WebSearchResult):
            results.append(item)
            continue
        if not isinstance(item, dict):
            logger.warning(f"【WebSearch】跳过非字典搜索结果: {type(item).__name__}")
            continue
        try:
            results.append(_normalize_web_search_result(item))
        except (TypeError, ValueError, ValidationError) as exc:
            logger.warning(f"【WebSearch】跳过格式异常的搜索结果: {exc}")
    return results


class WebSearchService:
    def __init__(self, client: WebSearchClient | None = None, enabled: bool | None = None):
        self.client = client or HttpWebSearchClient()
        if enabled is None:
            enabled = os.getenv("WEB_SEARCH_ENABLED", "false").lower() == "true"
        self.enabled = enabled

    async def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        if not self.enabled:
            return []
        try:
            raw_results = await self.client.search(query, max_results)
        except Exception as exc:
            logger.warning(f"【WebSearch】搜索失败，跳过公网兜底: {exc}")
            return []

        return normalize_web_search_results(raw_results, max_results)


web_search_service = WebSearchService()
