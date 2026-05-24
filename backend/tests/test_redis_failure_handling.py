import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.anyio
async def test_rate_limit_dependency_returns_503_when_redis_unavailable(monkeypatch):
    from app.core import rate_limit as rate_limit_module

    async def failing_connect_redis():
        raise OSError("redis down")

    monkeypatch.setattr(rate_limit_module, "connect_redis", failing_connect_redis)
    dependency = rate_limit_module.rate_limit(limit=1, window=60)
    request = Request({"type": "http", "client": ("127.0.0.1", 1234), "headers": []})

    with pytest.raises(HTTPException) as exc_info:
        await dependency(request)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Redis unavailable"


@pytest.mark.anyio
async def test_rate_limit_middleware_returns_503_when_redis_unavailable(monkeypatch):
    from app.core import rate_limit as rate_limit_module

    async def failing_connect_redis():
        raise OSError("redis down")

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    events = []

    async def send(event):
        events.append(event)
    monkeypatch.setattr(rate_limit_module, "connect_redis", failing_connect_redis)
    middleware = rate_limit_module.RateLimitMiddleware(app, limit=1, window=60)

    await middleware(
        {"type": "http", "path": "/api/agent/router/query", "client": ("127.0.0.1", 1234), "headers": []},
        lambda: None,
        send,
    )

    assert events[0]["status"] == 503


@pytest.mark.anyio
async def test_rate_limit_middleware_skips_health_endpoint(monkeypatch):
    from app.core import rate_limit as rate_limit_module

    async def failing_connect_redis():
        raise AssertionError("health endpoint should not touch redis")

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    events = []

    async def send(event):
        events.append(event)
    monkeypatch.setattr(rate_limit_module, "connect_redis", failing_connect_redis)
    middleware = rate_limit_module.RateLimitMiddleware(app, limit=1, window=60)

    await middleware(
        {"type": "http", "path": "/health", "client": ("127.0.0.1", 1234), "headers": []},
        lambda: None,
        send,
    )

    assert events[0]["status"] == 200


@pytest.mark.anyio
async def test_get_current_user_id_returns_503_when_blacklist_redis_unavailable(monkeypatch):
    from app.utils import auth_utils

    async def failing_connect_redis():
        raise OSError("redis down")

    monkeypatch.setattr(auth_utils, "decode_django_jwt", lambda token: {"jti": "token-1", "user_id": "user-1"})
    monkeypatch.setattr(auth_utils, "connect_redis", failing_connect_redis)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

    with pytest.raises(HTTPException) as exc_info:
        await auth_utils.get_current_user_id(credentials)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Redis unavailable"
