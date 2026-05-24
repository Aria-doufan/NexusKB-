from fastapi import Request, HTTPException
from redis.exceptions import RedisError
from starlette.responses import JSONResponse

from app.db.redis_config import connect_redis


def rate_limit(limit: int = 1, window: int = 60):
    """
    限流依赖函数
    :param limit: 时间窗口内的最大请求数
    :param window: 时间窗口大小（秒）
    :return: 依赖函数
    """
    async def dependency(request: Request):
        # 获取客户端IP
        client_ip = request.client.host
        if not client_ip:
            client_ip = request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or 'unknown'

        # 生成限流键
        key = f"rate_limit:aichat:{client_ip}"

        try:
            redis = await connect_redis()
            current = await redis.get(key)
            current = int(current) if current else 0

            if current >= limit:
                raise HTTPException(
                    status_code=429,
                    detail="请求过于频繁，请稍后再试"
                )

            if current == 0:
                await redis.setex(key, window, 1)
            else:
                await redis.incr(key)
        except HTTPException:
            raise
        except (RedisError, OSError) as exc:
            raise HTTPException(status_code=503, detail="Redis unavailable") from exc

    return dependency

class RateLimitMiddleware:
    """
    全局限流中间件
    """
    def __init__(self, app, limit: int = 100, window: int = 60):
        self.app = app
        self.limit = limit
        self.window = window

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        if scope.get('path', '').startswith('/health'):
            await self.app(scope, receive, send)
            return

        # 构建请求对象
        from fastapi import Request
        request = Request(scope, receive)
        
        # 获取客户端IP
        client_ip = request.client.host
        if not client_ip:
            client_ip = request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or 'unknown'

        # 生成限流键
        key = f"rate_limit:global:{client_ip}"

        try:
            redis = await connect_redis()
            current = await redis.get(key)
            current = int(current) if current else 0

            if current >= self.limit:
                response = JSONResponse(
                    {"detail": "请求过于频繁，请稍后再试"},
                    status_code=429
                )
                await response(scope, receive, send)
                return

            if current == 0:
                await redis.setex(key, self.window, 1)
            else:
                await redis.incr(key)
        except (RedisError, OSError):
            response = JSONResponse({"detail": "Redis unavailable"}, status_code=503)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)