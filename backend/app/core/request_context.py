from contextvars import ContextVar


request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
debug_id_ctx: ContextVar[str | None] = ContextVar("debug_id", default=None)


def set_request_context(request_id: str | None, debug_id: str | None):
    request_token = request_id_ctx.set(request_id)
    debug_token = debug_id_ctx.set(debug_id)
    return request_token, debug_token


def reset_request_context(tokens) -> None:
    request_token, debug_token = tokens
    request_id_ctx.reset(request_token)
    debug_id_ctx.reset(debug_token)
