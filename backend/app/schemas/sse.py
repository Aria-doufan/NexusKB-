import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class SseEvent(BaseModel):
    event: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    debug_id: str | None = None
    session_id: str | None = None
    stage: str = Field(min_length=1)
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


def format_sse_event(event: SseEvent | dict[str, Any]) -> str:
    if isinstance(event, SseEvent):
        payload = event.model_dump(mode="json")
    else:
        payload = event
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
