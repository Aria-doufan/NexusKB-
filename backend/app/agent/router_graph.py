"""Compatibility adapter around AgenticRagGraph for existing API response fields and SSE event shape.

RouterGraph preserves the public API contract while AgenticRagGraph owns the orchestration and business logic.
"""

import json
from typing import Any, AsyncGenerator

from app.rag.agentic_rag_graph import AgenticRagGraph


class RouterGraph:
    """Compatibility wrapper around the single AgenticRagGraph."""

    def __init__(self):
        self.agentic_rag_graph = AgenticRagGraph()

    async def invoke(self, query: str, user_id: str, session_id: str | None = None) -> dict[str, Any]:
        result = await self.agentic_rag_graph.invoke(query=query, user_id=user_id, session_id=session_id)
        return {
            "session_id": result.get("session_id"),
            "route": "agentic_rag",
            "request_id": result.get("request_id"),
            "debug_id": result.get("debug_id"),
            "rag_intent": result.get("rag_intent", "unknown"),
            "source_hints": result.get("source_hints", []),
            "confidence": result.get("confidence", 0.0),
            "reason": result.get("reason", ""),
            "response": result.get("response", ""),
            "steps": result.get("steps", []),
            "error": result.get("error"),
        }

    async def stream(self, query: str, user_id: str, session_id: str | None = None) -> AsyncGenerator[str, None]:
        result = await self.agentic_rag_graph.invoke(query=query, user_id=user_id, session_id=session_id)
        yield self._sse_event(
            {
                "type": "route",
                "session_id": result.get("session_id"),
                "request_id": result.get("request_id"),
                "debug_id": result.get("debug_id"),
                "route": "agentic_rag",
                "rag_intent": result.get("rag_intent", "unknown"),
                "source_hints": result.get("source_hints", []),
                "confidence": result.get("confidence", 0.0),
                "reason": result.get("reason", ""),
            }
        )
        for event in result.get("sse_events", []):
            yield self._sse_event(event)
        yield self._sse_event(
            {
                "type": "response",
                "content": result.get("response", ""),
                "session_id": result.get("session_id"),
                "request_id": result.get("request_id"),
                "debug_id": result.get("debug_id"),
            }
        )
        yield self._sse_event(
            {
                "type": "done",
                "session_id": result.get("session_id"),
                "request_id": result.get("request_id"),
                "debug_id": result.get("debug_id"),
            }
        )

    @staticmethod
    def _sse_event(data: dict[str, Any]) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


router_graph = RouterGraph()
