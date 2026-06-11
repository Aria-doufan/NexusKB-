import os
import sys
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_router_graph_exposes_only_agentic_rag_wrapper_api():
    from app.agent import router_graph
    from app.agent.router_graph import RouterGraph

    assert hasattr(RouterGraph, "invoke")
    assert hasattr(RouterGraph, "stream")
    assert hasattr(RouterGraph, "_sse_event")
    for legacy_name in {
        "ALLOWED_RAG_INTENTS",
        "ROUTER_SYSTEM_PROMPT",
        "ROUTER_HUMAN_PROMPT",
        "RouteDecision",
        "GraphState",
    }:
        assert not hasattr(router_graph, legacy_name)


def test_router_graph_no_longer_normalizes_legacy_router_intents():
    from app.agent.router_graph import RouterGraph

    assert not hasattr(RouterGraph, "_normalize_rag_intent")
    assert not hasattr(RouterGraph, "_normalize_route")
