import os
import re
import sys
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


NEW_AGENTIC_RAG_INTENTS = {
    "fact_lookup",
    "semantic_query",
    "multi_hop",
    "comparison",
    "procedure",
    "constrained",
    "follow_up",
    "not_enough_info",
    "unknown",
}


def test_allowed_rag_intents_use_agentic_taxonomy_only():
    from app.agent.router_graph import ALLOWED_RAG_INTENTS

    assert ALLOWED_RAG_INTENTS == NEW_AGENTIC_RAG_INTENTS


def test_normalize_rag_intent_maps_legacy_labels_to_agentic_taxonomy():
    from app.agent.router_graph import RouterGraph

    assert RouterGraph._normalize_rag_intent("basic") == "fact_lookup"
    assert RouterGraph._normalize_rag_intent("semantic") == "semantic_query"
    assert RouterGraph._normalize_rag_intent("intra_document_reasoning") == "multi_hop"
    assert RouterGraph._normalize_rag_intent("project_related") == "multi_hop"
    assert RouterGraph._normalize_rag_intent("conflicting_info") == "comparison"
    assert RouterGraph._normalize_rag_intent("completeness") == "multi_hop"
    assert RouterGraph._normalize_rag_intent("high_level") == "semantic_query"
    assert RouterGraph._normalize_rag_intent("info_not_found") == "not_enough_info"


def test_normalize_rag_intent_keeps_new_labels_and_rejects_invalid_values():
    from app.agent.router_graph import RouterGraph

    for intent in NEW_AGENTIC_RAG_INTENTS:
        assert RouterGraph._normalize_rag_intent(intent) == intent

    assert RouterGraph._normalize_rag_intent("") == "unknown"
    assert RouterGraph._normalize_rag_intent(None) == "unknown"
    assert RouterGraph._normalize_rag_intent("nonsense") == "unknown"


def test_router_prompts_request_only_agentic_taxonomy():
    from app.agent.router_graph import ROUTER_HUMAN_PROMPT, ROUTER_SYSTEM_PROMPT

    combined_prompt = f"{ROUTER_SYSTEM_PROMPT}\n{ROUTER_HUMAN_PROMPT}"

    for intent in NEW_AGENTIC_RAG_INTENTS:
        assert intent in combined_prompt

    prompt_tokens = set(re.findall(r"[A-Za-z_]+", combined_prompt))
    for legacy_intent in {
        "basic",
        "semantic",
        "intra_document_reasoning",
        "project_related",
        "conflicting_info",
        "completeness",
        "high_level",
        "info_not_found",
    }:
        assert legacy_intent not in prompt_tokens
