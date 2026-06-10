import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_agentic_rag_state_defaults_support_single_graph_actions():
    from app.schemas.rag import AgenticActionDecision, RagState, ToolExecutionResult

    state = RagState(
        request_id="req-1",
        debug_id="dbg-1",
        user_id="user-1",
        original_query="What is LangGraph?",
        current_query="What is LangGraph?",
    )

    assert state.action == "retrieve"
    assert state.response_type == "answer"
    assert state.history == []
    assert state.memory_summary == ""
    assert state.long_term_memories == []
    assert state.required_tools == []
    assert state.tool_results == []

    decision = AgenticActionDecision(
        intent="general_chat",
        action="direct_answer",
        needs_retrieval=False,
        needs_tool=False,
        needs_clarification=False,
        safety_risk=False,
        confidence=0.91,
        reason="General explanation does not need enterprise evidence.",
    )
    assert decision.action == "direct_answer"

    tool_result = ToolExecutionResult(
        tool_name="what_time_is_now",
        tool_input={},
        output="当前时间是：2026-06-10 10:30",
        success=True,
    )
    assert tool_result.success is True
