import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_format_context_includes_doc_semantic_type():
    from app.rag.enterprise_rag_service import EnterpriseRagService

    context = EnterpriseRagService._format_context(
        [
            {
                "source_type": "confluence",
                "metadata": {"doc_semantic_type": "policy_rule"},
                "title": "PTO Policy",
                "section_heading": "Eligibility",
                "parent_doc_id": "doc-1",
                "parent_text": "Policy text",
            }
        ]
    )

    assert "doc_semantic_type: policy_rule" in context
