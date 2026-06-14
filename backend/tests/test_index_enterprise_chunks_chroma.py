import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.index_enterprise_chunks_chroma import chunk_to_document


def test_chunk_to_document_preserves_doc_semantic_type():
    document = chunk_to_document(
        {
            "chunk_id": "chunk-1",
            "parent_doc_id": "doc-1",
            "text": "Refund policy text",
            "doc_semantic_type": "policy_doc",
        }
    )

    assert document.metadata["doc_semantic_type"] == "policy_doc"


def test_chunk_to_document_defaults_doc_semantic_type_to_generic_doc():
    document = chunk_to_document(
        {
            "chunk_id": "chunk-1",
            "parent_doc_id": "doc-1",
            "text": "General text",
        }
    )

    assert document.metadata["doc_semantic_type"] == "generic_doc"
