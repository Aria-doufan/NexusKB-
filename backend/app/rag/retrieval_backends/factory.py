from __future__ import annotations

import os
from typing import Any

import yaml

from app.rag.retrieval_backends.chroma_enterprise import ChromaEnterpriseRetrievalBackend


CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "rag.yaml")


def load_rag_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def selected_backend_name(config: dict[str, Any] | None = None) -> str:
    env_value = os.getenv("NEXUSKB_RETRIEVAL_BACKEND")
    if env_value:
        return env_value.strip().lower()
    return str((config or load_rag_config()).get("retrieval_backend", "chroma")).strip().lower()


def build_enterprise_retrieval_backend(service=None, config: dict[str, Any] | None = None):
    backend_name = selected_backend_name(config)
    if backend_name == "chroma":
        return ChromaEnterpriseRetrievalBackend(service=service)
    if backend_name == "elasticsearch":
        from app.rag.retrieval_backends.elasticsearch_enterprise import ElasticsearchEnterpriseRetrievalBackend

        return ElasticsearchEnterpriseRetrievalBackend.from_config(config or load_rag_config())
    raise ValueError(f"Unsupported enterprise retrieval backend: {backend_name}")
