"""Experimental offline Graph RAG extraction path."""

import re
from dataclasses import dataclass
from typing import Any

from app.rag.graph_index_service import EntityNode, normalize_entity_name, stable_entity_id, tokenize


@dataclass
class GraphExtractionResult:
    entities: list[EntityNode]
    relationships: list[dict[str, Any]]


def extract_graph_from_parent_chunk(
    row: dict[str, Any],
    max_entities: int = 8,
    max_relations: int = 8,
    max_relation_keywords: int = 6,
) -> GraphExtractionResult:
    source_chunk_ids = _source_chunk_ids(row)
    entities_by_normalized_name: dict[str, EntityNode] = {}

    for key, entity_type in (
        ("title", "title"),
        ("section_heading", "section"),
        ("source_type", "source_type"),
    ):
        _add_entity(
            entities_by_normalized_name,
            row.get(key),
            entity_type,
            source_chunk_ids,
            max_entities,
        )

    relationships: list[dict[str, Any]] = []
    for source, target in _extract_requires_phrases(str(row.get("text") or "")):
        if len(relationships) >= max_relations:
            break

        source_entity = _add_entity(entities_by_normalized_name, source, "concept", source_chunk_ids, max_entities)
        target_entity = _add_entity(entities_by_normalized_name, target, "concept", source_chunk_ids, max_entities)
        if source_entity is None or target_entity is None:
            continue

        relationships.append(
            {
                "relation_id": f"{source_entity.entity_id}__requires__{target_entity.entity_id}",
                "src_entity_id": source_entity.entity_id,
                "tgt_entity_id": target_entity.entity_id,
                "relation_type": "requires",
                "source": source,
                "target": target,
                "type": "requires",
                "source_entity_id": source_entity.entity_id,
                "target_entity_id": target_entity.entity_id,
                "keywords": _ordered_keywords(
                    f"{source} requires {target}",
                    max_relation_keywords,
                ),
                "source_chunk_ids": source_chunk_ids.copy(),
            }
        )

    return GraphExtractionResult(
        entities=list(entities_by_normalized_name.values())[:max_entities],
        relationships=relationships,
    )


def _source_chunk_ids(row: dict[str, Any]) -> list[str]:
    chunk_id = row.get("parent_chunk_id")
    if chunk_id is None or str(chunk_id).strip() == "":
        return []
    return [str(chunk_id)]


def _add_entity(
    entities_by_normalized_name: dict[str, EntityNode],
    raw_name: Any,
    entity_type: str,
    source_chunk_ids: list[str],
    max_entities: int,
) -> EntityNode | None:
    name = _clean_phrase(raw_name)
    normalized_name = normalize_entity_name(name)
    if not normalized_name:
        return None

    if normalized_name in entities_by_normalized_name:
        return entities_by_normalized_name[normalized_name]

    if len(entities_by_normalized_name) >= max_entities:
        return None

    entity = EntityNode(
        entity_id=stable_entity_id(name),
        name=name,
        normalized_name=normalized_name,
        entity_type=entity_type,
        source_chunk_ids=source_chunk_ids.copy(),
    )
    entities_by_normalized_name[normalized_name] = entity
    return entity


def _extract_requires_phrases(text: str) -> list[tuple[str, str]]:
    relations: list[tuple[str, str]] = []
    for sentence in re.split(r"[.!?]+", text):
        match = re.search(r"\b(?P<source>.+?)\s+requires\s+(?P<target>.+?)$", sentence, re.IGNORECASE)
        if not match:
            continue

        source = _clean_phrase(match.group("source"))
        target = _clean_phrase(re.split(r"\s+before\b", match.group("target"), maxsplit=1, flags=re.IGNORECASE)[0])
        if source and target:
            relations.append((source, target))
    return relations


def _clean_phrase(value: Any) -> str:
    phrase = str(value or "").strip()
    phrase = re.sub(r"^[\s\"'“”‘’.,;:()\[\]{}-]+", "", phrase)
    phrase = re.sub(r"[\s\"'“”‘’.,;:()\[\]{}-]+$", "", phrase)
    return re.sub(r"\s+", " ", phrase).strip()


def _ordered_keywords(text: str, limit: int) -> list[str]:
    if limit <= 0:
        return []

    keyword_tokens = tokenize(text)
    keywords: list[str] = []
    for token in normalize_entity_name(text).split():
        if token in keyword_tokens and token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords
