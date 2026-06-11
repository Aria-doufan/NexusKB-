"""Experimental offline Graph RAG indexing path."""

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EntityNode(BaseModel):
    entity_id: str
    name: str
    normalized_name: str
    entity_type: str
    description: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class RelationEdge(BaseModel):
    relation_id: str
    src_entity_id: str
    tgt_entity_id: str
    relation_type: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    weight: float = 1.0
    source_chunk_ids: list[str] = Field(default_factory=list)


class GraphRetrievedDocument(BaseModel):
    parent_chunk_id: str
    parent_doc_id: str | None = None
    source_type: str | None = None
    title: str | None = None
    section_heading: str | None = None
    text: str | None = None
    score: float = 0.0
    matched_entities: list[str] = Field(default_factory=list)
    matched_relations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def normalize_entity_name(value: str) -> str:
    normalized = re.sub(r"[^\w\s]+", " ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str) -> set[str]:
    return set(normalize_entity_name(text).split())


def stable_entity_id(name: str) -> str:
    normalized = normalize_entity_name(name)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if slug:
        return slug
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"entity_{digest}"


class GraphIndexService:
    def __init__(self, graph_dir: Path):
        self.graph_dir = Path(graph_dir)
        self.entities: dict[str, EntityNode] = {}
        self.relations: list[RelationEdge] = []
        self.entity_chunk_map: dict[str, list[str]] = {}
        self.parent_chunks: dict[str, dict[str, Any]] = {}

    async def load(self) -> None:
        await asyncio.to_thread(self.load_sync)

    async def save(self) -> None:
        await asyncio.to_thread(self.save_sync)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        depth: int = 1,
        source_hints: list[str] | None = None,
    ) -> list[GraphRetrievedDocument]:
        return await asyncio.to_thread(
            self.retrieve_sync,
            query=query,
            top_k=top_k,
            depth=depth,
            source_hints=source_hints,
        )

    def load_sync(self) -> None:
        self.entities = {}
        self.relations = []
        self.entity_chunk_map = {}
        self.parent_chunks = {}

        entities_path = self.graph_dir / "entities.jsonl"
        if entities_path.exists():
            with entities_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        entity = EntityNode(**json.loads(line))
                        self.entities[entity.entity_id] = entity

        relations_path = self.graph_dir / "relations.jsonl"
        if relations_path.exists():
            with relations_path.open("r", encoding="utf-8") as file:
                self.relations = [RelationEdge(**json.loads(line)) for line in file if line.strip()]

        entity_chunk_map_path = self.graph_dir / "entity_chunk_map.json"
        if entity_chunk_map_path.exists():
            self.entity_chunk_map = json.loads(entity_chunk_map_path.read_text(encoding="utf-8"))

        parent_chunks_path = self.graph_dir / "parent_chunks.json"
        if parent_chunks_path.exists():
            self.parent_chunks = json.loads(parent_chunks_path.read_text(encoding="utf-8"))

    def save_sync(self) -> None:
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        with (self.graph_dir / "entities.jsonl").open("w", encoding="utf-8") as file:
            for entity in self.entities.values():
                file.write(json.dumps(self._dump_model(entity), ensure_ascii=False) + "\n")

        with (self.graph_dir / "relations.jsonl").open("w", encoding="utf-8") as file:
            for relation in self.relations:
                file.write(json.dumps(self._dump_model(relation), ensure_ascii=False) + "\n")

        (self.graph_dir / "entity_chunk_map.json").write_text(
            json.dumps(self.entity_chunk_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.graph_dir / "parent_chunks.json").write_text(
            json.dumps(self.parent_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def retrieve_sync(
        self,
        query: str,
        top_k: int = 5,
        depth: int = 1,
        source_hints: list[str] | None = None,
    ) -> list[GraphRetrievedDocument]:
        normalized_query = normalize_entity_name(query)
        query_tokens = tokenize(query)
        scores: dict[str, float] = {}
        matched_entities: dict[str, set[str]] = {}
        matched_relations: dict[str, set[str]] = {}
        seed_entity_ids: set[str] = set()

        for entity in self.entities.values():
            score = self._entity_match_score(entity, normalized_query, query_tokens)
            if score <= 0:
                continue
            seed_entity_ids.add(entity.entity_id)
            for chunk_id in self.entity_chunk_map.get(entity.entity_id, entity.source_chunk_ids):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score
                matched_entities.setdefault(chunk_id, set()).add(entity.entity_id)

        if depth > 0:
            frontier = set(seed_entity_ids)
            visited = set(seed_entity_ids)
            scored_relation_ids: set[str] = set()
            for current_depth in range(depth):
                next_frontier: set[str] = set()
                for relation in self.relations:
                    touches_frontier = (
                        relation.src_entity_id in frontier or relation.tgt_entity_id in frontier
                    )
                    if not touches_frontier:
                        continue

                    for entity_id in (relation.src_entity_id, relation.tgt_entity_id):
                        if entity_id not in visited:
                            next_frontier.add(entity_id)
                            visited.add(entity_id)

                    if relation.relation_id in scored_relation_ids:
                        continue
                    scored_relation_ids.add(relation.relation_id)

                    relation_score = self._relation_match_score(relation, query_tokens)
                    if relation_score <= 0:
                        relation_score = 0.35 * relation.weight
                    depth_factor = 1.0 / (current_depth + 1)

                    for index, chunk_id in enumerate(relation.source_chunk_ids):
                        chunk_factor = max(0.1, 1.0 - (index * 0.2))
                        scores[chunk_id] = scores.get(chunk_id, 0.0) + relation_score * depth_factor * chunk_factor
                        matched_relations.setdefault(chunk_id, set()).add(relation.relation_id)
                frontier = next_frontier
                if not frontier:
                    break

        source_hint_values = {hint.lower() for hint in source_hints or []}
        if source_hint_values:
            for chunk_id in list(scores):
                source_type = str(self.parent_chunks.get(chunk_id, {}).get("source_type", "")).lower()
                if source_type in source_hint_values:
                    scores[chunk_id] += 0.15

        results = [
            self._build_result(chunk_id, score, matched_entities, matched_relations)
            for chunk_id, score in scores.items()
            if chunk_id in self.parent_chunks
        ]
        results.sort(key=lambda result: (-result.score, result.parent_chunk_id))
        return results[:top_k]

    @staticmethod
    def _dump_model(model: BaseModel) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    @staticmethod
    def _contains_token_boundary_phrase(normalized_query: str, normalized_phrase: str) -> bool:
        phrase_pattern = re.escape(normalized_phrase)
        return re.search(rf"(^|\s){phrase_pattern}(\s|$)", normalized_query) is not None

    @staticmethod
    def _entity_match_score(entity: EntityNode, normalized_query: str, query_tokens: set[str]) -> float:
        names = [entity.normalized_name] + [normalize_entity_name(alias) for alias in entity.aliases]
        best_score = 0.0
        for name in names:
            if not name:
                continue
            if GraphIndexService._contains_token_boundary_phrase(normalized_query, name):
                best_score = max(best_score, 2.0)
        return best_score

    @staticmethod
    def _relation_match_score(relation: RelationEdge, query_tokens: set[str]) -> float:
        relation_tokens = tokenize(
            " ".join([relation.relation_type, relation.description, " ".join(relation.keywords)])
        )
        overlap = relation_tokens & query_tokens
        if not overlap:
            return 0.0
        keyword_tokens = set()
        for keyword in relation.keywords:
            keyword_tokens.update(tokenize(keyword))
        keyword_overlap = keyword_tokens & query_tokens
        return relation.weight * (0.5 + (0.5 * len(overlap)) + (0.25 * len(keyword_overlap)))

    def _build_result(
        self,
        chunk_id: str,
        score: float,
        matched_entities: dict[str, set[str]],
        matched_relations: dict[str, set[str]],
    ) -> GraphRetrievedDocument:
        chunk = self.parent_chunks[chunk_id]
        return GraphRetrievedDocument(
            parent_chunk_id=chunk.get("parent_chunk_id", chunk_id),
            parent_doc_id=chunk.get("parent_doc_id"),
            source_type=chunk.get("source_type"),
            title=chunk.get("title"),
            section_heading=chunk.get("section_heading"),
            text=chunk.get("text"),
            score=score,
            matched_entities=sorted(matched_entities.get(chunk_id, set())),
            matched_relations=sorted(matched_relations.get(chunk_id, set())),
            metadata={key: value for key, value in chunk.items() if key not in {"parent_chunk_id", "parent_doc_id", "source_type", "title", "section_heading", "text"}},
        )
