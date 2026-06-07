import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.graph_extraction import extract_graph_from_parent_chunk
from app.rag.graph_index_service import (
    EntityNode,
    GraphIndexService,
    RelationEdge,
    stable_entity_id,
)

DEFAULT_PARENT_CHUNKS_PATH = BACKEND_DIR / "data" / "enterprise_rag_bench" / "parent_chunks_parent_child.jsonl"
DEFAULT_GRAPH_DIR = BACKEND_DIR / "data" / "enterprise_rag_bench" / "graph"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an EnterpriseRAG-Bench offline graph index.")
    parser.add_argument("--parent-chunks-path", type=Path, default=DEFAULT_PARENT_CHUNKS_PATH)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-entities-per-chunk", type=int, default=8)
    parser.add_argument("--max-relations-per-chunk", type=int, default=8)
    parser.add_argument("--max-relation-keywords", type=int, default=6)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def build_graph_from_parent_chunks(
    rows: Iterable[dict[str, Any]],
    graph_dir: Path,
    limit: int | None = None,
    max_entities_per_chunk: int = 8,
    max_relations_per_chunk: int = 8,
    max_relation_keywords: int = 6,
) -> GraphIndexService:
    if limit is not None and limit < 0:
        raise ValueError("--limit must be non-negative")

    service = GraphIndexService(graph_dir=Path(graph_dir))
    relations_by_id: dict[str, RelationEdge] = {}

    processed = 0
    for row in rows:
        parent_chunk_id = row.get("parent_chunk_id")
        if parent_chunk_id is None or str(parent_chunk_id).strip() == "":
            continue
        if limit is not None and processed >= limit:
            break

        chunk_id = str(parent_chunk_id)
        normalized_row = dict(row)
        normalized_row["parent_chunk_id"] = chunk_id
        service.parent_chunks[chunk_id] = normalized_row

        extraction = extract_graph_from_parent_chunk(
            normalized_row,
            max_entities=max_entities_per_chunk,
            max_relations=max_relations_per_chunk,
            max_relation_keywords=max_relation_keywords,
        )

        for entity in extraction.entities:
            _merge_entity(service, entity)

        for relationship in extraction.relationships:
            relation = _relation_from_extraction(relationship, chunk_id)
            existing = relations_by_id.get(relation.relation_id)
            if existing is None:
                relations_by_id[relation.relation_id] = relation
                service.relations.append(relation)
            else:
                existing.source_chunk_ids = sorted(set(existing.source_chunk_ids) | set(relation.source_chunk_ids))
                existing.keywords = sorted(set(existing.keywords) | set(relation.keywords))

        processed += 1
        if limit is not None and processed >= limit:
            break

    service.entity_chunk_map = {
        entity_id: sorted(entity.source_chunk_ids)
        for entity_id, entity in service.entities.items()
    }
    return service


def _merge_entity(service: GraphIndexService, entity: EntityNode) -> None:
    existing = service.entities.get(entity.entity_id)
    if existing is None:
        service.entities[entity.entity_id] = entity.model_copy(deep=True) if hasattr(entity, "model_copy") else entity.copy(deep=True)
        service.entities[entity.entity_id].source_chunk_ids = sorted(set(entity.source_chunk_ids))
        service.entities[entity.entity_id].aliases = sorted(set(entity.aliases))
        return

    existing.source_chunk_ids = sorted(set(existing.source_chunk_ids) | set(entity.source_chunk_ids))
    existing.aliases = sorted(set(existing.aliases) | set(entity.aliases))
    if not existing.description and entity.description:
        existing.description = entity.description


def _relation_from_extraction(relationship: dict[str, Any], chunk_id: str) -> RelationEdge:
    relation_type = str(relationship.get("relation_type") or relationship.get("type") or "related_to")
    src_entity_id = str(
        relationship.get("src_entity_id")
        or relationship.get("source_entity_id")
        or stable_entity_id(str(relationship.get("source") or ""))
    )
    tgt_entity_id = str(
        relationship.get("tgt_entity_id")
        or relationship.get("target_entity_id")
        or stable_entity_id(str(relationship.get("target") or ""))
    )
    relation_id = str(relationship.get("relation_id") or f"{src_entity_id}__{relation_type}__{tgt_entity_id}")
    source_chunk_ids = relationship.get("source_chunk_ids") or [chunk_id]

    return RelationEdge(
        relation_id=relation_id,
        src_entity_id=src_entity_id,
        tgt_entity_id=tgt_entity_id,
        relation_type=relation_type,
        description=str(relationship.get("description") or ""),
        keywords=sorted({str(keyword) for keyword in relationship.get("keywords", [])}),
        weight=float(relationship.get("weight", 1.0)),
        source_chunk_ids=sorted({str(source_chunk_id) for source_chunk_id in source_chunk_ids}),
    )


def main() -> None:
    args = parse_args()
    rows = iter_jsonl(args.parent_chunks_path)
    service = build_graph_from_parent_chunks(
        rows,
        graph_dir=args.graph_dir,
        limit=args.limit,
        max_entities_per_chunk=args.max_entities_per_chunk,
        max_relations_per_chunk=args.max_relations_per_chunk,
        max_relation_keywords=args.max_relation_keywords,
    )
    service.save_sync()
    print(
        json.dumps(
            {
                "graph_dir": str(service.graph_dir),
                "parent_chunks": len(service.parent_chunks),
                "entities": len(service.entities),
                "relations": len(service.relations),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
