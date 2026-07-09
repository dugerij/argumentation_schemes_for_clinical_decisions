from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import fsspec
from llama_index.core.bridge.pydantic import Field
from llama_index.core.graph_stores.types import (
    ChunkNode,
    DEFAULT_PERSIST_DIR,
    DEFAULT_PG_PERSIST_FNAME,
    EntityNode,
    LabelledNode,
    LabelledPropertyGraph,
    PropertyGraphStore,
    Relation,
)


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MERGED_LIST_KEYS = {"source_names", "source_paths", "sections", "evidence_samples", "aliases"}
_MERGED_COUNT_KEYS = {"mention_count", "evidence_count"}
_SINGULAR_TO_LIST_KEYS = {
    "source_name": "source_names",
    "source_path": "source_paths",
    "section": "sections",
    "evidence": "evidence_samples",
}
_DEFAULT_LIST_LIMIT = 12
_EVIDENCE_LIST_LIMIT = 8


def _slugify(value: str) -> str:
    compact = " ".join(str(value).split()).strip().lower()
    slug = _SLUG_RE.sub("_", compact).strip("_")
    return slug or "unknown"


def canonical_entity_id(label: str, name: str) -> str:
    return f"{_slugify(label)}:{_slugify(name)}"


def _as_unique_list(
    values: Any,
    *,
    limit: int = _DEFAULT_LIST_LIMIT,
) -> list[str]:
    if values is None:
        return []

    raw_values = list(values) if isinstance(values, (list, tuple, set)) else [values]
    merged: list[str] = []
    seen: set[str] = set()

    for value in raw_values:
        text = " ".join(str(value).split()).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
        if len(merged) >= limit:
            break

    return merged


def merge_graph_properties(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(existing or {})
    updates = dict(incoming or {})

    for singular_key, list_key in _SINGULAR_TO_LIST_KEYS.items():
        if singular_key in merged and list_key not in merged:
            limit = _EVIDENCE_LIST_LIMIT if list_key == "evidence_samples" else _DEFAULT_LIST_LIMIT
            merged[list_key] = _as_unique_list(merged[singular_key], limit=limit)

    for key, value in updates.items():
        if value is None or value == "" or value == []:
            continue

        if key in _MERGED_COUNT_KEYS:
            merged[key] = int(merged.get(key, 0) or 0) + int(value)
            continue

        if key in _MERGED_LIST_KEYS:
            limit = _EVIDENCE_LIST_LIMIT if key == "evidence_samples" else _DEFAULT_LIST_LIMIT
            merged[key] = _as_unique_list(
                [*merged.get(key, []), *_as_unique_list(value, limit=limit)],
                limit=limit,
            )
            continue

        if key in _SINGULAR_TO_LIST_KEYS:
            if not merged.get(key):
                merged[key] = value
            list_key = _SINGULAR_TO_LIST_KEYS[key]
            limit = _EVIDENCE_LIST_LIMIT if list_key == "evidence_samples" else _DEFAULT_LIST_LIMIT
            merged[list_key] = _as_unique_list(
                [*merged.get(list_key, []), *_as_unique_list(value, limit=limit)],
                limit=limit,
            )
            continue

        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value

    return merged


class ClinicalEntityNode(EntityNode):
    canonical_id: str | None = Field(
        default=None,
        description="Internal normalized identifier used for graph deduplication.",
    )

    @property
    def id(self) -> str:
        return self.canonical_id or super().id


def _node_priority(node: LabelledNode) -> tuple[int, int, int]:
    label = str(getattr(node, "label", "") or "")
    properties = getattr(node, "properties", {}) or {}
    is_chunk = isinstance(node, ChunkNode) or label == "text_chunk"
    explicit_label = label not in {"", "entity", "text_chunk"}
    return (
        1 if is_chunk else 0,
        1 if explicit_label else 0,
        len(properties),
    )


def _merge_nodes(existing: LabelledNode | None, incoming: LabelledNode) -> LabelledNode:
    if existing is None:
        return incoming

    merged_properties = merge_graph_properties(
        getattr(existing, "properties", {}) or {},
        getattr(incoming, "properties", {}) or {},
    )
    chosen = incoming if _node_priority(incoming) >= _node_priority(existing) else existing

    if isinstance(chosen, ChunkNode):
        merged = chosen.model_copy(deep=True)
        merged.properties = merged_properties
        return merged

    chosen_name = str(getattr(chosen, "name", "") or "")
    if chosen_name == str(getattr(chosen, "id", "")) and getattr(existing, "name", None):
        chosen_name = str(getattr(existing, "name"))
    if chosen_name == str(getattr(chosen, "id", "")) and getattr(incoming, "name", None):
        chosen_name = str(getattr(incoming, "name"))

    canonical_id = getattr(incoming, "canonical_id", None) or getattr(existing, "canonical_id", None)
    return ClinicalEntityNode(
        name=chosen_name or str(getattr(chosen, "id", "")),
        label=str(getattr(chosen, "label", "entity") or "entity"),
        canonical_id=canonical_id,
        properties=merged_properties,
    )


def _merge_relations(existing: Relation | None, incoming: Relation) -> Relation:
    if existing is None:
        return incoming
    return Relation(
        label=incoming.label or existing.label,
        source_id=incoming.source_id or existing.source_id,
        target_id=incoming.target_id or existing.target_id,
        properties=merge_graph_properties(existing.properties, incoming.properties),
    )


class ClinicalLabelledPropertyGraph(LabelledPropertyGraph):
    def add_node(self, node: LabelledNode) -> None:
        self.nodes[node.id] = _merge_nodes(self.nodes.get(node.id), node)

    def add_triplet(self, triplet) -> None:
        subj, rel, obj = triplet
        rel_key = self._get_relation_key(relation=rel)

        existing_subj = self.nodes.get(subj.id)
        existing_obj = self.nodes.get(obj.id)
        self.nodes[subj.id] = existing_subj if existing_subj is subj else _merge_nodes(existing_subj, subj)
        self.nodes[obj.id] = existing_obj if existing_obj is obj else _merge_nodes(existing_obj, obj)

        if (subj.id, rel.id, obj.id) in self.triplets:
            self.relations[rel_key] = _merge_relations(self.relations.get(rel_key), rel)
            return

        self.triplets.add((subj.id, rel.id, obj.id))
        self.relations[rel_key] = _merge_relations(self.relations.get(rel_key), rel)

    def add_relation(self, relation: Relation) -> None:
        if relation.source_id not in self.nodes:
            self.nodes[relation.source_id] = EntityNode(name=relation.source_id)
        if relation.target_id not in self.nodes:
            self.nodes[relation.target_id] = EntityNode(name=relation.target_id)

        self.add_triplet(
            (self.nodes[relation.source_id], relation, self.nodes[relation.target_id])
        )


class ClinicalPropertyGraphStore(PropertyGraphStore):
    supports_structured_queries: bool = False
    supports_vector_queries: bool = False

    def __init__(
        self,
        graph: Optional[ClinicalLabelledPropertyGraph] = None,
    ) -> None:
        self.graph = graph or ClinicalLabelledPropertyGraph()

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        nodes = list(self.graph.nodes.values())
        if properties:
            nodes = [
                node
                for node in nodes
                if any(node.properties.get(key) == value for key, value in properties.items())
            ]
        if ids:
            nodes = [node for node in nodes if node.id in ids]
        return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ):
        if not ids and not properties and not entity_names and not relation_names:
            return []

        triplets = self.graph.get_triplets()
        if entity_names:
            triplets = [
                triplet
                for triplet in triplets
                if triplet[0].id in entity_names or triplet[2].id in entity_names
            ]
        if relation_names:
            triplets = [triplet for triplet in triplets if triplet[1].id in relation_names]
        if properties:
            triplets = [
                triplet
                for triplet in triplets
                if any(
                    triplet[0].properties.get(key) == value
                    or triplet[1].properties.get(key) == value
                    or triplet[2].properties.get(key) == value
                    for key, value in properties.items()
                )
            ]
        if ids:
            triplets = [
                triplet
                for triplet in triplets
                if any(triplet[0].id == node_id or triplet[2].id == node_id for node_id in ids)
            ]
        return triplets

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ):
        triplets = []
        cur_depth = 0
        graph_triplets = self.get_triplets(ids=[graph_node.id for graph_node in graph_nodes])
        seen_triplets = set()

        while len(graph_triplets) > 0 and cur_depth < depth:
            triplets.extend(graph_triplets)
            graph_triplets = self.get_triplets(entity_names=[triplet[2].id for triplet in graph_triplets])
            graph_triplets = [triplet for triplet in graph_triplets if str(triplet) not in seen_triplets]
            seen_triplets.update(str(triplet) for triplet in graph_triplets)
            cur_depth += 1

        ignore_rels = ignore_rels or []
        triplets = [triplet for triplet in triplets if triplet[1].id not in ignore_rels]
        return triplets[:limit]

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        for node in nodes:
            self.graph.add_node(node)

    def upsert_relations(self, relations: List[Relation]) -> None:
        for relation in relations:
            self.graph.add_relation(relation)

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        triplets = self.get_triplets(
            entity_names=entity_names,
            relation_names=relation_names,
            properties=properties,
            ids=ids,
        )
        for triplet in triplets:
            self.graph.delete_triplet(triplet)

        nodes = self.get(properties=properties, ids=ids)
        for node in nodes:
            self.graph.delete_node(node)

    def persist(
        self,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        fs = fs or fsspec.filesystem("file")
        with fs.open(persist_path, "w", encoding="utf-8") as file:
            file.write(self.graph.model_dump_json())

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "ClinicalPropertyGraphStore":
        fs = fs or fsspec.filesystem("file")
        with fs.open(persist_path, "r", encoding="utf-8") as file:
            data = json.loads(file.read())
        return cls.from_dict(data)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "ClinicalPropertyGraphStore":
        persist_path = os.path.join(persist_dir, DEFAULT_PG_PERSIST_FNAME)
        return cls.from_persist_path(persist_path, fs=fs)

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "ClinicalPropertyGraphStore":
        node_dicts = data["nodes"]
        graph_nodes: Dict[str, LabelledNode] = {}
        for node_id, node_dict in node_dicts.items():
            if "text" in node_dict:
                graph_nodes[node_id] = ChunkNode.model_validate(node_dict)
            elif node_dict.get("canonical_id"):
                graph_nodes[node_id] = ClinicalEntityNode.model_validate(node_dict)
            elif "name" in node_dict:
                graph_nodes[node_id] = EntityNode.model_validate(node_dict)
            else:
                raise ValueError(f"Could not infer node type for data: {node_dict!s}")

        data["nodes"] = {}
        graph = ClinicalLabelledPropertyGraph.model_validate(data)
        graph.nodes = graph_nodes
        return cls(graph)

    def to_dict(self) -> dict:
        return self.graph.model_dump()

    def get_schema(self, refresh: bool = False) -> str:
        raise NotImplementedError(
            "Schema not implemented for ClinicalPropertyGraphStore."
        )

    def structured_query(self, query: str, param_map: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError(
            "Structured query not implemented for ClinicalPropertyGraphStore."
        )

    def vector_query(self, query, **kwargs: Any):
        raise NotImplementedError(
            "Vector query not implemented for ClinicalPropertyGraphStore."
        )

    @property
    def client(self) -> Any:
        raise NotImplementedError(
            "Client not implemented for ClinicalPropertyGraphStore."
        )
