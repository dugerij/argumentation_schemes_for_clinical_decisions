from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from clinical_cds.direct import (
    DirectDataset,
    label_key,
    load_direct_dataset,
    load_direct_graph,
)
from clinical_cds.schema import DiagnosticGraph


DEFAULT_EXTENSION_ROOT = Path(__file__).with_name("knowledge")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GraphExtension:
    extension_id: str
    graph: DiagnosticGraph
    source_manifest: dict[str, object]
    premise_sources: dict[str, tuple[str, ...]]
    provenance_sha256: str


def load_graph_extension(graph_path: Path) -> GraphExtension:
    graph_path = Path(graph_path)
    provenance_path = graph_path.with_name(
        f"{graph_path.stem}.provenance.json"
    )
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Graph extension provenance is missing: {provenance_path}"
        )
    graph = load_direct_graph(graph_path)
    manifest = json.loads(provenance_path.read_text(encoding="utf-8"))
    if manifest.get("category") != graph.category:
        raise ValueError(
            f"Graph extension category mismatch for {graph_path}."
        )
    extension_id = str(manifest.get("extension_id") or "").strip()
    if not extension_id:
        raise ValueError("Graph extension ID is missing.")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Graph extension sources are missing.")
    for source_id, source in sources.items():
        if (
            not str(source_id).strip()
            or not isinstance(source, dict)
            or not str(source.get("title") or "").strip()
            or not str(source.get("publisher") or "").strip()
            or not str(source.get("url") or "").startswith("https://")
        ):
            raise ValueError(f"Invalid graph source: {source_id}")

    claims = manifest.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Graph extension claims are missing.")
    sources_by_text: dict[str, tuple[str, ...]] = {}
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("Graph extension claim must be an object.")
        text = " ".join(str(claim.get("text") or "").split())
        source_ids = tuple(
            str(source_id) for source_id in claim.get("source_ids") or ()
        )
        if (
            not text
            or not source_ids
            or len(source_ids) != len(set(source_ids))
            or not set(source_ids) <= set(sources)
            or text in sources_by_text
        ):
            raise ValueError(f"Invalid graph extension claim: {text}")
        sources_by_text[text] = source_ids

    premise_nodes = tuple(
        node for node in graph.nodes if node.kind == "premise"
    )
    premise_texts = {
        " ".join(str(node.text or "").split()) for node in premise_nodes
    }
    if premise_texts != set(sources_by_text):
        missing = sorted(premise_texts - set(sources_by_text))
        extra = sorted(set(sources_by_text) - premise_texts)
        raise ValueError(
            f"Graph extension claim coverage mismatch; missing={missing}, "
            f"extra={extra}."
        )
    premise_sources = {
        node.node_id: sources_by_text[" ".join(str(node.text).split())]
        for node in premise_nodes
    }
    graph = replace(
        graph,
        nodes=tuple(
            replace(
                node,
                knowledge_source_ids=premise_sources[node.node_id],
            )
            if node.node_id in premise_sources
            else node
            for node in graph.nodes
        ),
    )
    return GraphExtension(
        extension_id=extension_id,
        graph=graph,
        source_manifest=manifest,
        premise_sources=premise_sources,
        provenance_sha256=_canonical_sha256(manifest),
    )


def load_project_graph_extensions(
    root: Path = DEFAULT_EXTENSION_ROOT,
) -> tuple[GraphExtension, ...]:
    root = Path(root)
    graph_paths = tuple(
        path
        for path in sorted(root.glob("*.json"))
        if not path.name.endswith(".provenance.json")
    )
    if not graph_paths:
        raise FileNotFoundError(f"No graph extensions found under {root}.")
    extensions = tuple(load_graph_extension(path) for path in graph_paths)
    categories = [extension.graph.category for extension in extensions]
    if len(categories) != len(set(categories)):
        raise ValueError("Graph extension categories must be unique.")
    return extensions


def extend_direct_dataset(
    direct: DirectDataset,
    extensions: Iterable[GraphExtension],
) -> DirectDataset:
    extension_list = tuple(extensions)
    existing_categories = {graph.category for graph in direct.graphs}
    collisions = tuple(
        extension.graph.category
        for extension in extension_list
        if extension.graph.category in existing_categories
    )
    if collisions:
        raise ValueError(
            f"Graph extensions collide with source graphs: {collisions}"
        )
    graphs = direct.graphs + tuple(
        extension.graph for extension in extension_list
    )
    graph_categories = {label_key(graph.category) for graph in graphs}
    diagnosis_labels = {
        label_key(node.label)
        for graph in graphs
        for node in graph.nodes
        if node.kind == "diagnosis"
    }
    missing_graph_categories = tuple(sorted({
        str(case.disease_category)
        for case in direct.cases
        if case.disease_category
        and label_key(case.disease_category) not in graph_categories
    }))
    outside_by_key = {
        label_key(case.gold_label): case.gold_label
        for case in direct.cases
        if label_key(case.gold_label) not in diagnosis_labels
    }
    flag_counts = Counter(
        flag for case in direct.cases for flag in case.quality_flags
    )
    audit = replace(
        direct.audit,
        graph_count=len(graphs),
        disease_category_count=len({
            case.disease_category for case in direct.cases
        }),
        missing_graph_categories=missing_graph_categories,
        conclusions_outside_graph=tuple(sorted(outside_by_key.values())),
        quality_flag_counts=dict(flag_counts),
    )
    return replace(direct, graphs=graphs, audit=audit)


def load_direct_dataset_with_project_extensions(
    root: Path,
    *,
    extension_root: Path = DEFAULT_EXTENSION_ROOT,
) -> tuple[DirectDataset, tuple[GraphExtension, ...]]:
    direct = load_direct_dataset(root)
    extensions = load_project_graph_extensions(extension_root)
    return extend_direct_dataset(direct, extensions), extensions
