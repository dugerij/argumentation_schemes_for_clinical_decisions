from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ExperimentMode(StrEnum):
    DIRECT = "direct"
    FLAT_RAG = "flat_rag"
    GRAPH_RAG = "graph_rag"
    STRUCTURED_ARGUMENT = "structured_argument"
    SYMBOLIC_ARGUMENT = "symbolic_argument"


@dataclass(frozen=True)
class AnnotationNode:
    node_id: str
    text: str
    role: str
    source_section: str | None = None
    annotation_index: int | None = None


@dataclass(frozen=True)
class AnnotationEdge:
    source_id: str
    target_id: str
    relation: str = "supports"


@dataclass(frozen=True)
class ClinicalCase:
    case_id: str
    dataset: str
    task: str
    sections: dict[str, str]
    gold_label: str
    options: dict[str, str] = field(default_factory=dict)
    disease_category: str | None = None
    directory_label: str | None = None
    annotation_nodes: tuple[AnnotationNode, ...] = ()
    annotation_edges: tuple[AnnotationEdge, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()

    @property
    def gold_observations(self) -> tuple[AnnotationNode, ...]:
        return tuple(node for node in self.annotation_nodes if node.role == "observation")

    @property
    def text(self) -> str:
        return "\n".join(
            f"{name.replace('_', ' ').title()}: {value}"
            for name, value in self.sections.items()
            if value.strip()
        )


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    label: str
    kind: str
    category: str
    text: str | None = None
    premise_type: str | None = None
    diagnosis_label: str | None = None


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation: str


@dataclass(frozen=True)
class DiagnosticGraph:
    graph_id: str
    category: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    diagnostic_paths: dict[str, tuple[str, ...]]

    def node_index(self) -> dict[str, GraphNode]:
        return {node.node_id: node for node in self.nodes}

    @property
    def diagnosis_labels(self) -> tuple[str, ...]:
        return tuple(node.label for node in self.nodes if node.kind == "diagnosis")

    @property
    def leaf_labels(self) -> tuple[str, ...]:
        parents = {
            edge.source_id
            for edge in self.edges
            if edge.relation == "refines_to"
        }
        return tuple(
            node.label
            for node in self.nodes
            if node.kind == "diagnosis" and node.node_id not in parents
        )


@dataclass(frozen=True)
class RetrievedFact:
    evidence_id: str
    node_id: str
    category: str
    diagnosis_label: str
    premise_type: str
    text: str
    score: float
    diagnostic_path: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalBundle:
    facts: tuple[RetrievedFact, ...]
    query_tokens: tuple[str, ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(fact.evidence_id for fact in self.facts)


@dataclass(frozen=True)
class PredictedObservation:
    text: str
    source_id: str | None = None


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    case_id: str
    dataset: str
    task: str
    mode: ExperimentMode
    model_id: str
    gold_label: str
    predicted_label: str
    reasoning: str
    citations: tuple[str, ...]
    observations: tuple[PredictedObservation, ...]
    abstained: bool
    latency_seconds: float
    prompt_hash: str
    cache_hit: bool
    valid_evidence_ids: tuple[str, ...]
    quality_flags: tuple[str, ...] = ()
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload
