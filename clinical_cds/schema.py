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
    EVIDENCE_GROUNDED_ARGUMENTATION = "evidence_grounded_argumentation"


LEGACY_EXPERIMENT_MODES = (
    ExperimentMode.DIRECT,
    ExperimentMode.FLAT_RAG,
    ExperimentMode.GRAPH_RAG,
    ExperimentMode.STRUCTURED_ARGUMENT,
    ExperimentMode.SYMBOLIC_ARGUMENT,
)


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
    knowledge_source_ids: tuple[str, ...] = ()
    source_chunk_id: str = ""


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
    knowledge_source_ids: tuple[str, ...] = ()
    source_chunk_id: str = ""


@dataclass(frozen=True)
class FamilyDiagnosisAlternative:
    candidate_id: str
    diagnosis_label: str
    graph_id: str
    diagnostic_path: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    original_candidate_rank: int
    representative: bool = False


@dataclass(frozen=True)
class RetrievedFamilyRoute:
    family_rank: int
    graph_id: str
    family_key: str
    representative_diagnosis: str
    alternatives: tuple[FamilyDiagnosisAlternative, ...]


@dataclass(frozen=True)
class FamilyChildFact:
    graph_id: str
    fact: RetrievedFact


@dataclass(frozen=True)
class RetrievalBundle:
    facts: tuple[RetrievedFact, ...]
    query_tokens: tuple[str, ...]
    citation_allowlist: tuple[str, ...] = ()
    family_routes: tuple[RetrievedFamilyRoute, ...] = ()
    family_child_facts: tuple[FamilyChildFact, ...] = ()

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            fact.evidence_id
            for fact in (
                *self.facts,
                *(item.fact for item in self.family_child_facts),
            )
        )


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictionRecord:
        return cls(
            run_id=str(payload["run_id"]),
            case_id=str(payload["case_id"]),
            dataset=str(payload["dataset"]),
            task=str(payload["task"]),
            mode=ExperimentMode(payload["mode"]),
            model_id=str(payload["model_id"]),
            gold_label=str(payload["gold_label"]),
            predicted_label=str(payload.get("predicted_label") or ""),
            reasoning=str(payload.get("reasoning") or ""),
            citations=tuple(payload.get("citations") or ()),
            observations=tuple(
                PredictedObservation(
                    text=str(item.get("text") or ""),
                    source_id=item.get("source_id"),
                )
                for item in payload.get("observations") or ()
            ),
            abstained=bool(payload.get("abstained")),
            latency_seconds=float(payload.get("latency_seconds") or 0.0),
            prompt_hash=str(payload.get("prompt_hash") or ""),
            cache_hit=bool(payload.get("cache_hit")),
            valid_evidence_ids=tuple(
                payload.get("valid_evidence_ids") or ()
            ),
            quality_flags=tuple(payload.get("quality_flags") or ()),
            error=payload.get("error"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload
