"""Diagnosis-agnostic compilation of sourced graph claims into evidence roles."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Iterable

from clinical_cds.argumentation import KnowledgeRole, knowledge_role
from clinical_cds.direct import label_key
from clinical_cds.schema import DiagnosticGraph, FamilyChildFact, RetrievedFact
from clinical_cds.typed_binding import atomic_criteria


EVIDENCE_GRAPH_COMPILER_ID = "sourced-clinical-evidence-graph-v1"
DECISIVE_ANCHOR_COMPILER_ID = "sourced-decisive-anchor-v1"
TEST_RE = re.compile(
    r"\b(?:test|testing|imaging|scan|radiograph|x[- ]?ray|ct|mri|ultrasound|"
    r"echo|endoscop|biopsy|histolog|patholog|culture|pcr|spirometr|"
    r"pulmonary function|ecg|ekg|assay|laboratory|level)\b",
    re.IGNORECASE,
)
MEASUREMENT_RE = re.compile(r"(?:<=|>=|≤|≥|<|>|\d+(?:\.\d+)?\s*(?:%|mmhg|mg/dl|mmol/l))", re.IGNORECASE)
CONJUNCTION_RE = re.compile(r"\b(?:and|all of|together with|plus)\b", re.IGNORECASE)
NEGATIVE_RE = re.compile(r"\b(?:absence|absent|negative|normal|without|no )\b", re.IGNORECASE)
# A negation marker that only appears inside one of these constructions is not
# excluding the claim it sits in -- it is either an inclusive "either way"
# qualifier ("with or without X") or a comparative clause naming what a
# *different*, contrasting diagnosis is defined/characterized by, which makes
# the sourced claim itself a positive defining criterion (see
# clinical_cds/knowledge/Gastritis.provenance.json: "...without the gland
# loss...that defines atrophic gastritis" is a defining criterion for the
# non-atrophic branch, not an exclusion of it).
NEGATION_EXCEPTION_RE = re.compile(
    r"\bwith or without\b|"
    r"\b(?:that|which)\s+(?:define|defines|characteriz(?:e|es))\b",
    re.IGNORECASE,
)


def _is_excluding(text: str) -> bool:
    return bool(NEGATIVE_RE.search(text)) and not NEGATION_EXCEPTION_RE.search(text)


class PropositionKind(StrEnum):
    DIAGNOSTIC_TEST = "diagnostic_test"
    MEASUREMENT = "measurement"
    CLINICAL_FINDING = "clinical_finding"
    RISK_CONTEXT = "risk_context"
    GUIDELINE = "guideline"


class EvidenceRelation(StrEnum):
    DIRECT_SUPPORT = "direct_support"
    WEAK_SUPPORT = "weak_support"
    CONTRADICTION_OR_EXCLUSION = "contradiction_or_exclusion"
    AUTHORITY = "authority"


class PropositionScope(StrEnum):
    FAMILY = "family"
    CHILD = "child"


@dataclass(frozen=True)
class EvidenceProposition:
    proposition_id: str
    graph_category: str
    diagnosis_label: str
    diagnostic_path: tuple[str, ...]
    parent_node_id: str
    parent_text: str
    text: str
    premise_type: str
    knowledge_role: KnowledgeRole
    kind: PropositionKind
    relation: EvidenceRelation
    scope: PropositionScope
    conjunctive: bool
    knowledge_source_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceGraphAudit:
    compiler_id: str
    graph_category: str
    source_premise_count: int
    proposition_count: int
    provenance_complete: bool
    atomic_text_complete: bool
    family_scope_count: int
    child_scope_count: int
    diagnostic_test_count: int
    measurement_count: int
    conjunctive_count: int


@dataclass(frozen=True)
class DecisiveAnchor:
    """A high-specificity, sourced graph claim suitable for case-level audit.

    This is a derived view, not new clinical knowledge.  It deliberately keeps
    the original graph node, wording, path, and source identifiers so a later
    resolver can only use a claim that was already present in the controlled
    graph.
    """

    anchor_id: str
    graph_category: str
    diagnosis_label: str
    diagnostic_path: tuple[str, ...]
    parent_node_id: str
    parent_text: str
    text: str
    premise_type: str
    kind: PropositionKind
    modalities: tuple[str, ...]
    threshold_bearing: bool
    knowledge_source_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DecisiveAnchorAudit:
    compiler_id: str
    graph_category: str
    source_premise_count: int
    anchor_count: int
    diagnosis_count: int
    modality_anchor_count: int
    threshold_anchor_count: int
    provenance_complete: bool


def _proposition_kind(text: str, role: KnowledgeRole) -> PropositionKind:
    if role == KnowledgeRole.RISK_FACTOR:
        return PropositionKind.RISK_CONTEXT
    if role == KnowledgeRole.GUIDELINE:
        return PropositionKind.GUIDELINE
    if MEASUREMENT_RE.search(text):
        return PropositionKind.MEASUREMENT
    if TEST_RE.search(text):
        return PropositionKind.DIAGNOSTIC_TEST
    return PropositionKind.CLINICAL_FINDING


def _relation(text: str, role: KnowledgeRole) -> EvidenceRelation:
    if _is_excluding(text):
        return EvidenceRelation.CONTRADICTION_OR_EXCLUSION
    if role == KnowledgeRole.RISK_FACTOR:
        return EvidenceRelation.WEAK_SUPPORT
    if role == KnowledgeRole.GUIDELINE:
        return EvidenceRelation.AUTHORITY
    return EvidenceRelation.DIRECT_SUPPORT


def compile_evidence_graph(graph: DiagnosticGraph) -> tuple[EvidenceProposition, ...]:
    """Atomize sourced premises while preserving conjunctions and provenance."""
    output: list[EvidenceProposition] = []
    for node in graph.nodes:
        if node.kind != "premise" or not node.text or not node.diagnosis_label:
            continue
        role = knowledge_role(node.premise_type or "")
        path = tuple(
            graph.diagnostic_paths.get(label_key(node.diagnosis_label))
            or (node.diagnosis_label,)
        )
        scope = PropositionScope.FAMILY if len(path) <= 1 else PropositionScope.CHILD
        for index, atom in enumerate(atomic_criteria(node.text), 1):
            text = atom.text
            identity = "|".join((graph.category, node.node_id, str(index), text))
            output.append(EvidenceProposition(
                proposition_id="ep:" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                graph_category=graph.category,
                diagnosis_label=node.diagnosis_label,
                diagnostic_path=tuple(path),
                parent_node_id=node.node_id,
                parent_text=node.text,
                text=text,
                premise_type=node.premise_type or "Clinical premise",
                knowledge_role=role,
                kind=_proposition_kind(text, role),
                relation=_relation(text, role),
                scope=scope,
                conjunctive=bool(CONJUNCTION_RE.search(text)),
                knowledge_source_ids=tuple(node.knowledge_source_ids),
            ))
    return tuple(output)


def _anchor_modalities(text: str) -> tuple[str, ...]:
    """Return only explicit diagnostic modalities named by a sourced claim."""
    names = []
    for name, pattern in (
        ("ctpa", r"\b(?:ctpa|ct pulmonary angiograph(?:y|ram))\b"),
        ("ct", r"\bct(?: scan| imaging)?\b"),
        ("mri", r"\bmri?\b"),
        ("endoscopy", r"\b(?:endoscop(?:y|ic)|egd)\b"),
        ("spirometry", r"\b(?:spirometr(?:y|ic)|pulmonary function test|pft)\b"),
        ("echocardiography", r"\b(?:echo|echocardiogram|echocardiography)\b"),
        ("ecg", r"\b(?:ecg|ekg|electrocardiogra(?:m|phy))\b"),
        ("angiography", r"\bangiograph(?:y|ram)\b"),
        ("pathology", r"\b(?:biopsy|pathology|histology)\b"),
        ("culture", r"\bculture\b"),
        ("pcr", r"\bpcr\b"),
    ):
        if re.search(pattern, text, re.IGNORECASE):
            names.append(name)
    return tuple(names)


def qualifies_decisive_anchor(
    text: str,
    role: KnowledgeRole,
) -> bool:
    """Whether one existing claim qualifies for the derived anchor view."""
    return (
        role == KnowledgeRole.DIAGNOSTIC_CRITERION
        and not _is_excluding(text)
        and bool(_anchor_modalities(text) or MEASUREMENT_RE.search(text))
    )


def compile_decisive_anchors(graph: DiagnosticGraph) -> tuple[DecisiveAnchor, ...]:
    """Derive auditable test/threshold anchors from existing sourced premises.

    Risk factors, symptoms, signs, guidance, exclusions, and generic clinical
    findings are intentionally excluded.  The rule is category-independent:
    an anchor must be a direct diagnostic criterion and explicitly name either
    a diagnostic modality or a numeric threshold.
    """
    output: list[DecisiveAnchor] = []
    for proposition in compile_evidence_graph(graph):
        modalities = _anchor_modalities(proposition.text)
        threshold_bearing = bool(MEASUREMENT_RE.search(proposition.text))
        if not qualifies_decisive_anchor(
            proposition.text, proposition.knowledge_role
        ):
            continue
        identity = "|".join((
            graph.category,
            proposition.diagnosis_label,
            proposition.parent_node_id,
            proposition.proposition_id,
        ))
        output.append(DecisiveAnchor(
            anchor_id="da:" + hashlib.sha256(identity.encode()).hexdigest()[:20],
            graph_category=graph.category,
            diagnosis_label=proposition.diagnosis_label,
            diagnostic_path=proposition.diagnostic_path,
            parent_node_id=proposition.parent_node_id,
            parent_text=proposition.parent_text,
            text=proposition.text,
            premise_type=proposition.premise_type,
            kind=proposition.kind,
            modalities=modalities,
            threshold_bearing=threshold_bearing,
            knowledge_source_ids=proposition.knowledge_source_ids,
        ))
    return tuple(output)


def audit_decisive_anchors(
    graph: DiagnosticGraph,
    anchors: Iterable[DecisiveAnchor],
) -> DecisiveAnchorAudit:
    values = tuple(anchors)
    return DecisiveAnchorAudit(
        compiler_id=DECISIVE_ANCHOR_COMPILER_ID,
        graph_category=graph.category,
        source_premise_count=sum(node.kind == "premise" for node in graph.nodes),
        anchor_count=len(values),
        diagnosis_count=len({item.diagnosis_label for item in values}),
        modality_anchor_count=sum(bool(item.modalities) for item in values),
        threshold_anchor_count=sum(item.threshold_bearing for item in values),
        provenance_complete=all(item.knowledge_source_ids for item in values),
    )


def compile_decisive_anchor_graphs(
    graphs: Iterable[DiagnosticGraph],
) -> dict[str, tuple[DecisiveAnchor, ...]]:
    """Compile the same anchor rule for every controlled graph category."""
    output: dict[str, tuple[DecisiveAnchor, ...]] = {}
    for graph in graphs:
        if graph.category in output:
            raise ValueError(f"Duplicate graph category: {graph.category}")
        output[graph.category] = compile_decisive_anchors(graph)
    return output


def audit_evidence_graph(
    graph: DiagnosticGraph,
    propositions: Iterable[EvidenceProposition],
) -> EvidenceGraphAudit:
    values = tuple(propositions)
    premise_nodes = tuple(node for node in graph.nodes if node.kind == "premise")
    parent_ids = {item.parent_node_id for item in values}
    return EvidenceGraphAudit(
        compiler_id=EVIDENCE_GRAPH_COMPILER_ID,
        graph_category=graph.category,
        source_premise_count=len(premise_nodes),
        proposition_count=len(values),
        provenance_complete=all(item.knowledge_source_ids for item in values),
        atomic_text_complete=all(item.text.strip() and item.parent_text.strip() for item in values),
        family_scope_count=sum(item.scope == PropositionScope.FAMILY for item in values),
        child_scope_count=sum(item.scope == PropositionScope.CHILD for item in values),
        diagnostic_test_count=sum(item.kind == PropositionKind.DIAGNOSTIC_TEST for item in values),
        measurement_count=sum(item.kind == PropositionKind.MEASUREMENT for item in values),
        conjunctive_count=sum(item.conjunctive for item in values),
    )


def compile_evidence_graphs(
    graphs: Iterable[DiagnosticGraph],
) -> dict[str, tuple[EvidenceProposition, ...]]:
    """Apply the same compiler to every controlled graph category."""
    output: dict[str, tuple[EvidenceProposition, ...]] = {}
    for graph in graphs:
        if graph.category in output:
            raise ValueError(f"Duplicate graph category: {graph.category}")
        output[graph.category] = compile_evidence_graph(graph)
    return output


def atomize_retrieved_family_facts(
    family_child_facts: Iterable[FamilyChildFact],
) -> tuple[FamilyChildFact, ...]:
    """Expose atomic views of existing retrieved facts without adding knowledge.

    Derived views retain the original node and source identities. Consequently,
    several clauses from one parent node remain one independence cluster.
    """
    output: list[FamilyChildFact] = []
    for item in family_child_facts:
        fact = item.fact
        atoms = atomic_criteria(fact.text)
        for index, atom in enumerate(atoms, 1):
            text = atom.text
            identity = "|".join((fact.evidence_id, fact.node_id, str(index), text))
            evidence_id = "KFA-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
            output.append(FamilyChildFact(
                graph_id=item.graph_id,
                fact=RetrievedFact(
                    evidence_id=evidence_id,
                    node_id=fact.node_id,
                    category=fact.category,
                    diagnosis_label=fact.diagnosis_label,
                    premise_type=fact.premise_type,
                    text=text,
                    score=fact.score,
                    diagnostic_path=fact.diagnostic_path,
                    knowledge_source_ids=fact.knowledge_source_ids,
                    source_chunk_id=fact.source_chunk_id,
                ),
            ))
    return tuple(output)
