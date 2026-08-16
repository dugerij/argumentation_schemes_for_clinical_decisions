from __future__ import annotations

import hashlib

import json

import re

from dataclasses import asdict, dataclass

from enum import StrEnum

from itertools import combinations

from typing import Any, Iterable, Mapping

from clinical_cds.direct import label_key, normalize_label

from clinical_cds.normalization import (
    UMLSNormalizer,
    diagnosis_modifiers,
    diagnosis_path_compatibility,
    lexical_diagnosis_key,
)

from clinical_cds.schema import (
    ClinicalCase,
    FamilyChildFact,
    RetrievedFact,
    RetrievedFamilyRoute,
)

from clinical_cds.typed_binding import (
    Polarity,
    assess_typed_binding,
)

class ArgumentScheme(StrEnum):
    BEST_EXPLANATION = "argument_from_best_explanation"
    CLINICAL_SIGN = "argument_from_clinical_sign"
    DIAGNOSTIC_CRITERION = "argument_from_diagnostic_criterion"
    RISK_FACTOR = "argument_from_risk_factor"
    GUIDELINE_AUTHORITY = "argument_from_guideline_authority"
    NEGATIVE_EVIDENCE = "argument_from_negative_evidence"
    ALTERNATIVE_EXPLANATION = "argument_from_alternative_explanation"
    CRITICAL_QUESTION = "critical_question_challenge"

class ReviewVerdict(StrEnum):
    SUPPORTED = "supported"
    UNDERCUT = "undercut"
    UNCERTAIN = "uncertain"

class RelationType(StrEnum):
    SUPPORTS = "supports"
    REBUTS = "rebuts"
    UNDERCUTS = "undercuts"

class KnowledgeRole(StrEnum):
    CLINICAL_FEATURE = "clinical_feature"
    DIAGNOSTIC_CRITERION = "diagnostic_criterion"
    RISK_FACTOR = "risk_factor"
    COUNTEREVIDENCE = "counterevidence"
    GUIDELINE = "guideline"

class BindingApplication(StrEnum):
    SATISFIES = "satisfies"
    PARTIALLY_SATISFIES = "partially_satisfies"
    CONFLICTS = "conflicts"
    RAISES_PLAUSIBILITY = "raises_plausibility"

EVIDENCE_VALIDATION_ID = "kg-typed-binding-review-counter-grounding-v4"

BINDING_VALIDATION_ID = (
    "umls-atomic-criterion-family-only-diagnostic-test-binding-v7"
)

LLM_ENTAILMENT_BINDING_ID = "extractive-clinical-entailment-judge-v1"

SCHEME_TYPING_ID = "kg-premise-role-pre-verifier-v1"

COUNTER_TYPING_ID = "kg-relation-role-pre-resolver-v1"

KG_PATIENT_COUNTER_BINDING_ID = "explicit-negative-s-kg-binding-v1"

REVIEW_GROUNDING_ID = "target-typed-binding-review-sanitizer-v2"

NON_DIAGNOSTIC_PROBABILITY_RE = re.compile(
    r"\b(?:family history|risk factors?|raises? (?:the )?(?:risk|probability)|"
    r"increases? (?:the )?(?:risk|probability)|associated with (?:an? )?"
    r"(?:increased|higher) (?:risk|probability))\b",
    re.IGNORECASE,
)

NORMALITY_ONLY_RE = re.compile(r"\b(?:normal|within normal limits|unremarkable)\b", re.I)

def _llm_entailment_safety_passes(
    assessment: Any,
    claim: KnowledgeClaim,
) -> bool:
    """Keep non-semantic safety checks deterministic for LLM entailment."""
    questions = {
        item.question_id: item.passed
        for item in assessment.critical_questions
    }
    return (
        claim.role == KnowledgeRole.DIAGNOSTIC_CRITERION
        and assessment.criterion_atom.polarity != Polarity.ABSENT
        and not NORMALITY_ONLY_RE.search(claim.text)
        and not NON_DIAGNOSTIC_PROBABILITY_RE.search(claim.text)
        and all(questions.get(question_id, False) for question_id in (
            "observation_present",
            "temporality_compatible",
            "quantity_compatible",
            "scheme_role_compatible",
        ))
    )

def _llm_entailment_questions_valid(
    questions: Iterable[Mapping[str, Any]],
) -> bool:
    results = {
        str(item.get("question_id") or ""): bool(item.get("passed"))
        for item in questions
    }
    return all(results.get(question_id, False) for question_id in (
        "observation_present",
        "temporality_compatible",
        "quantity_compatible",
        "scheme_role_compatible",
        "clinical_entailment_judged",
    ))

MAX_CANDIDATES = 12

SUPPORT_SCHEME_KNOWLEDGE_ROLES = {
    ArgumentScheme.CLINICAL_SIGN: frozenset({KnowledgeRole.CLINICAL_FEATURE}),
    ArgumentScheme.DIAGNOSTIC_CRITERION: frozenset(
        {KnowledgeRole.DIAGNOSTIC_CRITERION}
    ),
    ArgumentScheme.RISK_FACTOR: frozenset({KnowledgeRole.RISK_FACTOR}),
    ArgumentScheme.GUIDELINE_AUTHORITY: frozenset({KnowledgeRole.GUIDELINE}),
}

KNOWLEDGE_ROLE_SUPPORT_SCHEMES = {
    role: scheme
    for scheme, roles in SUPPORT_SCHEME_KNOWLEDGE_ROLES.items()
    for role in roles
}

POSITIVE_BINDING_APPLICATIONS = {
    KnowledgeRole.CLINICAL_FEATURE: frozenset({
        BindingApplication.SATISFIES,
        BindingApplication.PARTIALLY_SATISFIES,
    }),
    KnowledgeRole.DIAGNOSTIC_CRITERION: frozenset({
        BindingApplication.SATISFIES,
        BindingApplication.PARTIALLY_SATISFIES,
    }),
    KnowledgeRole.RISK_FACTOR: frozenset({
        BindingApplication.RAISES_PLAUSIBILITY,
    }),
    KnowledgeRole.GUIDELINE: frozenset({
        BindingApplication.SATISFIES,
        BindingApplication.PARTIALLY_SATISFIES,
    }),
}

SCHEME_PRIORITY = {
    ArgumentScheme.BEST_EXPLANATION: 5,
    ArgumentScheme.DIAGNOSTIC_CRITERION: 4,
    ArgumentScheme.CLINICAL_SIGN: 3,
    ArgumentScheme.GUIDELINE_AUTHORITY: 2,
    ArgumentScheme.RISK_FACTOR: 1,
    ArgumentScheme.NEGATIVE_EVIDENCE: 4,
    ArgumentScheme.ALTERNATIVE_EXPLANATION: 3,
    ArgumentScheme.CRITICAL_QUESTION: 6,
}

@dataclass(frozen=True)
class ProposedArgument:
    argument_id: str
    scheme: ArgumentScheme
    premise: str
    conclusion: str
    evidence_ids: tuple[str, ...]
    scheme_source: str = "model"
    patient_finding: str = ""
    application: BindingApplication | None = None
    model_rationale: str = ""
    knowledge_warrant: str = ""
    knowledge_node_id: str = ""
    diagnostic_path: tuple[str, ...] = ()
    premise_type: str = ""
    knowledge_role: KnowledgeRole | None = None
    patient_source_sha256: str = ""
    knowledge_warrant_sha256: str = ""
    binding_content_sha256: str = ""
    binding_validation_id: str = ""
    binding_critical_questions: tuple[dict[str, Any], ...] = ()

@dataclass(frozen=True)
class DiagnosisCandidate:
    candidate_id: str
    diagnosis: str
    arguments: tuple[ProposedArgument, ...]

@dataclass(frozen=True)
class ReasonerProposal:
    candidates: tuple[DiagnosisCandidate, ...]
    preferred_diagnosis: str
    abstain: bool
    raw_argument_count: int
    invalid_argument_count: int
    canonicalized_argument_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ArgumentReview:
    argument_id: str
    verdict: ReviewVerdict
    failed_critical_questions: tuple[str, ...]
    explanation: str
    evidence_ids: tuple[str, ...]
    raw_verdict: ReviewVerdict | None = None
    raw_evidence_ids: tuple[str, ...] = ()
    sanitization_rule_id: str = ""
    downgrade_reason: str = ""

@dataclass(frozen=True)
class CounterArgument:
    argument_id: str
    target_argument_id: str
    scheme: ArgumentScheme
    premise: str
    conclusion: str
    evidence_ids: tuple[str, ...]
    relation: RelationType
    scheme_source: str = "model"

@dataclass(frozen=True)
class VerifierReport:
    reviews: tuple[ArgumentReview, ...]
    counterarguments: tuple[CounterArgument, ...]
    abstain: bool
    raw_review_count: int
    invalid_review_count: int
    raw_counterargument_count: int
    invalid_counterargument_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ArgumentNode:
    argument_id: str
    node_type: str
    scheme: ArgumentScheme
    premise: str
    conclusion: str
    evidence_ids: tuple[str, ...]
    candidate_id: str | None
    source: str
    priority: int
    evidence_compatibility: float = 0.0
    # Explicit provenance for knowledge claims.  Evidence ids are run-local
    # handles; these fields preserve the stable KG identity and route used.
    knowledge_node_ids: tuple[str, ...] = ()
    diagnostic_paths: tuple[tuple[str, ...], ...] = ()
    premise_types: tuple[str, ...] = ()
    scheme_source: str = "system"
    patient_finding: str = ""
    application: BindingApplication | None = None
    model_rationale: str = ""
    knowledge_warrant: str = ""
    knowledge_role: KnowledgeRole | None = None
    patient_source_sha256: str = ""
    knowledge_warrant_sha256: str = ""
    binding_content_sha256: str = ""
    binding_validation_id: str = ""
    binding_critical_questions: tuple[dict[str, Any], ...] = ()

@dataclass(frozen=True)
class ArgumentRelation:
    source_id: str
    target_id: str
    relation: RelationType

@dataclass(frozen=True)
class ArgumentGraphQuality:
    argument_schema_validity: float
    argument_evidence_validity: float
    verifier_review_coverage: float
    valid_evidence_reference_fraction: float
    argument_evidence_compatibility: float = 0.0
    argument_scheme_validity: float = 0.0
    supported_review_grounding: float = 0.0
    counterargument_evidence_validity: float = 0.0
    typed_binding_validity: float = 0.0

@dataclass(frozen=True)
class PatientEvidenceClaim:
    evidence_id: str
    section_name: str
    text: str
    content_sha256: str

@dataclass(frozen=True)
class KnowledgeClaim:
    evidence_id: str
    node_id: str
    diagnosis_label: str
    diagnostic_path: tuple[str, ...]
    premise_type: str
    text: str
    role: KnowledgeRole
    counterevidence_eligible: bool
    content_sha256: str = ""

@dataclass(frozen=True)
class CandidateInventoryEntry:
    candidate_id: str
    diagnosis: str
    canonical_key: str
    sources: tuple[str, ...]
    retrieval_rank: int | None
    retrieval_score: float
    evidence_ids: tuple[str, ...]
    knowledge_node_ids: tuple[str, ...]
    diagnostic_paths: tuple[tuple[str, ...], ...]
    premise_types: tuple[str, ...]
    family_rank: int | None = None
    graph_id: str = ""
    family_representative: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class PatientArgumentGraph:
    graph_id: str
    nodes: tuple[ArgumentNode, ...]
    relations: tuple[ArgumentRelation, ...]
    quality: ArgumentGraphQuality
    patient_evidence_claims: tuple[PatientEvidenceClaim, ...] = ()
    knowledge_claims: tuple[KnowledgeClaim, ...] = ()
    seed_diagnosis: str = ""
    seed_candidate_id: str = ""
    seed_source: str = "flat_rag"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: object) -> str:
    return normalize_label(str(value or ""))

def _evidence_key(value: object) -> str:
    """Return the case-insensitive internal key for an evidence identifier.

    Evidence identifiers are opaque provenance handles.  Production KFA IDs
    contain lower-case hexadecimal suffixes, while some indexes historically
    stored upper-case keys.  Matching is therefore canonicalized internally;
    the original identifier remains unchanged on arguments and in traces.
    """
    return str(value or "").strip().upper()

def _content_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def _source_contains_extract(source_text: str, extract: str) -> bool:
    normalized_source = _text(source_text).casefold()
    normalized_extract = _text(extract).casefold()
    return len(normalized_extract) >= 8 and normalized_extract in normalized_source

def _binding_sha256(
    *,
    patient_evidence_ids: Iterable[str],
    patient_finding: str,
    knowledge_evidence_id: str,
    knowledge_warrant: str,
    knowledge_node_id: str,
    diagnostic_path: Iterable[str],
    premise_type: str,
    knowledge_role_value: KnowledgeRole,
    application: BindingApplication,
) -> str:
    payload = {
        "application": application.value,
        "diagnostic_path": list(diagnostic_path),
        "knowledge_evidence_id": knowledge_evidence_id,
        "knowledge_node_id": knowledge_node_id,
        "knowledge_role": knowledge_role_value.value,
        "knowledge_warrant": knowledge_warrant,
        "patient_evidence_ids": list(patient_evidence_ids),
        "patient_finding": patient_finding,
        "premise_type": premise_type,
    }
    return _content_sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )

def _binding_premise(
    *,
    patient_evidence_id: str,
    patient_finding: str,
    knowledge_evidence_id: str,
    knowledge_warrant: str,
    knowledge_node_id: str,
    diagnostic_path: Iterable[str],
    knowledge_role_value: KnowledgeRole,
    application: BindingApplication,
) -> str:
    path = " -> ".join(diagnostic_path)
    return (
        f'Patient finding [{patient_evidence_id}]: "{patient_finding}". '
        f"Application: {application.value}. "
        f"KG warrant [{knowledge_evidence_id}; node={knowledge_node_id}; "
        f'path={path}; role={knowledge_role_value.value}]: "{knowledge_warrant}".'
    )

def _candidate_label(value: object, case: ClinicalCase) -> str:
    text = _text(value)
    option_key = text.upper().strip(" .():")
    if case.options and option_key in case.options:
        return case.options[option_key]
    for option in case.options.values():
        if label_key(option) == label_key(text):
            return option
    return text

def knowledge_role(premise_type: str) -> KnowledgeRole:
    value = premise_type.casefold()
    if any(term in value for term in ("exclude", "contra", "negative")):
        return KnowledgeRole.COUNTEREVIDENCE
    if any(term in value for term in ("criterion", "criteria", "test", "result")):
        return KnowledgeRole.DIAGNOSTIC_CRITERION
    if any(term in value for term in ("symptom", "sign", "feature", "presentation")):
        return KnowledgeRole.CLINICAL_FEATURE
    if "risk" in value:
        return KnowledgeRole.RISK_FACTOR
    return KnowledgeRole.GUIDELINE

def knowledge_claims(
    facts: Iterable[RetrievedFact],
) -> tuple[KnowledgeClaim, ...]:
    claims: list[KnowledgeClaim] = []
    for fact in facts:
        role = knowledge_role(fact.premise_type)
        claims.append(
            KnowledgeClaim(
                evidence_id=fact.evidence_id,
                node_id=fact.node_id,
                diagnosis_label=fact.diagnosis_label,
                diagnostic_path=fact.diagnostic_path,
                premise_type=fact.premise_type,
                text=fact.text,
                role=role,
                counterevidence_eligible=role in {
                    KnowledgeRole.CLINICAL_FEATURE,
                    KnowledgeRole.DIAGNOSTIC_CRITERION,
                    KnowledgeRole.COUNTEREVIDENCE,
                },
                content_sha256=_content_sha256(fact.text),
            )
        )
    return tuple(claims)

def family_entailment_shortlist(
    exact_patient_findings: Iterable[tuple[str, str]],
    family_child_facts: Iterable[FamilyChildFact],
    *,
    normalizer: UMLSNormalizer | None = None,
    maximum_per_family: int = 3,
) -> tuple[FamilyChildFact, ...]:
    """Select blinded family-local criteria for model entailment review.

    Ranking is within each already-retrieved graph and uses only patient-to-
    criterion compatibility. Child diagnosis labels never participate in the
    rank and are not intended for prompt rendering.
    """
    if maximum_per_family < 1:
        raise ValueError("Entailment shortlist size must be positive.")
    findings = tuple(exact_patient_findings)
    by_graph: dict[str, list[tuple[float, float, str, FamilyChildFact]]] = {}
    for item in family_child_facts:
        fact = item.fact
        if _FAMILY_PREFIX_RE.match(fact.diagnosis_label):
            continue
        role = knowledge_role(fact.premise_type)
        if role != KnowledgeRole.DIAGNOSTIC_CRITERION:
            continue
        if NON_DIAGNOSTIC_PROBABILITY_RE.search(fact.text):
            continue
        best_score = 0.0
        admissible = False
        absence_only = False
        for _, finding in findings:
            assessment = assess_typed_binding(
                finding,
                fact.text,
                role=role.value,
                normalizer=normalizer,
            )
            absence_only = (
                absence_only
                or assessment.criterion_atom.polarity == Polarity.ABSENT
            )
            best_score = max(best_score, assessment.score)
            admissible = admissible or assessment.admissible
        if absence_only:
            continue
        by_graph.setdefault(item.graph_id, []).append((
            1.0 if admissible else 0.0,
            best_score,
            fact.evidence_id,
            item,
        ))
    output: list[FamilyChildFact] = []
    for graph_id in dict.fromkeys(item.graph_id for item in family_child_facts):
        ranked = sorted(
            by_graph.get(graph_id, ()),
            key=lambda row: (-row[0], -row[1], row[2]),
        )
        selected: list[tuple[float, float, str, FamilyChildFact]] = []
        selected_evidence_ids: set[str] = set()
        selected_node_ids: set[str] = set()
        # First preserve graph-node diversity, then fill remaining capacity by
        # relevance.  This prevents several near-duplicate clauses from one
        # node from hiding a different test, measurement, or subtype branch.
        for row in ranked:
            node_id = row[3].fact.node_id
            if node_id in selected_node_ids:
                continue
            selected.append(row)
            selected_evidence_ids.add(row[3].fact.evidence_id)
            selected_node_ids.add(node_id)
            if len(selected) >= maximum_per_family:
                break
        for row in ranked:
            if len(selected) >= maximum_per_family:
                break
            if row[3].fact.evidence_id in selected_evidence_ids:
                continue
            selected.append(row)
            selected_evidence_ids.add(row[3].fact.evidence_id)
        output.extend(row[3] for row in selected)
    return tuple(output)

_COUNTER_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[;:\r\n]+")

def _candidate_identity(
    diagnosis: str,
    normalizer: UMLSNormalizer | None,
) -> tuple[str, tuple[str, ...]]:
    canonical = (
        normalizer.diagnosis_key(diagnosis)
        if normalizer is not None
        else lexical_diagnosis_key(diagnosis)
    )
    return canonical, tuple(sorted(diagnosis_modifiers(diagnosis)))

def _candidate_canonical_key(
    diagnosis: str,
    normalizer: UMLSNormalizer | None,
) -> str:
    canonical, modifiers = _candidate_identity(diagnosis, normalizer)
    return f"{canonical}|modifiers={','.join(modifiers)}"

def candidate_inventory_entries(
    case: ClinicalCase,
    facts: Iterable[RetrievedFact],
    seed_diagnosis: str,
    normalizer: UMLSNormalizer | None = None,
    *,
    focal_only: bool = False,
    family_routes: Iterable[RetrievedFamilyRoute] = (),
) -> tuple[CandidateInventoryEntry, ...]:
    fact_list = tuple(facts)
    family_index = {
        _candidate_identity(alternative.diagnosis_label, normalizer): (
            route.family_rank,
            route.graph_id,
            alternative.representative,
        )
        for route in family_routes
        for alternative in route.alternatives
    }
    records: list[dict[str, Any]] = []
    positions: dict[tuple[str, tuple[str, ...]], int] = {}

    def fact_compatibility(
        diagnosis: str,
        fact: RetrievedFact,
    ) -> float:
        if focal_only:
            return float(
                _candidate_identity(diagnosis, normalizer)
                == _candidate_identity(fact.diagnosis_label, normalizer)
            )
        return diagnosis_path_compatibility(
            diagnosis,
            fact.diagnostic_path,
            normalizer,
        )

    def aligned_facts(
        diagnosis: str,
    ) -> tuple[tuple[int, RetrievedFact, float], ...]:
        aligned = []
        for rank, fact in enumerate(fact_list, 1):
            compatibility = fact_compatibility(diagnosis, fact)
            if compatibility > 0.0:
                aligned.append((rank, fact, compatibility))
        return tuple(aligned)

    def merge(
        position: int,
        routes: tuple[tuple[int, RetrievedFact, float], ...],
        source: str,
    ) -> None:
        record = records[position]
        record["sources"] = tuple(
            dict.fromkeys((*record["sources"], source))
        )
        if not routes:
            return
        route_ranks = tuple(rank for rank, _, _ in routes)
        current_rank = record["retrieval_rank"]
        record["retrieval_rank"] = min(
            route_ranks
            if current_rank is None
            else (*route_ranks, current_rank)
        )
        record["retrieval_score"] = max(
            record["retrieval_score"],
            *(fact.score * compatibility for _, fact, compatibility in routes),
        )
        record["evidence_ids"] = tuple(dict.fromkeys((
            *record["evidence_ids"],
            *(fact.evidence_id for _, fact, _ in routes),
        )))
        record["knowledge_node_ids"] = tuple(dict.fromkeys((
            *record["knowledge_node_ids"],
            *(fact.node_id for _, fact, _ in routes),
        )))
        record["diagnostic_paths"] = tuple(dict.fromkeys((
            *record["diagnostic_paths"],
            *(fact.diagnostic_path for _, fact, _ in routes),
        )))
        record["premise_types"] = tuple(dict.fromkeys((
            *record["premise_types"],
            *(fact.premise_type for _, fact, _ in routes),
        )))

    def add(value: str, source: str) -> None:
        diagnosis = _candidate_label(value, case)
        if not diagnosis:
            return
        identity = _candidate_identity(diagnosis, normalizer)
        routes = aligned_facts(diagnosis) if source == "kg_route" else ()
        position = positions.get(identity)
        if position is not None:
            merge(position, routes, source)
            return
        if len(records) == MAX_CANDIDATES:
            return
        positions[identity] = len(records)
        records.append({
            "diagnosis": diagnosis,
            "canonical_key": _candidate_canonical_key(diagnosis, normalizer),
            "sources": (source,),
            "retrieval_rank": min(
                (rank for rank, _, _ in routes),
                default=None,
            ),
            "retrieval_score": max(
                (
                    fact.score * compatibility
                    for _, fact, compatibility in routes
                ),
                default=0.0,
            ),
            "evidence_ids": tuple(
                dict.fromkeys(fact.evidence_id for _, fact, _ in routes)
            ),
            "knowledge_node_ids": tuple(
                dict.fromkeys(fact.node_id for _, fact, _ in routes)
            ),
            "diagnostic_paths": tuple(
                dict.fromkeys(fact.diagnostic_path for _, fact, _ in routes)
            ),
            "premise_types": tuple(
                dict.fromkeys(fact.premise_type for _, fact, _ in routes)
            ),
        })

    if case.options:
        option_values = tuple(case.options.values())
        ranked_options = sorted(
            option_values,
            key=lambda option: (
                -max(
                    (
                        fact.score
                        * fact_compatibility(option, fact)
                        for fact in fact_list
                    ),
                    default=0.0,
                ),
                option_values.index(option),
            ),
        )
        for option in ranked_options:
            if aligned_facts(option):
                add(option, "kg_route")
    else:
        for fact in fact_list:
            add(fact.diagnosis_label, "kg_route")
        # Nested family routes already enumerate provenance-bound alternatives.
        # Do not turn shared path ancestors into additional flat candidates.
        if not focal_only and not family_index:
            for fact in fact_list:
                for diagnosis in reversed(fact.diagnostic_path):
                    add(diagnosis, "kg_route")

    seed = _candidate_label(seed_diagnosis, case)
    if seed:
        seed_identity = _candidate_identity(seed, normalizer)
        position = positions.get(seed_identity)
        if position is not None:
            merge(position, (), "flat_seed")
        else:
            if len(records) == MAX_CANDIDATES:
                removed = records.pop()
                positions.pop(
                    _candidate_identity(str(removed["diagnosis"]), normalizer),
                    None,
                )
            add(seed, "flat_seed")

    return tuple(
        CandidateInventoryEntry(
            candidate_id=f"D{index}",
            diagnosis=str(record["diagnosis"]),
            canonical_key=str(record["canonical_key"]),
            sources=tuple(record["sources"]),
            retrieval_rank=record["retrieval_rank"],
            retrieval_score=round(float(record["retrieval_score"]), 12),
            evidence_ids=tuple(record["evidence_ids"]),
            knowledge_node_ids=tuple(record["knowledge_node_ids"]),
            diagnostic_paths=tuple(record["diagnostic_paths"]),
            premise_types=tuple(record["premise_types"]),
            family_rank=family_index.get(
                _candidate_identity(str(record["diagnosis"]), normalizer),
                (None, "", False),
            )[0],
            graph_id=family_index.get(
                _candidate_identity(str(record["diagnosis"]), normalizer),
                (None, "", False),
            )[1],
            family_representative=family_index.get(
                _candidate_identity(str(record["diagnosis"]), normalizer),
                (None, "", False),
            )[2],
        )
        for index, record in enumerate(records, 1)
    )

def kg_candidate_inventory_entries(
    case: ClinicalCase,
    facts: Iterable[RetrievedFact],
    normalizer: UMLSNormalizer | None = None,
    family_routes: Iterable[RetrievedFamilyRoute] = (),
) -> tuple[CandidateInventoryEntry, ...]:
    return candidate_inventory_entries(
        case,
        facts,
        "",
        normalizer,
        family_routes=family_routes,
    )

_FAMILY_PREFIX_RE = re.compile(
    r"^(?:(?:strongly\s+)?suspected|possible|probable)\s+",
    flags=re.IGNORECASE,
)

def _family_label(route: RetrievedFamilyRoute) -> str:
    """Return the controlled common ancestor used for family reasoning."""
    paths = tuple(alternative.diagnostic_path for alternative in route.alternatives)
    common: list[str] = []
    for values in zip(*paths):
        if len({label_key(value) for value in values}) != 1:
            break
        common.append(values[0])
    raw = common[0] if common else route.family_key.removeprefix("graph:")
    label = _FAMILY_PREFIX_RE.sub("", raw).strip()
    return label or route.representative_diagnosis

def family_candidate_inventory_entries(
    case: ClinicalCase,
    facts: Iterable[RetrievedFact],
    normalizer: UMLSNormalizer | None = None,
    family_routes: Iterable[RetrievedFamilyRoute] = (),
) -> tuple[CandidateInventoryEntry, ...]:
    """Expose one argument candidate per selected controlled graph family.

    All currently retrieved warrants from that graph belong to the family
    candidate. Diagnosis children remain in ``family_routes`` and are filtered
    only after a family has verified support.
    """
    fact_list = tuple(facts)
    output: list[CandidateInventoryEntry] = []
    for route in sorted(family_routes, key=lambda item: item.family_rank):
        source_ids = {
            source_id
            for alternative in route.alternatives
            for source_id in alternative.source_chunk_ids
        }
        family_facts = tuple(
            fact for fact in fact_list if fact.source_chunk_id in source_ids
        )
        if not family_facts:
            continue
        diagnosis = _candidate_label(_family_label(route), case)
        output.append(CandidateInventoryEntry(
            candidate_id=f"D{len(output) + 1}",
            diagnosis=diagnosis,
            canonical_key=f"graph:{route.graph_id}|modifiers=",
            sources=("kg_family_route",),
            retrieval_rank=route.family_rank,
            retrieval_score=max(fact.score for fact in family_facts),
            evidence_ids=tuple(dict.fromkeys(
                fact.evidence_id for fact in family_facts
            )),
            knowledge_node_ids=tuple(dict.fromkeys(
                fact.node_id for fact in family_facts
            )),
            diagnostic_paths=tuple(dict.fromkeys(
                fact.diagnostic_path for fact in family_facts
            )),
            premise_types=tuple(dict.fromkeys(
                fact.premise_type for fact in family_facts
            )),
            family_rank=route.family_rank,
            graph_id=route.graph_id,
            family_representative=True,
        ))
        if len(output) == 8:
            break
    return tuple(output)

def _best_typed_source_span(
    patient_source: str,
    claim: KnowledgeClaim,
    normalizer: UMLSNormalizer | None,
) -> tuple[str, Any] | None:
    spans = tuple(dict.fromkeys(
        " ".join(span.split())
        for span in _COUNTER_SENTENCE_RE.split(patient_source)
        if len(" ".join(span.split())) >= 8
    ))
    best: tuple[float, str, Any] | None = None
    for span in spans:
        assessment = assess_typed_binding(
            span,
            claim.text,
            role=claim.role.value,
            normalizer=normalizer,
        )
        if not assessment.admissible:
            continue
        record = (assessment.score, span, assessment)
        if best is None or record[0] > best[0]:
            best = record
    return (best[1], best[2]) if best is not None else None

def ground_reasoner_proposal(
    proposal: ReasonerProposal,
    candidates: Iterable[str | CandidateInventoryEntry],
    knowledge_roles: Mapping[str, KnowledgeRole] | None = None,
    patient_evidence_text: Mapping[str, str] | None = None,
    knowledge_inventory: Iterable[KnowledgeClaim] = (),
    *,
    canonicalize_positive_applications: bool = False,
    allow_model_entailment: bool = False,
    normalizer: UMLSNormalizer | None = None,
) -> ReasonerProposal:
    candidate_values = tuple(candidates)
    knowledge_claim_index = {
        _evidence_key(claim.evidence_id): claim for claim in knowledge_inventory
    }
    normalized_patient_evidence = (
        {
            _evidence_key(evidence_id): str(text)
            for evidence_id, text in patient_evidence_text.items()
        }
        if patient_evidence_text is not None
        else None
    )
    normalized_knowledge_roles = (
        {
            _evidence_key(evidence_id): role
            for evidence_id, role in knowledge_roles.items()
        }
        if knowledge_roles is not None
        else None
    )
    enforce_typed_bindings = (
        normalized_patient_evidence is not None or bool(knowledge_claim_index)
    )
    if enforce_typed_bindings and (
        normalized_patient_evidence is None or not knowledge_claim_index
    ):
        raise ValueError(
            "Typed binding validation requires patient and KG source text."
        )
    typed_inventory = tuple(
        candidate
        for candidate in candidate_values
        if isinstance(candidate, CandidateInventoryEntry)
    )
    inventory = tuple(dict.fromkeys(
        (
            candidate.diagnosis
            if isinstance(candidate, CandidateInventoryEntry)
            else candidate
        )
        for candidate in candidate_values
        if (
            candidate.diagnosis.strip()
            if isinstance(candidate, CandidateInventoryEntry)
            else candidate.strip()
        )
    ))[:MAX_CANDIDATES]
    candidate_knowledge_ids = {
        label_key(candidate.diagnosis): {
            _evidence_key(evidence_id) for evidence_id in candidate.evidence_ids
        }
        for candidate in typed_inventory
    }
    enforce_candidate_routes = len(typed_inventory) == len(candidate_values)
    proposed_by_diagnosis = {
        label_key(candidate.diagnosis): candidate
        for candidate in proposal.candidates
    }
    grounded_candidates: list[DiagnosisCandidate] = []
    next_argument_id = 1
    canonicalized_argument_count = proposal.canonicalized_argument_count
    invalid_grounding_count = 0
    for diagnosis in inventory:
        proposed = proposed_by_diagnosis.get(label_key(diagnosis))
        arguments: list[ProposedArgument] = []
        seen_arguments: set[
            tuple[ArgumentScheme, tuple[str, ...]]
        ] = set()
        for argument in proposed.arguments if proposed is not None else ():
            patient_ids = tuple(
                evidence_id
                for evidence_id in argument.evidence_ids
                if evidence_id.startswith("S-")
            )
            knowledge_ids = tuple(
                evidence_id
                for evidence_id in argument.evidence_ids
                if evidence_id.startswith("K")
            )
            if enforce_candidate_routes and (
                not patient_ids
                or len(knowledge_ids) != 1
                or _evidence_key(knowledge_ids[0])
                not in candidate_knowledge_ids.get(label_key(diagnosis), set())
            ):
                invalid_grounding_count += 1
                continue
            claim = (
                knowledge_claim_index.get(_evidence_key(knowledge_ids[0]))
                if len(knowledge_ids) == 1
                else None
            )
            application = argument.application
            if (
                canonicalize_positive_applications
                and claim is not None
                and application is not None
                and application != BindingApplication.CONFLICTS
            ):
                if claim.role == KnowledgeRole.RISK_FACTOR:
                    canonical_application = (
                        BindingApplication.RAISES_PLAUSIBILITY
                    )
                elif application == BindingApplication.RAISES_PLAUSIBILITY:
                    canonical_application = (
                        BindingApplication.PARTIALLY_SATISFIES
                    )
                else:
                    canonical_application = application
                if canonical_application != application:
                    canonicalized_argument_count += 1
                application = canonical_application
            patient_source = (
                normalized_patient_evidence.get(
                    _evidence_key(patient_ids[0]), ""
                )
                if normalized_patient_evidence is not None
                and len(patient_ids) == 1
                else ""
            )
            patient_finding = argument.patient_finding
            typed_assessment = None
            if claim is not None and canonicalize_positive_applications:
                derived = _best_typed_source_span(
                    patient_source,
                    claim,
                    normalizer,
                )
                if derived is not None:
                    patient_finding, typed_assessment = derived
                    canonicalized_argument_count += int(
                        patient_finding != argument.patient_finding
                    )
                elif patient_finding:
                    typed_assessment = assess_typed_binding(
                        patient_finding,
                        claim.text,
                        role=claim.role.value,
                        normalizer=normalizer,
                    )
            elif claim is not None and patient_finding:
                typed_assessment = assess_typed_binding(
                    patient_finding,
                    claim.text,
                    role=claim.role.value,
                    normalizer=normalizer,
                )
            model_entailment_valid = bool(
                allow_model_entailment
                and claim is not None
                and typed_assessment is not None
                and _llm_entailment_safety_passes(typed_assessment, claim)
            )
            if enforce_typed_bindings and (
                len(patient_ids) != 1
                or claim is None
                or not patient_finding
                or not _source_contains_extract(
                    patient_source,
                    patient_finding,
                )
                or typed_assessment is None
                or not (
                    typed_assessment.admissible or model_entailment_valid
                )
                or application is None
                or application
                not in POSITIVE_BINDING_APPLICATIONS.get(
                    claim.role,
                    frozenset(),
                )
            ):
                invalid_grounding_count += 1
                continue
            scheme = argument.scheme
            scheme_source = argument.scheme_source
            if normalized_knowledge_roles is not None:
                cited_roles = {
                    normalized_knowledge_roles.get(_evidence_key(evidence_id))
                    for evidence_id in knowledge_ids
                }
                if len(cited_roles) != 1 or None in cited_roles:
                    invalid_grounding_count += 1
                    continue
                role = next(iter(cited_roles))
                derived_scheme = KNOWLEDGE_ROLE_SUPPORT_SCHEMES.get(role)
                if derived_scheme is None:
                    invalid_grounding_count += 1
                    continue
                if (
                    scheme != derived_scheme
                    or scheme_source != SCHEME_TYPING_ID
                ):
                    canonicalized_argument_count += 1
                scheme = derived_scheme
                scheme_source = SCHEME_TYPING_ID
            if enforce_typed_bindings:
                assert claim is not None
                assert application is not None
                knowledge_warrant = claim.text
                knowledge_node_id = claim.node_id
                diagnostic_path = claim.diagnostic_path
                premise_type = claim.premise_type
                role = claim.role
                patient_source_sha256 = _content_sha256(patient_source)
                knowledge_warrant_sha256 = _content_sha256(
                    knowledge_warrant
                )
                binding_content_sha256 = _binding_sha256(
                    patient_evidence_ids=patient_ids,
                    patient_finding=patient_finding,
                    knowledge_evidence_id=knowledge_ids[0],
                    knowledge_warrant=knowledge_warrant,
                    knowledge_node_id=knowledge_node_id,
                    diagnostic_path=diagnostic_path,
                    premise_type=premise_type,
                    knowledge_role_value=role,
                    application=application,
                )
                premise = _binding_premise(
                    patient_evidence_id=patient_ids[0],
                    patient_finding=patient_finding,
                    knowledge_evidence_id=knowledge_ids[0],
                    knowledge_warrant=knowledge_warrant,
                    knowledge_node_id=knowledge_node_id,
                    diagnostic_path=diagnostic_path,
                    knowledge_role_value=role,
                    application=application,
                )
                binding_validation_id = (
                    BINDING_VALIDATION_ID
                    if typed_assessment.admissible
                    else LLM_ENTAILMENT_BINDING_ID
                )
                binding_critical_questions = tuple(
                    {
                        "question_id": item.question_id,
                        "passed": item.passed,
                        "detail": item.detail,
                    }
                    for item in typed_assessment.critical_questions
                ) + (
                    ({
                        "question_id": "clinical_entailment_judged",
                        "passed": True,
                        "detail": "extractive generator judgment; verifier required",
                    },)
                    if model_entailment_valid and not typed_assessment.admissible
                    else ()
                )
            else:
                knowledge_warrant = argument.knowledge_warrant
                knowledge_node_id = argument.knowledge_node_id
                diagnostic_path = argument.diagnostic_path
                premise_type = argument.premise_type
                role = argument.knowledge_role
                patient_source_sha256 = argument.patient_source_sha256
                knowledge_warrant_sha256 = (
                    argument.knowledge_warrant_sha256
                )
                binding_content_sha256 = argument.binding_content_sha256
                premise = argument.premise
                binding_validation_id = argument.binding_validation_id
                binding_critical_questions = argument.binding_critical_questions
            argument_key = (
                scheme,
                tuple(sorted(argument.evidence_ids)),
            )
            if argument_key in seen_arguments:
                canonicalized_argument_count += 1
                continue
            seen_arguments.add(argument_key)
            arguments.append(
                ProposedArgument(
                    argument_id=f"A{next_argument_id}",
                    scheme=scheme,
                    premise=premise,
                    conclusion=diagnosis,
                    evidence_ids=argument.evidence_ids,
                    scheme_source=scheme_source,
                    patient_finding=patient_finding,
                    application=application,
                    model_rationale=argument.model_rationale,
                    knowledge_warrant=knowledge_warrant,
                    knowledge_node_id=knowledge_node_id,
                    diagnostic_path=diagnostic_path,
                    premise_type=premise_type,
                    knowledge_role=role,
                    patient_source_sha256=patient_source_sha256,
                    knowledge_warrant_sha256=knowledge_warrant_sha256,
                    binding_content_sha256=binding_content_sha256,
                    binding_validation_id=binding_validation_id,
                    binding_critical_questions=binding_critical_questions,
                )
            )
            next_argument_id += 1
        grounded_candidates.append(
            DiagnosisCandidate(
                candidate_id=f"D{len(grounded_candidates) + 1}",
                diagnosis=diagnosis,
                arguments=tuple(arguments),
            )
        )

    preferred = next(
        (
            diagnosis
            for diagnosis in inventory
            if label_key(diagnosis) == label_key(proposal.preferred_diagnosis)
        ),
        "",
    )
    inventory_keys = {label_key(diagnosis) for diagnosis in inventory}
    unrouted_argument_count = sum(
        len(candidate.arguments)
        for candidate in proposal.candidates
        if label_key(candidate.diagnosis) not in inventory_keys
    )
    return ReasonerProposal(
        candidates=tuple(grounded_candidates),
        preferred_diagnosis=preferred,
        abstain=proposal.abstain or not preferred,
        raw_argument_count=proposal.raw_argument_count,
        invalid_argument_count=(
            proposal.invalid_argument_count
            + unrouted_argument_count
            + invalid_grounding_count
        ),
        canonicalized_argument_count=canonicalized_argument_count,
    )

def _argument_evidence_compatibility(
    argument: ProposedArgument,
    valid_ids: set[str],
    knowledge_support: Mapping[str, tuple[str, ...]],
    normalizer: UMLSNormalizer | None,
    knowledge_roles: Mapping[str, KnowledgeRole] | None = None,
) -> float:
    valid_keys = {_evidence_key(value) for value in valid_ids}
    support_by_key = {
        _evidence_key(evidence_id): tuple(path)
        for evidence_id, path in knowledge_support.items()
    }
    roles_by_key = (
        {
            _evidence_key(evidence_id): role
            for evidence_id, role in knowledge_roles.items()
        }
        if knowledge_roles
        else {}
    )
    evidence_ids = {_evidence_key(value) for value in argument.evidence_ids}
    if not evidence_ids or not evidence_ids.issubset(valid_keys):
        return 0.0
    has_patient_evidence = any(value.startswith("S-") for value in evidence_ids)
    knowledge_evidence = sorted({
        value for value in evidence_ids if value.startswith("K")
    })
    compatibility_scores = []
    for evidence_id in knowledge_evidence:
        path = support_by_key.get(evidence_id, ())
        compatibility = diagnosis_path_compatibility(
            argument.conclusion,
            path,
            normalizer,
        )
        if compatibility == 0.0:
            conclusion_key = label_key(argument.conclusion)
            matching_positions = [
                index
                for index, value in enumerate(path)
                if label_key(_FAMILY_PREFIX_RE.sub("", str(value)).strip())
                == conclusion_key
            ]
            if matching_positions:
                distance = len(path) - 1 - max(matching_positions)
                compatibility = (
                    1.0 if distance == 0
                    else 0.75 if distance == 1
                    else 0.5
                )
        compatibility_scores.append(compatibility)
    if not compatibility_scores:
        return 0.0
    if roles_by_key:
        permitted_roles = SUPPORT_SCHEME_KNOWLEDGE_ROLES.get(
            argument.scheme,
            frozenset(),
        )
        cited_roles = [
            roles_by_key.get(evidence_id)
            for evidence_id in knowledge_evidence
        ]
        if (
            not permitted_roles
            or any(role not in permitted_roles for role in cited_roles)
        ):
            return 0.0
    if (
        argument.scheme != ArgumentScheme.GUIDELINE_AUTHORITY
        and not has_patient_evidence
    ):
        return 0.0
    return min(compatibility_scores)

def build_patient_argument_graph(
    *,
    case_id: str,
    proposal: ReasonerProposal,
    verifier: VerifierReport,
    valid_evidence_ids: Iterable[str],
    knowledge_support: Mapping[str, Iterable[str]],
    knowledge_provenance: Mapping[str, Mapping[str, Any]] | None = None,
    knowledge_inventory: Iterable[KnowledgeClaim] = (),
    seed_diagnosis: str = "",
    seed_source: str = "flat_rag",
    patient_evidence_text: Mapping[str, str] | None = None,
    patient_evidence_inventory: Iterable[PatientEvidenceClaim] = (),
    normalizer: UMLSNormalizer | None = None,
) -> PatientArgumentGraph:
    valid_ids = {_evidence_key(value) for value in valid_evidence_ids}
    normalized_knowledge_support = {
        _evidence_key(evidence_id): tuple(
            str(label) for label in labels if str(label).strip()
        )
        for evidence_id, labels in knowledge_support.items()
    }
    normalized_provenance = {
        _evidence_key(evidence_id): dict(value)
        for evidence_id, value in (knowledge_provenance or {}).items()
    }
    normalized_knowledge_roles = {
        evidence_id: knowledge_role(str(record.get("premise_type") or ""))
        for evidence_id, record in normalized_provenance.items()
    }
    normalized_patient_evidence = {
        _evidence_key(evidence_id): str(text)
        for evidence_id, text in (patient_evidence_text or {}).items()
    }
    patient_claims = tuple(patient_evidence_inventory)
    if not patient_claims and normalized_patient_evidence:
        patient_claims = tuple(
            PatientEvidenceClaim(
                evidence_id=evidence_id,
                section_name="",
                text=text,
                content_sha256=_content_sha256(text),
            )
            for evidence_id, text in normalized_patient_evidence.items()
        )
    binding_contract_enabled = bool(
        normalized_patient_evidence and normalized_provenance
    )
    seed_candidate_id = next(
        (
            candidate.candidate_id
            for candidate in proposal.candidates
            if seed_diagnosis
            and _candidate_identity(candidate.diagnosis, normalizer)
            == _candidate_identity(seed_diagnosis, normalizer)
        ),
        "",
    )

    def provenance(evidence_ids: Iterable[str]) -> tuple[
        tuple[str, ...], tuple[tuple[str, ...], ...], tuple[str, ...]
    ]:
        records = [
            normalized_provenance[_evidence_key(evidence_id)]
            for evidence_id in evidence_ids
            if _evidence_key(evidence_id) in normalized_provenance
        ]
        node_ids = tuple(dict.fromkeys(
            str(record.get("node_id") or "") for record in records
            if record.get("node_id")
        ))
        paths = tuple(dict.fromkeys(
            tuple(str(item) for item in record.get("diagnostic_path") or ())
            for record in records
            if record.get("diagnostic_path")
        ))
        premise_types = tuple(dict.fromkeys(
            str(record.get("premise_type") or "") for record in records
            if record.get("premise_type")
        ))
        return node_ids, paths, premise_types
    nodes: list[ArgumentNode] = []
    relations: list[ArgumentRelation] = []
    node_ids: set[str] = set()

    def add_node(node: ArgumentNode) -> None:
        if node.argument_id in node_ids:
            raise ValueError(f"Duplicate argument node id: {node.argument_id}")
        node_ids.add(node.argument_id)
        nodes.append(node)

    reviews = {review.argument_id: review for review in verifier.reviews}
    proposed_arguments = [
        argument
        for candidate in proposal.candidates
        for argument in candidate.arguments
    ]
    valid_argument_count = 0
    valid_typed_binding_count = 0
    scheme_valid_argument_count = 0
    argument_compatibility_total = 0.0
    supported_review_count = 0
    grounded_supported_review_count = 0
    reference_count = sum(len(argument.evidence_ids) for argument in proposed_arguments)
    valid_reference_count = sum(
        _evidence_key(evidence_id) in valid_ids
        for argument in proposed_arguments
        for evidence_id in argument.evidence_ids
    )

    for candidate in proposal.candidates:
        for argument in candidate.arguments:
            kg_node_ids, kg_paths, kg_premise_types = provenance(
                argument.evidence_ids
            )
            evidence_compatibility = _argument_evidence_compatibility(
                argument,
                valid_ids,
                normalized_knowledge_support,
                normalizer,
                normalized_knowledge_roles,
            )
            patient_ids = tuple(
                evidence_id
                for evidence_id in argument.evidence_ids
                if evidence_id.startswith("S-")
            )
            knowledge_ids = tuple(
                evidence_id
                for evidence_id in argument.evidence_ids
                if evidence_id.startswith("K")
            )
            patient_source = (
                normalized_patient_evidence.get(
                    _evidence_key(patient_ids[0]), ""
                )
                if len(patient_ids) == 1
                else ""
            )
            knowledge_record = (
                normalized_provenance.get(_evidence_key(knowledge_ids[0]))
                if len(knowledge_ids) == 1
                else None
            )
            expected_role = (
                normalized_knowledge_roles.get(
                    _evidence_key(knowledge_ids[0])
                )
                if len(knowledge_ids) == 1
                else None
            )
            expected_path = tuple(
                str(value)
                for value in (
                    knowledge_record.get("diagnostic_path")
                    if knowledge_record is not None
                    else ()
                )
            )
            expected_warrant = str(
                knowledge_record.get("text")
                if knowledge_record is not None
                else ""
            )
            expected_node_id = str(
                knowledge_record.get("node_id")
                if knowledge_record is not None
                else ""
            )
            expected_premise_type = str(
                knowledge_record.get("premise_type")
                if knowledge_record is not None
                else ""
            )
            expected_binding_hash = (
                _binding_sha256(
                    patient_evidence_ids=patient_ids,
                    patient_finding=argument.patient_finding,
                    knowledge_evidence_id=knowledge_ids[0],
                    knowledge_warrant=expected_warrant,
                    knowledge_node_id=expected_node_id,
                    diagnostic_path=expected_path,
                    premise_type=expected_premise_type,
                    knowledge_role_value=expected_role,
                    application=argument.application,
                )
                if (
                    len(patient_ids) == 1
                    and len(knowledge_ids) == 1
                    and expected_role is not None
                    and argument.application is not None
                )
                else ""
            )
            expected_premise = (
                _binding_premise(
                    patient_evidence_id=patient_ids[0],
                    patient_finding=argument.patient_finding,
                    knowledge_evidence_id=knowledge_ids[0],
                    knowledge_warrant=expected_warrant,
                    knowledge_node_id=expected_node_id,
                    diagnostic_path=expected_path,
                    knowledge_role_value=expected_role,
                    application=argument.application,
                )
                if (
                    len(patient_ids) == 1
                    and len(knowledge_ids) == 1
                    and expected_role is not None
                    and argument.application is not None
                )
                else ""
            )
            expected_assessment = (
                assess_typed_binding(
                    argument.patient_finding,
                    expected_warrant,
                    role=expected_role.value,
                    normalizer=normalizer,
                )
                if expected_role is not None and argument.patient_finding
                else None
            )
            expected_critical_questions = (
                tuple({
                    "question_id": item.question_id,
                    "passed": item.passed,
                    "detail": item.detail,
                } for item in expected_assessment.critical_questions)
                if expected_assessment is not None
                else ()
            )
            provenance_binding_valid = (
                len(patient_ids) == 1
                and len(knowledge_ids) == 1
                and knowledge_record is not None
                and expected_role is not None
                and _source_contains_extract(
                    patient_source,
                    argument.patient_finding,
                )
                and argument.application
                in POSITIVE_BINDING_APPLICATIONS.get(
                    expected_role,
                    frozenset(),
                )
                and argument.knowledge_warrant == expected_warrant
                and argument.knowledge_node_id == expected_node_id
                and argument.diagnostic_path == expected_path
                and argument.premise_type == expected_premise_type
                and argument.knowledge_role == expected_role
                and argument.patient_source_sha256
                == _content_sha256(patient_source)
                and argument.knowledge_warrant_sha256
                == _content_sha256(expected_warrant)
                and argument.binding_content_sha256 == expected_binding_hash
                and argument.premise == expected_premise
            )
            typed_binding_valid = bool(
                provenance_binding_valid
                and expected_assessment is not None
                and (
                    (
                        argument.binding_validation_id == BINDING_VALIDATION_ID
                        and expected_assessment.admissible
                        and argument.binding_critical_questions
                        == expected_critical_questions
                    )
                    or (
                        argument.binding_validation_id
                        == LLM_ENTAILMENT_BINDING_ID
                        and _llm_entailment_questions_valid(
                            argument.binding_critical_questions
                        )
                    )
                )
            )
            valid_typed_binding_count += int(typed_binding_valid)
            if binding_contract_enabled and not typed_binding_valid:
                evidence_compatibility = 0.0
            evidence_valid = evidence_compatibility > 0.0
            valid_argument_count += int(evidence_valid)
            cited_knowledge_ids = tuple(
                evidence_id
                for evidence_id in argument.evidence_ids
                if evidence_id.startswith("K")
            )
            permitted_roles = SUPPORT_SCHEME_KNOWLEDGE_ROLES.get(
                argument.scheme,
                frozenset(),
            )
            scheme_valid = (
                not normalized_knowledge_roles
                or (
                    bool(cited_knowledge_ids)
                    and all(
                        normalized_knowledge_roles.get(
                            _evidence_key(evidence_id)
                        )
                        in permitted_roles
                        for evidence_id in cited_knowledge_ids
                    )
                )
            )
            scheme_valid_argument_count += int(scheme_valid)
            argument_compatibility_total += evidence_compatibility
            add_node(
                ArgumentNode(
                    argument_id=argument.argument_id,
                    node_type="support",
                    scheme=argument.scheme,
                    premise=argument.premise,
                    conclusion=argument.conclusion,
                    evidence_ids=argument.evidence_ids,
                    candidate_id=candidate.candidate_id,
                    source="reasoner_agent",
                    priority=SCHEME_PRIORITY[argument.scheme],
                    evidence_compatibility=evidence_compatibility,
                    knowledge_node_ids=kg_node_ids,
                    diagnostic_paths=kg_paths,
                    premise_types=kg_premise_types,
                    scheme_source=argument.scheme_source,
                    patient_finding=argument.patient_finding,
                    application=argument.application,
                    model_rationale=argument.model_rationale,
                    knowledge_warrant=argument.knowledge_warrant,
                    knowledge_role=argument.knowledge_role,
                    patient_source_sha256=argument.patient_source_sha256,
                    knowledge_warrant_sha256=(
                        argument.knowledge_warrant_sha256
                    ),
                    binding_content_sha256=argument.binding_content_sha256,
                    binding_validation_id=argument.binding_validation_id,
                    binding_critical_questions=(
                        argument.binding_critical_questions
                    ),
                )
            )

            review = reviews.get(argument.argument_id)
            challenge_reasons: list[str] = []
            challenge_evidence: tuple[str, ...] = ()
            if not evidence_valid:
                challenge_reasons.append(
                    "The argument does not contain a validated patient-to-KG "
                    "binding with diagnosis-aligned evidence."
                )
            if review is None:
                challenge_reasons.append("The verifier did not review this argument.")
            else:
                challenge_evidence = review.evidence_ids
                if review.verdict == ReviewVerdict.SUPPORTED:
                    supported_review_count += 1
                    review_ids = set(review.evidence_ids)
                    argument_patient_ids = {
                        evidence_id
                        for evidence_id in argument.evidence_ids
                        if evidence_id.startswith("S-")
                    }
                    argument_knowledge_ids = {
                        evidence_id
                        for evidence_id in argument.evidence_ids
                        if evidence_id.startswith("K")
                    }
                    review_grounded = (
                        review_ids == set(argument.evidence_ids)
                        and review_ids.issubset(valid_ids)
                        and len(argument_patient_ids) == 1
                        and len(argument_knowledge_ids) == 1
                        and not review.failed_critical_questions
                        and (
                            not binding_contract_enabled
                            or typed_binding_valid
                        )
                    )
                    grounded_supported_review_count += int(review_grounded)
                    if not review_grounded:
                        challenge_reasons.append(
                            "The supported review does not cite both the "
                            "argument's patient evidence and KG warrant."
                        )
                if review.verdict != ReviewVerdict.SUPPORTED:
                    challenge_reasons.append(
                        f"The verifier labelled the argument {review.verdict.value}."
                    )
                if review.failed_critical_questions:
                    challenge_reasons.append(
                        "Failed critical questions: "
                        + ", ".join(review.failed_critical_questions)
                        + "."
                    )
            if challenge_reasons:
                challenge_id = f"Q-{argument.argument_id}"
                add_node(
                    ArgumentNode(
                        argument_id=challenge_id,
                        node_type="challenge",
                        scheme=ArgumentScheme.CRITICAL_QUESTION,
                        premise=" ".join(challenge_reasons),
                        conclusion=f"{argument.argument_id} is undercut.",
                        evidence_ids=challenge_evidence,
                        candidate_id=candidate.candidate_id,
                        source="evidence_validator",
                        priority=SCHEME_PRIORITY[ArgumentScheme.CRITICAL_QUESTION],
                    )
                )
                relations.append(
                    ArgumentRelation(
                        source_id=challenge_id,
                        target_id=argument.argument_id,
                        relation=RelationType.UNDERCUTS,
                    )
                )

        candidate_evidence = tuple(
            dict.fromkeys(
                evidence_id
                for argument in candidate.arguments
                for evidence_id in argument.evidence_ids
            )
        )
        route_evidence = tuple(
            evidence_id
            for evidence_id, path in normalized_knowledge_support.items()
            if diagnosis_path_compatibility(
                candidate.diagnosis,
                path,
                normalizer,
            ) > 0.0
        )
        kg_node_ids, kg_paths, kg_premise_types = provenance(
            (*candidate_evidence, *route_evidence)
        )
        add_node(
            ArgumentNode(
                argument_id=candidate.candidate_id,
                node_type="diagnosis",
                scheme=ArgumentScheme.BEST_EXPLANATION,
                premise=(
                    f"The verified observations and guideline warrants are jointly "
                    f"explained by {candidate.diagnosis}."
                ),
                conclusion=candidate.diagnosis,
                evidence_ids=candidate_evidence,
                candidate_id=candidate.candidate_id,
                source=(
                    (
                        "flat_rag_seed_and_knowledge_graph_route"
                        if kg_node_ids
                        else "flat_rag_seed"
                    )
                    if candidate.candidate_id == seed_candidate_id
                    else "knowledge_graph_route"
                ),
                priority=SCHEME_PRIORITY[ArgumentScheme.BEST_EXPLANATION],
                knowledge_node_ids=kg_node_ids,
                diagnostic_paths=kg_paths,
                premise_types=kg_premise_types,
            )
        )
        relations.extend(
            ArgumentRelation(
                source_id=argument.argument_id,
                target_id=candidate.candidate_id,
                relation=RelationType.SUPPORTS,
            )
            for argument in candidate.arguments
        )
        if not candidate.arguments:
            challenge_id = f"Q-{candidate.candidate_id}"
            add_node(
                ArgumentNode(
                    argument_id=challenge_id,
                    node_type="challenge",
                    scheme=ArgumentScheme.CRITICAL_QUESTION,
                    premise="The candidate has no structurally valid supporting arguments.",
                    conclusion=f"{candidate.candidate_id} lacks support.",
                    evidence_ids=(),
                    candidate_id=candidate.candidate_id,
                    source="evidence_validator",
                    priority=SCHEME_PRIORITY[ArgumentScheme.CRITICAL_QUESTION],
                )
            )
            relations.append(
                ArgumentRelation(
                    source_id=challenge_id,
                    target_id=candidate.candidate_id,
                    relation=RelationType.UNDERCUTS,
                )
            )

    valid_counterargument_count = 0
    for counterargument in verifier.counterarguments:
        target = next(
            (node for node in nodes if node.argument_id == counterargument.target_argument_id),
            None,
        )
        candidate_id = target.candidate_id if target is not None else None
        candidate = next(
            (
                item for item in proposal.candidates
                if item.candidate_id == candidate_id
            ),
            None,
        )
        kg_node_ids, kg_paths, kg_premise_types = provenance(
            counterargument.evidence_ids
        )
        counter_ids = {
            _evidence_key(evidence_id)
            for evidence_id in counterargument.evidence_ids
        }
        knowledge_ids = tuple(sorted(
            evidence_id for evidence_id in counter_ids
            if evidence_id.startswith("K")
        ))
        target_compatibilities = tuple(
            diagnosis_path_compatibility(
                candidate.diagnosis if candidate is not None else "",
                normalized_knowledge_support.get(evidence_id, ()),
                normalizer,
            )
            for evidence_id in knowledge_ids
        )
        if counterargument.scheme == ArgumentScheme.ALTERNATIVE_EXPLANATION:
            alternative_compatibilities = tuple(
                diagnosis_path_compatibility(
                    counterargument.conclusion,
                    normalized_knowledge_support.get(evidence_id, ()),
                    normalizer,
                )
                for evidence_id in knowledge_ids
            )
            counter_compatibility = min(
                alternative_compatibilities,
                default=0.0,
            )
            route_is_valid = (
                counterargument.relation == RelationType.REBUTS
                and counter_compatibility > 0.0
                and max(target_compatibilities, default=0.0) == 0.0
            )
        else:
            counter_compatibility = min(
                target_compatibilities,
                default=0.0,
            )
            route_is_valid = counter_compatibility > 0.0
        counter_roles = {
            normalized_knowledge_roles.get(evidence_id)
            for evidence_id in knowledge_ids
        }
        if counterargument.scheme == ArgumentScheme.NEGATIVE_EVIDENCE:
            roles_are_valid = (
                not normalized_knowledge_roles
                or counter_roles.issubset(
                    {
                        KnowledgeRole.CLINICAL_FEATURE,
                        KnowledgeRole.DIAGNOSTIC_CRITERION,
                        KnowledgeRole.COUNTEREVIDENCE,
                    }
                )
            )
        else:
            roles_are_valid = (
                not normalized_knowledge_roles
                or KnowledgeRole.COUNTEREVIDENCE not in counter_roles
            )
        counter_evidence_valid = (
            bool(knowledge_ids)
            and any(value.startswith("S-") for value in counter_ids)
            and counter_ids.issubset(valid_ids)
            and route_is_valid
            and roles_are_valid
        )
        if (
            counter_evidence_valid
            and normalized_patient_evidence
            and counterargument.scheme == ArgumentScheme.NEGATIVE_EVIDENCE
        ):
            cited_patient_text = " ".join(
                normalized_patient_evidence.get(evidence_id, "")
                for evidence_id in counter_ids
                if evidence_id.startswith("S-")
            ).casefold()
            explicit_negative_text = (
                counterargument.premise + " " + cited_patient_text
            ).casefold()
            counter_evidence_valid = any(
                marker in f" {explicit_negative_text} "
                for marker in (
                    " no ",
                    " not ",
                    " without ",
                    " denies ",
                    " denied ",
                    " negative ",
                    " absent ",
                    " normal ",
                    " unremarkable ",
                    " lacks ",
                    " lack of ",
                )
            )
        if not counter_evidence_valid:
            counter_compatibility = 0.0
        valid_counterargument_count += int(counter_evidence_valid)
        add_node(
            ArgumentNode(
                argument_id=counterargument.argument_id,
                node_type="counterargument",
                scheme=counterargument.scheme,
                premise=counterargument.premise,
                conclusion=counterargument.conclusion,
                evidence_ids=counterargument.evidence_ids,
                candidate_id=candidate_id,
                source="verifier_agent",
                priority=SCHEME_PRIORITY[counterargument.scheme],
                evidence_compatibility=counter_compatibility,
                knowledge_node_ids=kg_node_ids,
                diagnostic_paths=kg_paths,
                premise_types=kg_premise_types,
                scheme_source=counterargument.scheme_source,
            )
        )
        relations.append(
            ArgumentRelation(
                source_id=counterargument.argument_id,
                target_id=counterargument.target_argument_id,
                relation=counterargument.relation,
            )
        )
        if not counter_evidence_valid:
            challenge_id = f"Q-{counterargument.argument_id}"
            add_node(
                ArgumentNode(
                    argument_id=challenge_id,
                    node_type="challenge",
                    scheme=ArgumentScheme.CRITICAL_QUESTION,
                    premise=(
                        "The counterargument lacks valid patient evidence and "
                        "diagnosis-aligned knowledge-graph evidence."
                    ),
                    conclusion=f"{counterargument.argument_id} is undercut.",
                    evidence_ids=(),
                    candidate_id=candidate_id,
                    source="evidence_validator",
                    priority=SCHEME_PRIORITY[ArgumentScheme.CRITICAL_QUESTION],
                )
            )
            relations.append(
                ArgumentRelation(
                    source_id=challenge_id,
                    target_id=counterargument.argument_id,
                    relation=RelationType.UNDERCUTS,
                )
            )

    for left, right in combinations(proposal.candidates, 2):
        relations.extend(
            (
                ArgumentRelation(
                    source_id=left.candidate_id,
                    target_id=right.candidate_id,
                    relation=RelationType.REBUTS,
                ),
                ArgumentRelation(
                    source_id=right.candidate_id,
                    target_id=left.candidate_id,
                    relation=RelationType.REBUTS,
                ),
            )
        )

    if proposal.abstain or verifier.abstain:
        for candidate in proposal.candidates:
            challenge_id = f"Q-ABSTAIN-{candidate.candidate_id}"
            add_node(
                ArgumentNode(
                    argument_id=challenge_id,
                    node_type="challenge",
                    scheme=ArgumentScheme.CRITICAL_QUESTION,
                    premise="An argumentation agent explicitly requested abstention.",
                    conclusion=f"{candidate.candidate_id} is not accepted.",
                    evidence_ids=(),
                    candidate_id=candidate.candidate_id,
                    source="argument_graph_builder",
                    priority=SCHEME_PRIORITY[ArgumentScheme.CRITICAL_QUESTION],
                )
            )
            relations.append(
                ArgumentRelation(
                    source_id=challenge_id,
                    target_id=candidate.candidate_id,
                    relation=RelationType.UNDERCUTS,
                )
            )

    parsed_argument_count = len(proposed_arguments)
    structurally_valid_argument_count = max(
        proposal.raw_argument_count - proposal.invalid_argument_count,
        0,
    )
    reviewed_count = len({review.argument_id for review in verifier.reviews})
    quality = ArgumentGraphQuality(
        argument_schema_validity=(
            structurally_valid_argument_count / proposal.raw_argument_count
            if proposal.raw_argument_count
            else 1.0
        ),
        argument_evidence_validity=(
            valid_argument_count / parsed_argument_count
            if parsed_argument_count
            else 0.0
        ),
        verifier_review_coverage=(
            reviewed_count / parsed_argument_count
            if parsed_argument_count
            else 0.0
        ),
        valid_evidence_reference_fraction=(
            valid_reference_count / reference_count
            if reference_count
            else 0.0
        ),
        argument_evidence_compatibility=(
            argument_compatibility_total / parsed_argument_count
            if parsed_argument_count
            else 0.0
        ),
        argument_scheme_validity=(
            scheme_valid_argument_count / parsed_argument_count
            if parsed_argument_count
            else 0.0
        ),
        supported_review_grounding=(
            grounded_supported_review_count / supported_review_count
            if supported_review_count
            else 1.0
        ),
        counterargument_evidence_validity=(
            valid_counterargument_count / len(verifier.counterarguments)
            if verifier.counterarguments
            else 1.0
        ),
        typed_binding_validity=(
            valid_typed_binding_count / parsed_argument_count
            if parsed_argument_count
            else 1.0
        ),
    )
    return PatientArgumentGraph(
        graph_id=f"argument:{case_id}",
        nodes=tuple(nodes),
        relations=tuple(relations),
        quality=quality,
        patient_evidence_claims=patient_claims,
        knowledge_claims=tuple(knowledge_inventory),
        seed_diagnosis=seed_diagnosis,
        seed_candidate_id=seed_candidate_id,
        seed_source=seed_source,
    )
