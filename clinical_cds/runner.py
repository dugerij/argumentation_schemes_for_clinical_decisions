from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from clinical_cds.argumentation import (
    ArgumentReview,
    ArgumentScheme,
    BINDING_VALIDATION_ID,
    BindingApplication,
    COUNTER_TYPING_ID,
    KG_PATIENT_COUNTER_BINDING_ID,
    KnowledgeClaim,
    KnowledgeRole,
    LLM_ENTAILMENT_BINDING_ID,
    CandidateInventoryEntry,
    CounterArgument,
    DiagnosisCandidate,
    EVIDENCE_VALIDATION_ID,
    MAX_CANDIDATES,
    PatientArgumentGraph,
    PatientEvidenceClaim,
    ProposedArgument,
    REVIEW_GROUNDING_ID,
    ReviewVerdict,
    ReasonerProposal,
    SCHEME_TYPING_ID,
    VerifierReport,
    build_patient_argument_graph,
    family_candidate_inventory_entries,
    family_entailment_shortlist,
    ground_reasoner_proposal,
    kg_candidate_inventory_entries,
    knowledge_claims,
)
from clinical_cds.argumentation_v3 import (
    VERSION_III_ACTIVATION_ID, DirectDecision, ProtectedIncumbent,
    ResolutionAction,
    attack_validation_schema,
    evidence_aware_activation_schema,
    direct_differential_schema, differential_attack_schema,
    parse_direct_differential, parse_differential_attack,
    parse_attack_validation, parse_evidence_aware_activation,
    resolve_direct_differential,
)
from clinical_cds.method_contract import (
    ARGUMENT_METHOD,
    DEFAULT_ARGUMENT_METHOD,
    ArgumentMethodContract,
    argument_method_contract,
)
from clinical_cds.model import DiagnosticModel, OUTPUT_SCHEMA
from clinical_cds.normalization import label_key
from clinical_cds.evidence_graph import (
    atomize_retrieved_family_facts,
    qualifies_decisive_anchor,
)
from clinical_cds.typed_binding import (
    assess_typed_binding,
    deterministic_binding_authority,
)
from graphrag_runtime.provenance_contract import (
    canonical_sha256 as provenance_contract_sha256,
    validate_bundle_citation_allowlist,
)
from graphrag_runtime.medgemma_prompt_budget import MAX_COMPLETION_TOKENS
from clinical_cds.retrieval import (
    CANDIDATE_FIRST_RETRIEVER_VERSION,
    SEMANTIC_ROUTED_RETRIEVER_VERSION,
    patient_evidence,
    render_case,
    render_flat_retrieval,
    render_graph_retrieval,
)
from clinical_cds.schema import (
    ClinicalCase,
    ExperimentMode,
    PredictedObservation,
    PredictionRecord,
    RetrievalBundle,
)

STANDARD_PROMPT_VERSION = "clinical-diagnostic-standard-v1"

ARGUMENT_PROMPT_VERSION = ARGUMENT_METHOD.prompt_version
PROMPT_VERSION = ARGUMENT_PROMPT_VERSION
ARGUMENT_MODES = {
    ExperimentMode.STRUCTURED_ARGUMENT,
    ExperimentMode.SYMBOLIC_ARGUMENT,
    ExperimentMode.EVIDENCE_GROUNDED_ARGUMENTATION,
}
STANDARD_SYSTEM_PROMPT = """Determine the single most likely current diagnosis
from only the submitted clinical state and supplied evidence. Separate current
findings from history, and preserve the scope of each result: a normal value is
an observation about that measured property, not an unstated negation of a
broader disease; a subtype, severity, or risk modifier does not establish its
parent disease alone. Treat evidence as negative only when the record states an
explicit scoped negative or an exact executable comparison contradicts the
proposition. Cite only supplied S or K identifiers. If you provide a diagnosis
in answer, set abstain=false. Set abstain=true only when no diagnosis, including
a defensible parent diagnosis, can be returned; in that case answer must be an
empty string. Never return a diagnosis and abstain=true together. Return exactly
one JSON object matching the schema and no other text."""

VERSION_III_ACTIVATION_SYSTEM_PROMPT = """Choose up to four supplied disease
families that most deserve direct diagnostic comparison for this current
encounter. Retrieval rank is only a weak tie-breaker. Prefer families with
current positive diagnostic tests, explicit current diagnoses, exact abnormal
measurements satisfying warrants, or independent high-information findings.
Down-rank historical-only conditions, incidental comorbidities, risk-only
support, severity-only support, and many weak or duplicated lexical matches.
Pairs marked established_decisive_anchor=true meet a conservative executable
test or threshold boundary against a sourced diagnostic criterion. Prefer them
over unmarked evidence only when they explain the same current finding; do not
treat a marker as permission to invent a result or ignore contradiction.
Preserve competing explanations of the same major finding and coverage of
different major findings. Cite one or two immutable pair IDs owned by every
selected family. Do not diagnose, rewrite evidence, invent IDs, or use the gold
label. Return only schema-valid JSON."""

VERSION_III_DIRECT_SYSTEM_PROMPT = """Answer the current-encounter diagnostic
question in two explicit steps. First select the single best-supported supplied
disease family by comparing all families together. Only then consider a child
label within that selected family. Use
only the immutable patient/KG pair records. Select the candidate that best
explains the high-information current findings, not merely any condition that
may coexist. Cite decisive pair IDs owned by that candidate, name the strongest
different supplied alternative and its evidence, and list important patient evidence left
unexplained. An established_decisive_anchor=true pair is a sourced criterion
with an executable current test or threshold binding; prefer it over generic
support when it explains the same finding, but do not let it override a scoped
contradiction. A symptom, risk factor, manifestation, historical condition, or
severity fact cannot establish a diagnosis alone. Choose insufficient or
none-of-supplied-candidates when warranted. When two evidence-cited families
remain plausible but neither is established, use candidate_id and
strongest_alternative_id with their owned pair IDs to preserve the leading
differential; this remains an abstention, not a diagnosis. A child label is
allowed only when its own decisive pair evidence distinguishes it from sibling
subtypes. Otherwise return the selected parent family. Do not rewrite evidence, invent
IDs, or use outside knowledge. Return only schema-valid JSON."""

VERSION_III_VERIFY_SYSTEM_PROMPT = """Independently attack the proposed direct
differential using the complete immutable context, including evidence belonging
to rejected candidates. Attack only for citation failure, wrong subject or
encounter, diagnostic insufficiency, explicit contradiction, a better-supported
supplied alternative, unsupported child specificity, or material unexplained
evidence. Evidence supporting another diagnosis does not itself attack the
proposal: diagnoses may coexist. A family-level attack must cite a target-owned
pair that directly falsifies a necessary proposition of the target. A
better-alternative attack must name that candidate and cite both its independently
establishing pair evidence and the target-owned falsifying pair. If no concrete
grounded defect exists, return attack=false. Do not rewrite evidence, invent
IDs, or make the terminal decision. Return only schema-valid JSON."""
VERSION_III_ATTACK_VALIDATION_SYSTEM_PROMPT = """Independently validate only
the stated attack against the exact proposed diagnosis, its decisive evidence,
the cited immutable pairs, and the original patient findings. Apply this strict
counterfactual test: if the alternative diagnosis were removed, would the cited
evidence still directly falsify a necessary proposition of the target? If no,
the attack does not defeat the target. Evidence supporting another family,
failure to explain one finding, coexistence, a risk factor, or evidence against
a different subtype is not a family-level contradiction. Absence of a severity
feature cannot refute the parent disease. Separately decide whether alternative-
owned citations independently establish the alternative; mere compatibility,
possible causation, or a test used to investigate it is insufficient. Mark
scope=child_only when only the proposed child is falsified, and scope=family
only when a necessary family proposition is falsified. Do not select a diagnosis
or invent evidence. Return only schema-valid JSON."""

MAX_COMPLETION_ATTEMPTS = 3
MAX_RETRY_OUTPUT_TOKENS = MAX_COMPLETION_TOKENS
MAX_RETRY_CONTEXT_WINDOW = 32768
CURRENT_VERIFIER_RETRY_OUTPUT_TOKEN_CAP = 3072
CURRENT_GENERATION_FAILURE_POLICY = "record-case-error-and-continue"
CURRENT_PATIENT_FINDING_CONTRACT = "schema-enumerated-extractive-s-span-v1"
CURRENT_APPLICATION_CANONICALIZATION = "kg-role-positive-application-v1"
CURRENT_PATIENT_FINDING_PROMPT_RENDERING = "schema-enum-no-duplicated-inventory-v1"

def _uses_evidence_grounded_method(method: ArgumentMethodContract) -> bool:
    return method == ARGUMENT_METHOD

def _uses_compact_prompt_rendering(method: ArgumentMethodContract) -> bool:
    return False

@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    output_dir: Path
    predictions_path: Path
    argument_traces_path: Path
    manifest_path: Path
    records: tuple[PredictionRecord, ...]

@dataclass(frozen=True)
class CachedCompletion:
    response: str
    latency_seconds: float
    cache_hit: bool
    prompt_hash: str
    attempt_count: int
    context_window: int
    max_output_tokens: int
    prompt_version: str
    content_fingerprint: str
    system_prompt_sha256: str
    user_prompt_sha256: str
    output_schema_sha256: str

@dataclass(frozen=True)
class ArgumentTeamResult:
    trace_id: str
    case_id: str
    proposal: ReasonerProposal | None
    verifier: VerifierReport | None
    graph: PatientArgumentGraph | None
    reasoner_prompt_hash: str
    verifier_prompt_hash: str
    reasoner_attempt_count: int
    verifier_attempt_count: int
    latency_seconds: float
    cache_hit: bool
    error: str | None
    argument_method: ArgumentMethodContract = DEFAULT_ARGUMENT_METHOD
    candidate_inventory: tuple[CandidateInventoryEntry, ...] = ()
    retrieval_bundle_sha256: str = ""
    candidate_inventory_sha256: str = ""
    reasoner_artifact_sha256: str = ""
    critic_artifact_sha256: str = ""
    shared_argument_critic_artifact_sha256: str = ""
    shared_argument_critic_artifact_bytes: int = 0
    critic_relations: tuple[dict[str, Any], ...] = ()
    flat_rag_seed_diagnosis: str = ""
    flat_rag_seed_prompt_hash: str = ""
    flat_rag_seed_abstained: bool = True
    flat_rag_seed_audit: dict[str, Any] = field(default_factory=dict)
    raw_reasoner_payload: dict[str, Any] = field(default_factory=dict)
    raw_verifier_payload: dict[str, Any] = field(default_factory=dict)
    verifier_output_token_cap: int | None = None
    system_counterarguments: tuple[CounterArgument, ...] = ()
    generation_recovery: dict[str, Any] = field(default_factory=dict)
    knowledge_source_provenance: tuple[dict[str, Any], ...] = ()
    semantic_routing: dict[str, Any] = field(default_factory=dict)
    semantic_routing_sha256: str = ""
    semantic_routing_authorization: dict[str, Any] = field(
        default_factory=dict
    )
    citation_allowlist: tuple[str, ...] = ()
    citation_allowlist_sha256: str = ""
    entailment_shortlist: tuple[dict[str, Any], ...] = ()
    dialectical_trace: dict[str, Any] = field(default_factory=dict)
    dialectical_resolution: dict[str, Any] = field(default_factory=dict)
    selected_candidate_id: str = ""
    selected_argument_ids: tuple[str, ...] = ()
    selected_evidence_ids: tuple[str, ...] = ()
    subtype_filter: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "case_id": self.case_id,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "raw_reasoner_payload": self.raw_reasoner_payload,
            "verifier": self.verifier.to_dict() if self.verifier else None,
            "raw_verifier_payload": self.raw_verifier_payload,
            "argument_graph": self.graph.to_dict() if self.graph else None,
            "reasoner_prompt_hash": self.reasoner_prompt_hash,
            "verifier_prompt_hash": self.verifier_prompt_hash,
            "generation_attempts": {
                "reasoner": self.reasoner_attempt_count,
                "verifier": self.verifier_attempt_count,
            },
            "latency_seconds": self.latency_seconds,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "candidate_inventory": [
                candidate.to_dict() for candidate in self.candidate_inventory
            ],
            "entailment_shortlist": [
                dict(item) for item in self.entailment_shortlist
            ],
            "dialectical_trace": copy.deepcopy(self.dialectical_trace),
            "dialectical_resolution": copy.deepcopy(
                self.dialectical_resolution
            ),
            "subtype_filter": copy.deepcopy(self.subtype_filter),
            "argument_input": {
                "source": self.argument_method.retrieval_bundle_role,
                "ordered_retrieval_bundle_sha256": (
                    self.retrieval_bundle_sha256
                ),
                "candidate_inventory_sha256": (
                    self.candidate_inventory_sha256
                ),
                **(
                    {
                        "citation_allowlist": list(self.citation_allowlist),
                        "citation_allowlist_sha256": (
                            self.citation_allowlist_sha256
                        ),
                    }
                    if _uses_evidence_grounded_method(self.argument_method)
                    else {}
                ),
                "flat_rag_dependency": (
                    "seed" if self.argument_method.uses_flat_rag_seed else "none"
                ),
                **(
                    {
                        "semantic_routing_sha256": (
                            self.semantic_routing_sha256
                        ),
                        "semantic_routing_authorization_sha256": (
                            _json_sha256(
                                self.semantic_routing_authorization
                            )
                        ),
                    }
                    if self.semantic_routing
                    else {}
                ),
            },
            "critic_relations": [dict(value) for value in self.critic_relations],
            "shared_argument_critic_artifact": {
                "reasoner_artifact_sha256": self.reasoner_artifact_sha256,
                "critic_artifact_sha256": self.critic_artifact_sha256,
                "combined_sha256": (
                    self.shared_argument_critic_artifact_sha256
                ),
                "canonical_json_bytes": (
                    self.shared_argument_critic_artifact_bytes
                ),
                "used_by": (
                    ["evidence_grounded_argumentation"]
                    if _uses_evidence_grounded_method(self.argument_method)
                    else ["structured_argument", "symbolic_argument"]
                ),
            },
            "argumentation_method": {
                **self.argument_method.to_dict(),
                "maximum_candidates": MAX_CANDIDATES,
                "evidence_validation_id": EVIDENCE_VALIDATION_ID,
                "binding_validation_id": BINDING_VALIDATION_ID,
                "scheme_typing_id": SCHEME_TYPING_ID,
                "counter_typing_id": COUNTER_TYPING_ID,
                "review_grounding_id": REVIEW_GROUNDING_ID,
                "verifier_output_token_cap": self.verifier_output_token_cap,
                **(
                    {
                        "kg_patient_counter_binding_id": (
                            KG_PATIENT_COUNTER_BINDING_ID
                        ),
                        "patient_finding_contract": (
                            CURRENT_PATIENT_FINDING_CONTRACT
                        ),
                        "application_canonicalization": (
                            CURRENT_APPLICATION_CANONICALIZATION
                        ),
                        "generation_failure_policy": CURRENT_GENERATION_FAILURE_POLICY,
                        "verifier_retry_output_token_cap": (
                            CURRENT_VERIFIER_RETRY_OUTPUT_TOKEN_CAP
                        ),
                        **(
                            {
                                "patient_finding_prompt_rendering": (
                                    CURRENT_PATIENT_FINDING_PROMPT_RENDERING
                                )
                            }
                            if _uses_compact_prompt_rendering(
                                self.argument_method
                            )
                            else {}
                        ),
                    }
                    if _uses_evidence_grounded_method(self.argument_method)
                    else {}
                ),
            },
        }
        if self.argument_method.uses_flat_rag_seed:
            payload["flat_rag_seed"] = {
                "diagnosis": self.flat_rag_seed_diagnosis,
                "prompt_hash": self.flat_rag_seed_prompt_hash,
                "abstained": self.flat_rag_seed_abstained,
                "role": self.argument_method.flat_rag_role,
                **self.flat_rag_seed_audit,
            }
        if (
            _uses_evidence_grounded_method(self.argument_method)
            or self.system_counterarguments
            or self.generation_recovery
        ):
            payload["system_counterarguments"] = [
                asdict(counterargument)
                for counterargument in self.system_counterarguments
            ]
            payload["generation_recovery"] = dict(self.generation_recovery)
        if self.knowledge_source_provenance:
            payload["knowledge_source_provenance"] = [
                dict(item) for item in self.knowledge_source_provenance
            ]
        if self.semantic_routing:
            payload["semantic_routing"] = copy.deepcopy(
                self.semantic_routing
            )
            payload["semantic_routing_authorization"] = copy.deepcopy(
                self.semantic_routing_authorization
            )
        return payload

class ModelCompletionFailure(RuntimeError):
    """A model call exhausted bounded recovery without a valid completion."""

    def __init__(self, role: str, cause: object):
        self.role = role
        super().__init__(f"{role}: {cause}")

def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def _json_sha256(value: object) -> str:
    return _text_sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"))
    )

def _retrieval_bundle_sha256(bundle: RetrievalBundle) -> str:
    return _json_sha256([
        {
            "evidence_id": fact.evidence_id,
            "node_id": fact.node_id,
            "category": fact.category,
            "diagnosis_label": fact.diagnosis_label,
            "premise_type": fact.premise_type,
            "text": fact.text,
            "score": fact.score,
            "diagnostic_path": list(fact.diagnostic_path),
        }
        for fact in bundle.facts
    ])

def provenance_retrieval_bundle_sha256(bundle: RetrievalBundle) -> str:
    """Hash the exact v1 bundle, including all source-provenance fields.

    The historical bundle hash above is intentionally unchanged so old result
    manifests retain their original meaning.
"""

    return _json_sha256({
        "facts": [
            {
                "evidence_id": fact.evidence_id,
                "node_id": fact.node_id,
                "category": fact.category,
                "diagnosis_label": fact.diagnosis_label,
                "premise_type": fact.premise_type,
                "text": fact.text,
                "score": fact.score,
                "diagnostic_path": list(fact.diagnostic_path),
                "knowledge_source_ids": list(fact.knowledge_source_ids),
                "source_chunk_id": fact.source_chunk_id,
            }
            for fact in bundle.facts
        ],
        "citation_allowlist": list(_bundle_citation_allowlist(bundle)),
        "family_routes": [
            {
                "family_rank": route.family_rank,
                "graph_id": route.graph_id,
                "family_key": route.family_key,
                "representative_diagnosis": route.representative_diagnosis,
                "alternatives": [
                    {
                        "candidate_id": alternative.candidate_id,
                        "diagnosis_label": alternative.diagnosis_label,
                        "graph_id": alternative.graph_id,
                        "diagnostic_path": list(alternative.diagnostic_path),
                        "source_chunk_ids": list(alternative.source_chunk_ids),
                        "original_candidate_rank": (
                            alternative.original_candidate_rank
                        ),
                        "representative": alternative.representative,
                    }
                    for alternative in route.alternatives
                ],
            }
            for route in bundle.family_routes
        ],
        "family_child_facts": [
            {
                "graph_id": item.graph_id,
                "evidence_id": item.fact.evidence_id,
                "node_id": item.fact.node_id,
                "diagnosis_label": item.fact.diagnosis_label,
                "premise_type": item.fact.premise_type,
                "text": item.fact.text,
                "diagnostic_path": list(item.fact.diagnostic_path),
                "knowledge_source_ids": list(item.fact.knowledge_source_ids),
                "source_chunk_id": item.fact.source_chunk_id,
            }
            for item in bundle.family_child_facts
        ],
    })

def _bundle_citation_allowlist(bundle: RetrievalBundle) -> tuple[str, ...]:
    configured = bundle.citation_allowlist or tuple(
        fact.source_chunk_id for fact in bundle.facts
    )
    if not configured and not bundle.facts:
        return ()
    return validate_bundle_citation_allowlist(
        (
            fact.source_chunk_id
            for fact in (
                *bundle.facts,
                *(item.fact for item in bundle.family_child_facts),
            )
        ),
        configured,
    )

def _method_bundle_sha256(
    bundle: RetrievalBundle,
    method: ArgumentMethodContract,
) -> str:
    if method == ARGUMENT_METHOD:
        return provenance_retrieval_bundle_sha256(bundle)
    return _retrieval_bundle_sha256(bundle)

def _candidate_inventory_sha256(
    candidates: Iterable[CandidateInventoryEntry],
) -> str:
    return _json_sha256([
        candidate.to_dict() for candidate in candidates
    ])

def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

class ResponseCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def _path(self, prompt_hash: str) -> Path:
        return self.cache_dir / prompt_hash[:2] / f"{prompt_hash}.json"

    def get(self, prompt_hash: str) -> tuple[str, float] | None:
        path = self._path(prompt_hash)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            str(payload["response"]),
            float(payload.get("latency_seconds") or 0.0),
        )

    def put(
        self,
        prompt_hash: str,
        response: str,
        model_id: str,
        latency_seconds: float,
        role: str,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        path = self._path(prompt_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "prompt_hash": prompt_hash,
                    "model_id": model_id,
                    "prompt_version": prompt_version,
                    "role": role,
                    "response": response,
                    "latency_seconds": latency_seconds,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

def _first_json_object(value: str) -> dict[str, Any]:
    normalized = value.replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(normalized)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = normalized.find("{")
    if start < 0:
        raise ValueError("Model output did not contain a JSON object.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(normalized)):
        character = normalized[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                payload = json.loads(normalized[start : index + 1])
                if not isinstance(payload, dict):
                    raise ValueError("Model output JSON must be an object.")
                return payload
    raise ValueError("Model output contained an incomplete JSON object.")

def _string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        for chunk in str(item).split(","):
            text = " ".join(chunk.split()).strip().lstrip(",")
            text = text.strip()
            if text and text.casefold() not in seen:
                seen.add(text.casefold())
                output.append(text)
    return tuple(output)


def _graph_family_match(
    entry: CandidateInventoryEntry,
    incumbent_key: str,
    child_labels: Iterable[str],
) -> bool:
    if incumbent_key == label_key(entry.diagnosis):
        return True
    if any(incumbent_key == label_key(label) for label in child_labels):
        return True
    for path in entry.diagnostic_paths:
        for node in path:
            if incumbent_key == label_key(node):
                return True
    return False


def _graph_family_match_with_evidence_overlap(
    entry: CandidateInventoryEntry,
    incumbent_key: str,
    candidate_records: Iterable[Mapping[str, str]],
    cited: Iterable[str],
    child_labels: Iterable[str],
) -> bool:
    if _graph_family_match(entry, incumbent_key, child_labels):
        return True
    cited_ids = {str(item).strip() for item in cited if str(item).strip()}
    if not cited_ids:
        return False
    for record in candidate_records:
        if (
            str(record.get("candidate_id")) != entry.candidate_id
            or record.get("knowledge_evidence_id") is None
        ):
            continue
        if str(record.get("knowledge_evidence_id")) in cited_ids:
            return True
    return False


def _evidence_anchor_report(
    candidates: Iterable[CandidateInventoryEntry],
    records: Iterable[Mapping[str, str]],
    patient_sections: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """Rank families from typed, current evidence before LLM comparison.

    The score is intentionally transparent: a current imaging/result binding to
    a diagnostic criterion outranks generic clinical features and risk factors.
    It is a shortlist rule, not a diagnostic classifier.
    """
    authority_weight = {
        "exact_positive_test": 6,
        "exact_measurement": 5,
        "explicit_current_diagnosis": 5,
        "explicit_negation": 4,
    }
    current_sections = {"imaging", "results", "laboratory", "labs", "investigations"}
    history_sections = {"past_medical_history", "family_history", "social_history"}
    by_candidate: dict[str, list[Mapping[str, str]]] = {}
    for record in records:
        by_candidate.setdefault(str(record["candidate_id"]), []).append(record)
    report: list[dict[str, Any]] = []
    for candidate in candidates:
        best_by_patient: dict[str, tuple[int, Mapping[str, str]]] = {}
        for record in by_candidate.get(candidate.candidate_id, []):
            if not bool(record.get("anchor_authorized")):
                continue
            section = str(patient_sections.get(str(record["patient_evidence_id"]), ""))
            authority_tier = authority_weight.get(
                str(record.get("anchor_authority_type", "")), 0
            )
            if section in history_sections:
                section_tier = 0
            elif section in current_sections:
                section_tier = 2
            else:
                section_tier = 1
            tier = authority_tier + section_tier
            patient_id = str(record["patient_evidence_id"])
            current = best_by_patient.get(patient_id)
            if current is None or tier > current[0] or (
                tier == current[0] and str(record["pair_id"]) < str(current[1]["pair_id"])
            ):
                best_by_patient[patient_id] = (tier, record)
        strongest = max((item[0] for item in best_by_patient.values()), default=0)
        independent = sum(1 for tier, _ in best_by_patient.values() if tier >= 4)
        score = strongest * 100 + independent * 10 + len(best_by_patient)
        anchors = [record for _, record in sorted(
            best_by_patient.values(), key=lambda item: (-item[0], str(item[1]["pair_id"]))
        )][:4]
        report.append({
            "candidate_id": candidate.candidate_id,
            "diagnosis": candidate.diagnosis,
            "evidence_anchor_score": score,
            "strongest_evidence_tier": strongest,
            "independent_high_tier_findings": independent,
            "risk_only_support": False,
            "anchor_authority_required": True,
            "anchor_pair_ids": [str(item["pair_id"]) for item in anchors],
            "decisive_source_anchor_pair_ids": [
                str(record["pair_id"])
                for record in by_candidate.get(candidate.candidate_id, [])
                if bool(record.get("decisive_source_anchor"))
            ],
            "established_decisive_anchor_pair_ids": [
                str(record["pair_id"])
                for record in by_candidate.get(candidate.candidate_id, [])
                if bool(record.get("established_decisive_anchor"))
            ],
        })
    return tuple(sorted(
        report,
        key=lambda item: (-int(item["evidence_anchor_score"]),
                          -int(item["strongest_evidence_tier"]),
                          str(item["candidate_id"])),
    ))

def _parse_observations(value: object) -> tuple[PredictedObservation, ...]:
    if not isinstance(value, list):
        return ()
    observations: list[PredictedObservation] = []
    for item in value:
        if isinstance(item, dict):
            text = " ".join(str(item.get("text") or "").split()).strip()
            source_id = " ".join(str(item.get("source_id") or "").split()).strip() or None
        else:
            text = " ".join(str(item).split()).strip()
            source_id = None
        if text:
            observations.append(
                PredictedObservation(text=text, source_id=source_id)
            )
    return tuple(observations)

def _normalize_answer(answer: object, case: ClinicalCase) -> str:
    text = " ".join(str(answer or "").split()).strip()
    option_key = text.upper().strip(" .():")
    if case.options and option_key in case.options:
        return case.options[option_key]
    if case.options:
        for option in case.options.values():
            if text.casefold().strip(" .") == option.casefold().strip(" ."):
                return option

    diagnostic_prefixes = (
        r"^(?:answer|diagnosis)\s*:\s*",
        r"^(?:the\s+)?(?:most\s+likely\s+)?diagnosis\s+is\s+",
        r"^(?:the\s+)?patient(?:'s)?\s+clinical\s+presentation\s+is\s+"
        r"(?:most\s+)?consistent\s+with\s+",
        r"^(?:the\s+)?clinical\s+presentation\s+is\s+"
        r"(?:most\s+)?consistent\s+with\s+",
    )
    for pattern in diagnostic_prefixes:
        normalized = re.sub(pattern, "", text, flags=re.IGNORECASE)
        if normalized != text:
            text = normalized
            break
    text = re.split(r"\s+(?:based on|because|given)\s+", text, maxsplit=1)[0]
    return text.strip(" .")

def _retrieved_metadata(bundle: RetrievalBundle) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": fact.evidence_id,
            "node_id": fact.node_id,
            "category": fact.category,
            "diagnosis_label": fact.diagnosis_label,
            "premise_type": fact.premise_type,
            "score": fact.score,
            "diagnostic_path": list(fact.diagnostic_path),
            "source_chunk_id": fact.source_chunk_id,
            **(
                {
                    "knowledge_source_ids": list(
                        fact.knowledge_source_ids
                    )
                }
                if fact.knowledge_source_ids
                else {}
            ),
        }
        for fact in bundle.facts
    ]

def _semantic_routing_metadata(
    retriever: object,
    case_id: str,
) -> dict[str, Any]:
    getter = getattr(retriever, "routing_result", None)
    if not callable(getter):
        return {}
    result = getter(case_id)
    if result is None:
        return {}
    to_dict = getattr(result, "to_dict", None)
    payload = to_dict() if callable(to_dict) else result
    if not isinstance(payload, dict):
        raise TypeError("Semantic routing provenance must be a dictionary.")
    return copy.deepcopy(payload)

def _semantic_routing_authorization(retriever: object) -> dict[str, Any]:
    payload = getattr(retriever, "authorization_manifest", {})
    if not isinstance(payload, dict):
        raise TypeError(
            "Semantic routing authorization must be a dictionary."
        )
    return copy.deepcopy(payload)

_PATIENT_FINDING_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|[;\r\n]+")

def _exact_patient_findings(
    case: ClinicalCase,
) -> tuple[tuple[str, str], ...]:
    findings: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for evidence_id, _, source in patient_evidence(case):
        normalized_source = " ".join(source.split())
        parts = [
            " ".join(part.split())
            for part in _PATIENT_FINDING_BOUNDARY_RE.split(source)
            if " ".join(part.split())
        ]
        if 8 <= len(normalized_source) <= 240:
            parts.insert(0, normalized_source)
        for part in parts:
            if len(part) < 8:
                continue
            if len(part) > 240:
                words = part.split()
                chunks: list[str] = []
                current: list[str] = []
                for word in words:
                    candidate = " ".join((*current, word))
                    if current and len(candidate) > 240:
                        chunks.append(" ".join(current))
                        current = [word]
                    else:
                        current.append(word)
                if current:
                    chunks.append(" ".join(current))
            else:
                chunks = [part]
            for chunk in chunks:
                key = (evidence_id, chunk)
                if len(chunk) >= 8 and key not in seen:
                    seen.add(key)
                    findings.append(key)
    return tuple(findings)

def _build_standard_prompt(
    case: ClinicalCase,
    mode: ExperimentMode,
    bundle: RetrievalBundle,
) -> tuple[str, tuple[str, ...]]:
    task_instruction = (
        "Select the single best answer from the supplied options."
        if case.options
        else "State the single most likely diagnosis."
    )
    sections = [
        f"Task: {task_instruction}",
        "",
        "Clinical state:",
        render_case(case),
    ]
    if mode == ExperimentMode.FLAT_RAG:
        sections.extend(
            [
                "",
                "Retrieved guideline premises (independent flat records):",
                render_flat_retrieval(bundle),
            ]
        )
    elif mode == ExperimentMode.GRAPH_RAG:
        sections.extend(
            [
                "",
                "Retrieved guideline subgraph:",
                render_graph_retrieval(bundle),
            ]
        )
    elif mode != ExperimentMode.DIRECT:
        raise ValueError(f"Standard prediction does not support mode {mode.value}.")

    section_ids = tuple(item[0] for item in patient_evidence(case))
    knowledge_ids = bundle.evidence_ids if mode != ExperimentMode.DIRECT else ()
    valid_ids = section_ids + knowledge_ids
    sections.extend(
        [
            "",
            (
                "Return JSON with: answer, concise reasoning of at most three sentences, "
                "at most four citations, at most three observations "
                "(each with text and source_id), and abstain."
            ),
            (
                "The answer value must contain only the diagnosis or selected "
                "option text. Citations and observation source_id values must "
                "use only these exact identifiers: "
                + ", ".join(valid_ids)
            ),
        ]
    )
    return "\n".join(sections), valid_ids

def _evidence_constrained_output_schema(
    valid_ids: Iterable[str],
) -> dict[str, object]:
    schema = copy.deepcopy(OUTPUT_SCHEMA)
    identifiers = list(dict.fromkeys(valid_ids))
    if not identifiers:
        return schema
    identifier_schema = {
        "type": "string",
        "enum": identifiers,
    }
    properties = schema["properties"]
    properties["citations"]["items"] = identifier_schema
    properties["observations"]["items"]["properties"]["source_id"] = (
        identifier_schema
    )
    return schema

def _validate_standard_diagnostic_output(payload: Mapping[str, Any]) -> None:
    """Reject the contradictory diagnosis-plus-abstention state."""
    answer = str(payload.get("answer") or "").strip()
    abstain = payload.get("abstain")
    if not isinstance(abstain, bool):
        raise ValueError("Standard diagnostic abstain must be boolean.")
    if bool(answer) == abstain:
        raise ValueError(
            "A standard diagnostic output must contain either one diagnosis "
            "or an abstention, never both or neither."
        )

def _argument_scheme_for_role(role: KnowledgeRole) -> ArgumentScheme:
    return {
        KnowledgeRole.DIAGNOSTIC_CRITERION: ArgumentScheme.DIAGNOSTIC_CRITERION,
        KnowledgeRole.CLINICAL_FEATURE: ArgumentScheme.CLINICAL_SIGN,
        KnowledgeRole.RISK_FACTOR: ArgumentScheme.RISK_FACTOR,
        KnowledgeRole.GUIDELINE: ArgumentScheme.GUIDELINE_AUTHORITY,
    }.get(role, ArgumentScheme.DIAGNOSTIC_CRITERION)

MAX_DIALOGUE_BINDING_OPTIONS = 8

def _rank_dossier_binding_options(
    patient_pairs: Iterable[tuple[str, str]],
    knowledge_evidence_ids: Iterable[str],
    claims: Iterable[KnowledgeClaim],
    normalizer: object | None,
) -> tuple[dict[str, str], ...]:
    """Rank patient-finding and knowledge-warrant pairs for activation."""
    claim_index = {claim.evidence_id: claim for claim in claims}
    ranked: list[tuple[int, int, float, int, str, str, dict[str, str]]] = []
    role_priority = {
        KnowledgeRole.DIAGNOSTIC_CRITERION: 4,
        KnowledgeRole.CLINICAL_FEATURE: 3,
        KnowledgeRole.GUIDELINE: 2,
        KnowledgeRole.RISK_FACTOR: 1,
    }
    for patient_id, finding in patient_pairs:
        for knowledge_id in tuple(dict.fromkeys(knowledge_evidence_ids)):
            claim = claim_index.get(knowledge_id)
            if claim is None:
                continue
            assessment = assess_typed_binding(
                finding, claim.text, role=claim.role.value, normalizer=normalizer
            )
            questions = {
                item.question_id: item.passed for item in assessment.critical_questions
            }
            ranked.append((
                int(assessment.admissible),
                int(questions.get("concept_compatible", False)),
                assessment.score,
                role_priority.get(claim.role, 0),
                patient_id,
                knowledge_id,
                {"patient_evidence_id": patient_id, "patient_finding": finding,
                 "knowledge_evidence_id": knowledge_id,
                 "typed_binding_admissible": bool(assessment.admissible)},
            ))
    ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], -row[3], row[4], row[5]))
    selected = []
    seen_nodes: set[str] = set()
    for row in ranked:
        node_id = claim_index[row[5]].node_id or row[5]
        if node_id in seen_nodes:
            continue
        selected.append(row)
        seen_nodes.add(node_id)
        if len(selected) >= MAX_DIALOGUE_BINDING_OPTIONS:
            break
    selected_ids = {(row[4], row[5]) for row in selected}
    for row in ranked:
        if len(selected) >= MAX_DIALOGUE_BINDING_OPTIONS:
            break
        identity = (row[4], row[5])
        if identity not in selected_ids:
            selected.append(row)
            selected_ids.add(identity)
    return tuple(row[-1] for row in selected)

def _deterministic_bypass_authorized(
    argument: ProposedArgument,
    normalizer: object | None,
) -> bool:
    if argument.binding_validation_id != BINDING_VALIDATION_ID:
        return False
    authority = deterministic_binding_authority(
        argument.patient_finding,
        argument.knowledge_warrant,
        candidate_label=argument.conclusion,
        role=argument.knowledge_role.value,
        normalizer=normalizer,
    )
    return authority.authorized

def _ground_dialogue_binding(
    *,
    argument_id: str,
    candidate: CandidateInventoryEntry,
    patient_finding: str,
    patient_evidence_id: str,
    knowledge_evidence_id: str,
    patient_text: dict[str, str],
    claims: Iterable[KnowledgeClaim],
    normalizer: object | None,
    deterministic_options: Iterable[ProposedArgument] = (),
    semantic_admitted: bool = False,
) -> ProposedArgument | None:
    for option in deterministic_options:
        if (
            option.patient_finding == patient_finding
            and tuple(option.evidence_ids)
            == (patient_evidence_id, knowledge_evidence_id)
        ):
            return replace(option, argument_id=argument_id, model_rationale="")
    claim_index = {claim.evidence_id: claim for claim in claims}
    claim = claim_index.get(knowledge_evidence_id)
    if claim is None:
        return None
    raw_argument = ProposedArgument(
        argument_id="A1",
        scheme=_argument_scheme_for_role(claim.role),
        premise="Submitted evidence binding pending deterministic grounding.",
        conclusion=candidate.diagnosis,
        evidence_ids=(patient_evidence_id, knowledge_evidence_id),
        patient_finding=patient_finding,
        application=BindingApplication.SATISFIES,
        model_rationale="",
    )
    raw = ReasonerProposal(
        candidates=(DiagnosisCandidate(
            candidate.candidate_id,
            candidate.diagnosis,
            (raw_argument,),
        ),),
        preferred_diagnosis="",
        abstain=False,
        raw_argument_count=1,
        invalid_argument_count=0,
    )
    grounded = ground_reasoner_proposal(
        raw,
        (candidate,),
        {claim.evidence_id: claim.role},
        patient_text,
        tuple(claims),
        canonicalize_positive_applications=True,
        allow_model_entailment=True,
        normalizer=normalizer,
    )
    arguments = tuple(
        argument
        for item in grounded.candidates
        for argument in item.arguments
    )
    if not arguments:
        return None
    argument = replace(arguments[0], argument_id=argument_id, model_rationale="")
    if semantic_admitted and not _deterministic_bypass_authorized(argument, normalizer):
        questions = tuple(argument.binding_critical_questions)
        if not any(
            item.get("question_id") == "clinical_entailment_judged"
            for item in questions
        ):
            questions = (*questions, {
                "question_id": "clinical_entailment_judged",
                "passed": True,
                "detail": "independent MedGemma immutable-pair judgment",
            })
        argument = replace(
            argument,
            binding_validation_id=LLM_ENTAILMENT_BINDING_ID,
            binding_critical_questions=questions,
        )
    return argument

class ExperimentRunner:
    def __init__(
        self,
        *,
        model: DiagnosticModel,
        retriever: Any,
        flat_retriever: Any | None = None,
        cache_dir: Path,
        top_k: int = 6,
        reasoner_model: DiagnosticModel | None = None,
        verifier_model: DiagnosticModel | None = None,
        standard_output_token_cap: int | None = None,
        reasoner_output_token_cap: int | None = None,
        verifier_output_token_cap: int | None = None,
        argument_method: (
            str | ArgumentMethodContract
        ) = DEFAULT_ARGUMENT_METHOD,
    ):
        for name, cap in (
            ("standard", standard_output_token_cap),
            ("reasoner", reasoner_output_token_cap),
            ("verifier", verifier_output_token_cap),
        ):
            if cap is not None and cap < 1:
                raise ValueError(f"{name.title()} output token cap must be positive.")
            if cap is not None and cap > MAX_COMPLETION_TOKENS:
                raise ValueError(
                    f"{name.title()} output token cap exceeds the frozen remote "
                    f"maximum of {MAX_COMPLETION_TOKENS}."
                )
        self.model = model
        self.reasoner_model = reasoner_model or model
        self.verifier_model = verifier_model or model
        self.standard_output_token_cap = standard_output_token_cap
        self.reasoner_output_token_cap = reasoner_output_token_cap
        self.verifier_output_token_cap = verifier_output_token_cap
        self.argument_method = argument_method_contract(argument_method)
        uses_candidate_first = retriever.retriever_id.startswith(
            CANDIDATE_FIRST_RETRIEVER_VERSION
        )
        uses_semantic_routing = (
            SEMANTIC_ROUTED_RETRIEVER_VERSION in retriever.retriever_id
            and callable(getattr(retriever, "routing_result", None))
        )
        if uses_candidate_first or uses_semantic_routing:
            raise ValueError(
                "The current method requires the fixed GraphRAG retriever; "
                "candidate-first and semantic-routing variants are not supported."
            )
        self.retriever = retriever
        self.flat_retriever = flat_retriever or retriever
        self.cache = ResponseCache(cache_dir)
        self.top_k = top_k

    def _assert_data_boundary(self, case: ClinicalCase) -> None:
        checked: set[int] = set()
        for model in (
            self.model,
            self.reasoner_model,
            self.verifier_model,
        ):
            if id(model) in checked:
                continue
            checked.add(id(model))
            boundary_check = getattr(model, "assert_data_boundary", None)
            if callable(boundary_check):
                boundary_check(case.dataset)

    def _complete(
        self,
        *,
        role: str,
        prompt_version: str,
        model: DiagnosticModel,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, object] | None,
        recovery_output_schema: dict[str, object] | None = None,
        response_validator: Callable[[dict[str, Any]], None] | None = None,
        response_parser: Callable[[str], dict[str, Any]] | None = None,
        attempt_observer: Callable[[int, str, str | None], None] | None = None,
        max_output_tokens_cap: int | None = None,
        retry_output_tokens_cap: int | None = None,
    ) -> CachedCompletion:
        for name, cap in (
            ("initial", max_output_tokens_cap),
            ("retry", retry_output_tokens_cap),
        ):
            if cap is not None and cap > MAX_COMPLETION_TOKENS:
                raise ValueError(
                    f"{name.title()} completion cap {cap} exceeds the frozen "
                    f"remote maximum of {MAX_COMPLETION_TOKENS}."
                )
        model_identity = getattr(model, "cache_identity", model.model_id)
        base_context = int(getattr(model, "context_window", 8192))
        base_output = min(
            int(getattr(model, "max_output_tokens", 1024)),
            MAX_COMPLETION_TOKENS,
        )
        if max_output_tokens_cap is not None:
            base_output = min(base_output, max_output_tokens_cap)
        retry_context = min(
            max(base_context * 2, base_context + base_output),
            MAX_RETRY_CONTEXT_WINDOW,
        )
        retry_output = min(
            max(base_output * 2, base_output + 512),
            MAX_RETRY_OUTPUT_TOKENS,
        )
        resolved_retry_cap = (
            retry_output_tokens_cap
            if retry_output_tokens_cap is not None
            else max_output_tokens_cap
        )
        if resolved_retry_cap is not None:
            retry_output = min(retry_output, resolved_retry_cap)
        final_context = min(
            max(retry_context * 2, retry_context + retry_output),
            MAX_RETRY_CONTEXT_WINDOW,
        )
        final_output = min(
            max(retry_output * 2, retry_output + 512),
            MAX_RETRY_OUTPUT_TOKENS,
        )
        if resolved_retry_cap is not None:
            final_output = min(final_output, resolved_retry_cap)
        attempt_configs = (
            (base_context, base_output, output_schema),
            (retry_context, retry_output, output_schema),
            (
                final_context,
                final_output,
                recovery_output_schema or output_schema,
            ),
        )
        total_latency = 0.0
        all_cache_hits = True
        last_error: Exception | None = None
        parse_response = response_parser or _first_json_object

        for attempt_count, (
            context_window,
            max_output_tokens,
            attempt_output_schema,
        ) in enumerate(
            attempt_configs,
            start=1,
        ):
            # Greedy decoding can repeat a structurally invalid answer exactly.
            # Make a retry materially different while preserving its evidence
            # and output schema.
            attempt_user_prompt = user_prompt
            if attempt_count > 1 and last_error is not None:
                correction = " ".join(str(last_error).split())[:300]
                attempt_user_prompt = (
                    f"{user_prompt}\n\nValidation correction: the previous response "
                    f"was rejected because {correction}. Return corrected JSON using "
                    "only the supplied evidence and identifiers."
                )
            schema_text = json.dumps(
                attempt_output_schema,
                sort_keys=True,
                separators=(",", ":"),
            )
            retry_identity = (
                ""
                if attempt_count == 1
                else (
                    f"\0retry_attempt={attempt_count}"
                    f"\0context={context_window}"
                    f"\0max_output={max_output_tokens}"
                )
            )
            content_input = (
                role
                + "\0"
                + model_identity
                + (
                    f"\0max_output_cap={max_output_tokens_cap}"
                    if max_output_tokens_cap is not None
                    else ""
                )
                + (
                    f"\0retry_output_cap={retry_output_tokens_cap}"
                    if retry_output_tokens_cap is not None
                    else ""
                )
                + retry_identity
                + "\0"
                + schema_text
                + "\0"
                + system_prompt
                + "\0"
                + attempt_user_prompt
            )
            prompt_hash = hashlib.sha256(
                (
                    prompt_version
                    + "\0"
                    + content_input
                ).encode("utf-8")
            ).hexdigest()
            content_fingerprint = _text_sha256(content_input)
            cached = self.cache.get(prompt_hash)
            if cached is not None:
                response, latency_seconds = cached
                total_latency += latency_seconds
                try:
                    parsed_response = parse_response(response)
                    if response_validator is not None:
                        response_validator(parsed_response)
                except (ValueError, KeyError, TypeError) as exc:
                    if attempt_observer is not None:
                        attempt_observer(attempt_count, response, str(exc))
                    last_error = exc
                    if attempt_count < len(attempt_configs):
                        continue
                else:
                    if attempt_observer is not None:
                        attempt_observer(attempt_count, response, None)
                    return CachedCompletion(
                        response=response,
                        latency_seconds=total_latency,
                        cache_hit=all_cache_hits,
                        prompt_hash=prompt_hash,
                        attempt_count=attempt_count,
                        context_window=context_window,
                        max_output_tokens=max_output_tokens,
                        prompt_version=prompt_version,
                        content_fingerprint=content_fingerprint,
                        system_prompt_sha256=_text_sha256(system_prompt),
                        user_prompt_sha256=_text_sha256(attempt_user_prompt),
                        output_schema_sha256=_text_sha256(schema_text),
                    )

            all_cache_hits = False
            started = time.perf_counter()
            try:
                response = model.complete(
                    system_prompt,
                    attempt_user_prompt,
                    output_schema=attempt_output_schema,
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as exc:
                total_latency += time.perf_counter() - started
                all_cache_hits = False
                if attempt_observer is not None:
                    attempt_observer(attempt_count, "", str(exc))
                raise ModelCompletionFailure(role, exc) from exc
            latency_seconds = time.perf_counter() - started
            total_latency += latency_seconds
            self.cache.put(
                prompt_hash,
                response,
                model.model_id,
                latency_seconds,
                role,
                prompt_version,
            )
            try:
                parsed_response = parse_response(response)
                if response_validator is not None:
                    response_validator(parsed_response)
            except (ValueError, KeyError, TypeError) as exc:
                if attempt_observer is not None:
                    attempt_observer(attempt_count, response, str(exc))
                last_error = exc
                continue
            if attempt_observer is not None:
                attempt_observer(attempt_count, response, None)
            return CachedCompletion(
                response=response,
                latency_seconds=total_latency,
                cache_hit=False,
                prompt_hash=prompt_hash,
                attempt_count=attempt_count,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                prompt_version=prompt_version,
                content_fingerprint=content_fingerprint,
                system_prompt_sha256=_text_sha256(system_prompt),
                user_prompt_sha256=_text_sha256(attempt_user_prompt),
                output_schema_sha256=_text_sha256(schema_text),
            )

        raise ModelCompletionFailure(
            role,
            "Model output remained incomplete or invalid JSON after "
            f"{MAX_COMPLETION_ATTEMPTS} attempts: {last_error}",
        )

    def predict(
        self,
        case: ClinicalCase,
        mode: ExperimentMode,
        *,
        run_id: str,
        bundle: RetrievalBundle | None = None,
    ) -> PredictionRecord:
        if mode in ARGUMENT_MODES:
            raise ValueError(
                "Argument modes require run_argument_team and predict_argument_mode."
            )
        self._assert_data_boundary(case)
        selected_retriever = (
            self.flat_retriever
            if mode == ExperimentMode.FLAT_RAG
            else self.retriever
        )
        if bundle is None:
            bundle = (
                RetrievalBundle(facts=(), query_tokens=())
                if mode == ExperimentMode.DIRECT
                else selected_retriever.retrieve(case, top_k=self.top_k)
            )
        user_prompt, valid_ids = _build_standard_prompt(case, mode, bundle)

        started = time.perf_counter()
        completion_attempt_count = 0
        completion_context_window = 0
        completion_max_output_tokens = 0
        completion_audit: dict[str, Any] = {}
        try:
            completion = self._complete(
                role=f"standard:{mode.value}",
                prompt_version=STANDARD_PROMPT_VERSION,
                model=self.model,
                system_prompt=STANDARD_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=OUTPUT_SCHEMA,
                recovery_output_schema=(
                    _evidence_constrained_output_schema(valid_ids)
                ),
                response_validator=_validate_standard_diagnostic_output,
                max_output_tokens_cap=self.standard_output_token_cap,
                retry_output_tokens_cap=self.standard_output_token_cap,
            )
            payload = _first_json_object(completion.response)
            predicted_label = _normalize_answer(payload.get("answer"), case)
            reasoning = " ".join(
                str(payload.get("reasoning") or "").split()
            ).strip()
            citations = _string_list(payload.get("citations"))
            observations = _parse_observations(payload.get("observations"))
            abstained = not bool(predicted_label)
            error = None
            latency_seconds = completion.latency_seconds
            cache_hit = completion.cache_hit
            prompt_hash = completion.prompt_hash
            completion_attempt_count = completion.attempt_count
            completion_context_window = completion.context_window
            completion_max_output_tokens = completion.max_output_tokens
            completion_audit = {
                "prompt_version": completion.prompt_version,
                "content_fingerprint": completion.content_fingerprint,
                "system_prompt_sha256": completion.system_prompt_sha256,
                "user_prompt_sha256": completion.user_prompt_sha256,
                "output_schema_sha256": completion.output_schema_sha256,
            }
        except Exception as exc:
            predicted_label = ""
            reasoning = ""
            citations = ()
            observations = ()
            abstained = True
            error = f"{type(exc).__name__}: {exc}"
            latency_seconds = time.perf_counter() - started
            cache_hit = False
            prompt_hash = ""

        return PredictionRecord(
            run_id=run_id,
            case_id=case.case_id,
            dataset=case.dataset,
            task=case.task,
            mode=mode,
            model_id=self.model.model_id,
            gold_label=case.gold_label,
            predicted_label=predicted_label,
            reasoning=reasoning,
            citations=citations,
            observations=observations,
            abstained=abstained,
            latency_seconds=round(latency_seconds, 6),
            prompt_hash=prompt_hash,
            cache_hit=cache_hit,
            valid_evidence_ids=valid_ids,
            quality_flags=case.quality_flags,
            error=error,
            metadata={
                "retriever_id": selected_retriever.retriever_id,
                "retrieved_facts": (
                    _retrieved_metadata(bundle)
                    if mode != ExperimentMode.DIRECT
                    else []
                ),
                "directory_label": case.directory_label,
                "disease_category": case.disease_category,
                "generation_attempt_count": completion_attempt_count,
                "generation_context_window": completion_context_window,
                "generation_max_output_tokens": (
                    completion_max_output_tokens
                ),
                "generation_content": completion_audit,
                "patient_input_sha256": _text_sha256(render_case(case)),
                "ordered_retrieval_bundle_sha256": (
                    _method_bundle_sha256(bundle, self.argument_method)
                ),
                **(
                    {
                        "semantic_routing": routing,
                        "semantic_routing_sha256": _json_sha256(routing),
                        "semantic_routing_authorization_sha256": (
                            _json_sha256(
                                _semantic_routing_authorization(
                                    selected_retriever
                                )
                            )
                        ),
                    }
                    if (
                        mode != ExperimentMode.DIRECT
                        and (
                            routing := _semantic_routing_metadata(
                                selected_retriever,
                                case.case_id,
                            )
                        )
                    )
                    else {}
                ),
            },
        )

    def _run_version_iii_argument_team(
        self, case: ClinicalCase, bundle: RetrievalBundle, *, run_id: str,
        graph_incumbent: PredictionRecord | None = None,
        stage_callback: Callable[[str, str], None] | None = None,
    ) -> ArgumentTeamResult:
        """Ask one enriched differential question, then one adversarial question."""
        generator_model = self.reasoner_model
        verifier_model = self.verifier_model
        if stage_callback:
            stage_callback("reasoner", "started")
        completions: list[CachedCompletion] = []
        completion_roles: list[str] = []
        exact_findings = _exact_patient_findings(case)
        patient_rows = patient_evidence(case)
        patient_text = {eid: text for eid, _, text in patient_rows}
        patient_inventory = tuple(PatientEvidenceClaim(eid, section, text, _text_sha256(text))
                                  for eid, section, text in patient_rows)
        candidates = family_candidate_inventory_entries(
            case, bundle.facts, self.retriever.normalizer, bundle.family_routes
        ) or kg_candidate_inventory_entries(case, bundle.facts, self.retriever.normalizer)
        if not candidates:
            raise ValueError("Version III requires retrieved family candidates.")
        candidate_pool = tuple(candidates[:8])
        child_facts = atomize_retrieved_family_facts(bundle.family_child_facts)
        all_child_facts = tuple(dict.fromkeys((
            *(x.fact for x in bundle.family_child_facts), *(x.fact for x in child_facts),
        )))
        claims = knowledge_claims((*bundle.facts, *all_child_facts))
        claim_index = {x.evidence_id: x for x in claims}
        claim_alias_by_node: dict[str, tuple[str, ...]] = {}
        for claim in claims:
            aliases = claim_alias_by_node.setdefault(claim.node_id, ())
            if claim.evidence_id not in aliases:
                claim_alias_by_node[claim.node_id] = (*aliases, claim.evidence_id)
        shortlist = family_entailment_shortlist(
            exact_findings, child_facts, normalizer=self.retriever.normalizer,
            maximum_per_family=MAX_DIALOGUE_BINDING_OPTIONS,
        )
        shortlist_by_graph = {entry.graph_id: tuple(dict.fromkeys(
            x.fact.evidence_id for x in shortlist if x.graph_id == entry.graph_id
        )) for entry in candidate_pool}
        pool_records: list[dict[str, Any]] = []
        for entry in candidate_pool:
            knowledge_ids = tuple(dict.fromkeys((
                *shortlist_by_graph.get(entry.graph_id, ()),
                *(x for x in entry.evidence_ids if x in claim_index),
            )))
            for index, option in enumerate(_rank_dossier_binding_options(
                exact_findings, knowledge_ids, claims, self.retriever.normalizer
            ), 1):
                claim = claim_index[option["knowledge_evidence_id"]]
                authority = deterministic_binding_authority(
                    option["patient_finding"], claim.text,
                    candidate_label=entry.diagnosis, role=claim.role.value,
                    normalizer=self.retriever.normalizer,
                )
                source_anchor = qualifies_decisive_anchor(
                    claim.text, claim.role
                )
                pool_records.append({"pair_id": f"P-{entry.candidate_id}-{index}", **option,
                    "candidate_id": entry.candidate_id, "candidate": entry.diagnosis,
                    "graph_id": entry.graph_id, "knowledge_warrant": claim.text,
                    "knowledge_role": claim.role.value,
                    "diagnostic_path": " -> ".join(claim.diagnostic_path),
                    "child_label": claim.diagnosis_label,
                    # A graph criterion alone is not enough.  It becomes an
                    # established anchor only after the existing typed binding
                    # boundary confirms a current, completed patient result or
                    # exact threshold.  The model receives this as evidence
                    # priority, never as a resolver override.
                    "decisive_source_anchor": source_anchor,
                    "established_decisive_anchor": (
                        source_anchor and authority.authorized
                        and authority.authority_type in {
                            "exact_positive_test", "exact_measurement",
                            "explicit_current_diagnosis",
                        }
                    ),
                    "anchor_authorized": authority.authorized,
                    "anchor_authority_type": authority.authority_type,
                    # A risk factor can only raise plausibility
                    # (BindingApplication.RAISES_PLAUSIBILITY, never SATISFIES
                    # -- see canonicalize_positive_applications in
                    # argumentation.py), so it can never by itself be the
                    # citation that establishes a diagnosis, even when the
                    # typed-binding critical questions all pass.
                    "establishes_diagnosis": (
                        option["typed_binding_admissible"]
                        and claim.role != KnowledgeRole.RISK_FACTOR
                    )})
        pool_pair_owner = {
            x["pair_id"]: x["candidate_id"] for x in pool_records
        }
        selectable_pool = tuple(
            entry for entry in candidate_pool
            if any(x["candidate_id"] == entry.candidate_id for x in pool_records)
        )
        if not selectable_pool:
            raise ValueError(
                "Version III activation requires candidate-owned evidence pairs."
            )
        patient_sections = {evidence_id: section for evidence_id, section, _ in patient_rows}
        evidence_anchor_report = _evidence_anchor_report(
            selectable_pool, pool_records, patient_sections
        )

        def complete_json(*, role: str, version: str, model: DiagnosticModel,
                          system: str, state: Mapping[str, Any], schema: dict[str, object],
                          validator: Callable[[Mapping[str, Any]], object],
                          output_cap: int, retry_cap: int):
            completion = self._complete(role=role, prompt_version=version, model=model,
                system_prompt=system, user_prompt=json.dumps(state, ensure_ascii=False, sort_keys=True),
                output_schema=schema, response_validator=lambda value: validator(value),
                max_output_tokens_cap=output_cap,
                retry_output_tokens_cap=retry_cap)
            completions.append(completion)
            completion_roles.append(role)
            payload = _first_json_object(completion.response)
            return payload, validator(payload)

        activation_profiles = []
        activation_candidates = selectable_pool
        activation_limit = min(4, len(activation_candidates))
        activation_candidate_ids = {entry.candidate_id for entry in activation_candidates}
        if (
            graph_incumbent is not None
            and not graph_incumbent.abstained
            and not graph_incumbent.error
            and str(graph_incumbent.predicted_label).strip()
        ):
            incumbent_key = label_key(graph_incumbent.predicted_label)
            for entry in candidate_pool:
                if entry.candidate_id in activation_candidate_ids:
                    continue
                family_match = _graph_family_match(
                    entry,
                    incumbent_key,
                    (),
                )
                if not family_match:
                    continue
                if not any(
                    record["candidate_id"] == entry.candidate_id for record in pool_records
                ):
                    continue
                activation_candidates += (entry,)
                activation_candidate_ids.add(entry.candidate_id)
                activation_limit = min(4, len(activation_candidates))
                if len(activation_candidates) >= 4:
                    break

        for entry in activation_candidates:
            family_pairs = [
                {
                    "pair_id": record["pair_id"],
                    "patient_evidence_id": record["patient_evidence_id"],
                    "patient_finding": record["patient_finding"],
                    "knowledge_warrant": record["knowledge_warrant"],
                    "knowledge_role": record["knowledge_role"],
                    "diagnostic_path": record["diagnostic_path"],
                    "established_decisive_anchor": bool(
                        record.get("established_decisive_anchor")
                    ),
                }
                for record in pool_records
                if record["candidate_id"] == entry.candidate_id
            ][:2]
            activation_profiles.append({
                "candidate_id": entry.candidate_id,
                "candidate": entry.diagnosis,
                "retrieval_rank": entry.family_rank or entry.retrieval_rank,
                "evidence_pairs": family_pairs,
            })
        activation_pair_owner = {
            pair_id: owner for pair_id, owner in pool_pair_owner.items()
            if owner in {entry.candidate_id for entry in activation_candidates}
        }
        activation_state = {
                "clinical_question": (
                    "Which up to four supplied families most deserve direct "
                    "comparison for the current encounter?"
                ),
                "patient_findings": [
                    {"patient_evidence_id": eid, "section": sec, "finding": text}
                    for eid, sec, text in patient_rows
                ],
                "candidate_profiles": activation_profiles,
            }
        activation_payload, activation = complete_json(
            role="version_iii:evidence_aware_activation",
            version="clinical-argumentation-version-iii-evidence-aware-activation",
            model=generator_model,
            system=VERSION_III_ACTIVATION_SYSTEM_PROMPT,
            state=activation_state,
            schema=evidence_aware_activation_schema(
                (x.candidate_id for x in activation_candidates),
                activation_pair_owner, limit=activation_limit,
            ),
            validator=lambda value: parse_evidence_aware_activation(
                value,
                (x.candidate_id for x in activation_candidates),
                activation_pair_owner, limit=activation_limit, strict=False,
            ),
            output_cap=self.reasoner_output_token_cap,
            retry_cap=MAX_RETRY_OUTPUT_TOKENS,
        )
        entry_by_id = {entry.candidate_id: entry for entry in candidate_pool}
        admitted_candidate_ids = set(activation.candidate_ids)
        active_candidate_ids = tuple(
            entry.candidate_id for entry in selectable_pool
            if entry.candidate_id in admitted_candidate_ids
        )
        active = tuple(entry_by_id[value] for value in active_candidate_ids)
        active_ids = set(active_candidate_ids)
        records = [
            record for record in pool_records
            if record["candidate_id"] in active_ids
        ]
        pair_owner = {x["pair_id"]: x["candidate_id"] for x in records}
        all_pair_owner = {
            record["pair_id"]: record["candidate_id"]
            for record in pool_records
        }
        # Pairs the model may cite without any of them independently
        # establishing a diagnosis (e.g. risk factors, or findings that fail
        # the typed-binding critical questions). A decision can only remain
        # SUPPORTED if at least one decisive citation is in this set.
        establishing_pair_ids = frozenset(
            x["pair_id"] for x in records if x.get("establishes_diagnosis")
        )
        candidate_ids = tuple(x.candidate_id for x in active)
        diagnoses = {x.candidate_id: x.diagnosis for x in active}
        allowed_children = {entry.candidate_id: tuple(dict.fromkeys(
            x["child_label"] for x in records
            if x["candidate_id"] == entry.candidate_id and x["child_label"]
            and x["child_label"].casefold() != entry.diagnosis.casefold()
        )) for entry in active}

        activation_pair_ids = {
            selection.candidate_id: selection.evidence_pair_ids
            for selection in activation.selections
        }
        direct_schema = direct_differential_schema(
            candidate_ids, pair_owner, patient_text,
            (x for values in allowed_children.values() for x in values),
        )
        direct_payload, direct = complete_json(
            role="version_iii:direct_differential",
            version="clinical-argumentation-version-iii-direct-differential",
            model=generator_model,
            system=VERSION_III_DIRECT_SYSTEM_PROMPT,
            state={"clinical_question": "Which supplied diagnosis best explains the current encounter?",
                   "patient_findings": [{"patient_evidence_id": eid, "section": sec, "finding": text}
                                        for eid, sec, text in patient_rows],
                   "candidate_families": [asdict(x) for x in active],
                   "activation_selected_pair_ids": activation_pair_ids,
                   "enriched_immutable_pairs": records},
            schema=direct_schema,
            validator=lambda value: parse_direct_differential(
                value, candidate_ids, pair_owner, patient_text, allowed_children,
                establishing_pair_ids=establishing_pair_ids, strict=False,
            ),
            output_cap=self.reasoner_output_token_cap,
            retry_cap=MAX_RETRY_OUTPUT_TOKENS,
        )
        if direct.child_label and not any(
            pair_id in direct.decisive_pair_ids
            and record["candidate_id"] == direct.candidate_id
            and label_key(record["child_label"]) == label_key(direct.child_label)
            for pair_id, record in (
                (item["pair_id"], item) for item in records
            )
        ):
            direct = replace(direct, child_label="")
        protected_incumbent = None
        if (
            graph_incumbent is not None
            and not graph_incumbent.abstained
            and not graph_incumbent.error
            and graph_incumbent.predicted_label.strip()
        ):
            cited = set(_string_list(graph_incumbent.citations))
            for citation in tuple(cited):
                linked_claim = claim_index.get(citation)
                if linked_claim is None:
                    continue
                cited.update(claim_alias_by_node.get(linked_claim.node_id, (citation,)))
            incumbent_key = label_key(graph_incumbent.predicted_label)
            direct_fallback_candidates = {
                active_candidate_id
                for active_candidate_id in active_ids
                if active_candidate_id == direct.candidate_id
                and direct.decisive_pair_ids
            }
            all_records_by_candidate = {
                entry.candidate_id: tuple(
                    record for record in pool_records
                    if record["candidate_id"] == entry.candidate_id
                )
                for entry in candidate_pool
            }
            graph_shape_candidates = {
                entry.candidate_id for entry in candidate_pool
                if _graph_family_match_with_evidence_overlap(
                    entry,
                    incumbent_key,
                    all_records_by_candidate[entry.candidate_id],
                    cited,
                    allowed_children.get(entry.candidate_id, ()),
                )
            }
            eligible: list[tuple[int, int, int, CandidateInventoryEntry, tuple[str, ...]]] = []
            for entry in candidate_pool:
                used_fallback_pairs = False
                strict_pairs = tuple(
                    record["pair_id"] for record in all_records_by_candidate[entry.candidate_id]
                    if record["candidate_id"] == entry.candidate_id
                    and record["patient_evidence_id"] in cited
                    and record["knowledge_evidence_id"] in cited
                )
                knowledge_pairs = tuple(
                    record["pair_id"] for record in all_records_by_candidate[entry.candidate_id]
                    if record["candidate_id"] == entry.candidate_id
                    and record["knowledge_evidence_id"] in cited
                )
                owned_pairs = strict_pairs or knowledge_pairs
                label_match = int(_graph_family_match(
                    entry,
                    incumbent_key,
                    allowed_children.get(entry.candidate_id, ()),
                ))
                citation_match = _graph_family_match_with_evidence_overlap(
                    entry,
                    incumbent_key,
                    all_records_by_candidate[entry.candidate_id],
                    cited,
                    allowed_children.get(entry.candidate_id, ()),
                )
                if not owned_pairs:
                    if not label_match and not citation_match:
                        continue
                    if (
                        entry.candidate_id not in direct_fallback_candidates
                        and entry.candidate_id not in graph_shape_candidates
                    ):
                        continue
                    owned_pairs = tuple(
                        record["pair_id"] for record in all_records_by_candidate[entry.candidate_id]
                        if record["candidate_id"] == entry.candidate_id
                    )
                    if not owned_pairs:
                        continue
                    used_fallback_pairs = True
                # A Graph RAG label can be a concrete clinical observation
                # (for example, "right cerebellar infarct") rather than the
                # family name ("Stroke").  Its cited, candidate-owned
                # knowledge evidence is the stable family-level link in that
                # situation, so treat citation overlap as a family match.
                family_match = bool(label_match or citation_match)
                if not family_match and not used_fallback_pairs:
                    if (
                        entry.candidate_id in direct_fallback_candidates
                        and not graph_shape_candidates
                    ):
                        direct_pairs = tuple(
                            pair_id for pair_id in direct.decisive_pair_ids
                            if pair_id in owned_pairs
                        )
                        if direct_pairs:
                            owned_pairs = direct_pairs
                        else:
                            continue
                        used_fallback_pairs = True
                    else:
                        continue
                if not family_match and used_fallback_pairs:
                    evidence_mode = 0
                else:
                    evidence_mode = int(bool(strict_pairs)) if owned_pairs else 0
                eligible.append((label_match, evidence_mode, len(owned_pairs), entry, owned_pairs))
            if eligible:
                eligible.sort(
                    key=lambda item: (-item[0], -item[1], -item[2],
                                      item[3].family_rank or item[3].retrieval_rank)
                )
                best = eligible[0]
                if len(eligible) == 1 or best[:3] > eligible[1][:3] or best[0] == 1:
                    protected_incumbent = ProtectedIncumbent(
                        best[3].candidate_id,
                        best[3].diagnosis,
                        best[4][:4],
                    )
        if stage_callback:
            stage_callback("reasoner", "completed"); stage_callback("verifier", "started")
        verification_direct = direct
        if direct.decision != DirectDecision.SUPPORTED and protected_incumbent:
            verification_direct = replace(
                direct,
                candidate_id=protected_incumbent.candidate_id,
                child_label="",
                decisive_pair_ids=protected_incumbent.evidence_pair_ids,
            )
        attack_payload, attack = complete_json(
            role="version_iii:adversarial_verifier",
            version="clinical-argumentation-version-iii-adversarial-verifier",
            model=verifier_model, system=VERSION_III_VERIFY_SYSTEM_PROMPT,
            state={"clinical_question": "Should the proposed diagnosis survive?",
                   "patient_findings": [{"patient_evidence_id": eid, "section": sec, "finding": text}
                                        for eid, sec, text in patient_rows],
                   "candidate_families": [asdict(x) for x in active],
                   "enriched_immutable_pairs": records,
                   "proposed_answer": asdict(verification_direct)},
            schema=differential_attack_schema(candidate_ids, pair_owner),
            validator=lambda value: parse_differential_attack(
                value, candidate_ids, pair_owner, verification_direct.candidate_id,
                establishing_pair_ids=establishing_pair_ids, strict=False,
            ),
            output_cap=self.verifier_output_token_cap,
            retry_cap=CURRENT_VERIFIER_RETRY_OUTPUT_TOKEN_CAP,
        )
        attack_validation_payload: dict[str, Any] = {}
        attack_validation = None
        if attack.attack:
            cited_records = [
                record for record in records
                if record["pair_id"] in attack.evidence_pair_ids
            ]
            proposed_records = [
                record for record in records
                if record["pair_id"] in verification_direct.decisive_pair_ids
            ]
            attack_validation_payload, attack_validation = complete_json(
                role="version_iii:attack_validation",
                version="clinical-argumentation-version-iii-attack-validation",
                model=verifier_model,
                system=VERSION_III_ATTACK_VALIDATION_SYSTEM_PROMPT,
                state={
                    "proposed_answer": asdict(verification_direct),
                    "stated_attack": asdict(attack),
                    "patient_findings": [
                        {"patient_evidence_id": eid, "section": sec, "finding": text}
                        for eid, sec, text in patient_rows
                    ],
                    "proposed_decisive_pairs": proposed_records,
                    "cited_immutable_pairs": cited_records,
                },
                schema=attack_validation_schema(),
                validator=parse_attack_validation,
                output_cap=self.verifier_output_token_cap,
                retry_cap=CURRENT_VERIFIER_RETRY_OUTPUT_TOKEN_CAP,
            )
        if stage_callback:
            stage_callback("verifier", "completed")
        resolution = resolve_direct_differential(
            direct, attack, diagnoses, attack_validation, protected_incumbent
        )
        selected_id = resolution.selected_candidate_id
        selected_pairs = tuple(
            x for x in resolution.selected_pair_ids
            if all_pair_owner.get(x) == selected_id
        )
        if direct.child_label and resolution.action == ResolutionAction.MAINTAIN:
            selected_pairs = tuple(
                pair_id for pair_id in selected_pairs
                for pair_map_record in (next(
                    item for item in records if item["pair_id"] == pair_id
                ),)
                if label_key(pair_map_record["child_label"]) == label_key(direct.child_label)
            )
        grounded: list[ProposedArgument] = []
        pair_map = {x["pair_id"]: x for x in pool_records}
        selected_entry = (
            next((x for x in candidate_pool if x.candidate_id == selected_id), None)
            if selected_id else None
        )
        grounding_entry = (
            replace(
                selected_entry,
                diagnosis=resolution.selected_diagnosis,
                canonical_key=label_key(resolution.selected_diagnosis),
            )
            if selected_entry is not None
            and resolution.selected_diagnosis != selected_entry.diagnosis
            else selected_entry
        )
        for idx, pair_id in enumerate(selected_pairs, 1):
            record = pair_map[pair_id]
            argument = _ground_dialogue_binding(
                argument_id=f"A{idx}", candidate=grounding_entry,
                patient_finding=record["patient_finding"],
                patient_evidence_id=record["patient_evidence_id"],
                knowledge_evidence_id=record["knowledge_evidence_id"],
                patient_text=patient_text, claims=claims, normalizer=self.retriever.normalizer,
                semantic_admitted=True)
            if argument is not None:
                grounded.append(argument)
        # Protected Graph RAG fallback may intentionally retain a
        # candidate that evidence-aware activation did not select.  The
        # resolved candidate must still appear in the final proposal so the
        # argument graph and structural contract agree about its identity.
        proposal_candidates = active
        if selected_entry is not None and all(
            item.candidate_id != selected_entry.candidate_id
            for item in proposal_candidates
        ):
            proposal_candidates = (*proposal_candidates, selected_entry)
        proposal = ReasonerProposal(
            candidates=tuple(DiagnosisCandidate(x.candidate_id,
                resolution.selected_diagnosis if x.candidate_id == selected_id else x.diagnosis,
                tuple(grounded) if x.candidate_id == selected_id else ())
                for x in proposal_candidates),
            preferred_diagnosis=resolution.selected_diagnosis,
            abstain=not bool(selected_id), raw_argument_count=len(grounded),
            invalid_argument_count=0, canonicalized_argument_count=len(grounded))
        reviews = tuple(ArgumentReview(x.argument_id, ReviewVerdict.SUPPORTED, (),
            "Version III direct differential survived adversarial resolution.", x.evidence_ids)
            for x in grounded)
        verifier = VerifierReport(reviews, (), not bool(selected_id), len(reviews), 0, 0, 0)
        valid_ids = tuple(dict.fromkeys((*(x[0] for x in patient_rows), *bundle.evidence_ids)))
        graph = build_patient_argument_graph(case_id=case.case_id, proposal=proposal,
            verifier=verifier, valid_evidence_ids=valid_ids,
            knowledge_support={x.evidence_id: x.diagnostic_path for x in (*bundle.facts, *all_child_facts)},
            knowledge_provenance={x.evidence_id: {"node_id": x.node_id, "diagnostic_path": x.diagnostic_path,
                "premise_type": x.premise_type, "diagnosis_label": x.diagnosis_label, "text": x.text}
                for x in (*bundle.facts, *all_child_facts)}, knowledge_inventory=claims,
            patient_evidence_text=patient_text, patient_evidence_inventory=patient_inventory,
            normalizer=self.retriever.normalizer)
        selected_evidence = tuple(dict.fromkeys(v for pid in selected_pairs for v in
            (pair_map[pid]["patient_evidence_id"], pair_map[pid]["knowledge_evidence_id"])))
        raw_reasoner = {
            "evidence_aware_activation": {
                "model_selection": activation_payload,
                "active_candidate_ids": list(candidate_ids),
                "evidence_anchor_report": list(evidence_anchor_report),
            },
            "direct_differential": direct_payload,
            "protected_graph_incumbent": (
                asdict(protected_incumbent) if protected_incumbent else None
            ),
            "attack_validation": attack_validation_payload,
        }
        raw_verifier = {
            "adversarial_attack": attack_payload,
            "independent_attack_validation": attack_validation_payload,
        }
        combined = json.dumps({"reasoner": raw_reasoner, "verifier": raw_verifier},
                              sort_keys=True, separators=(",", ":")).encode()
        resolution_payload = {**resolution.to_dict(), "outcome": "accepted" if selected_id else "abstain",
            "selected_hypothesis_id": selected_id, "hypothesis_attempts_used": len(active),
            "hypothesis_attempt_limit": 4, "why": {"decision": resolution.action.value,
                                                     "reason": resolution.reason}}
        return ArgumentTeamResult(trace_id=f"{run_id}:{case.case_id}", case_id=case.case_id,
            proposal=proposal, verifier=verifier, graph=graph,
            reasoner_prompt_hash=_json_sha256([
                completion.prompt_hash
                for role, completion in zip(completion_roles, completions)
                if role in {"version_iii:evidence_aware_activation",
                            "version_iii:direct_differential"}
            ]),
            verifier_prompt_hash=_json_sha256([
                completion.prompt_hash
                for role, completion in zip(completion_roles, completions)
                if role in {"version_iii:adversarial_verifier",
                            "version_iii:attack_validation"}
            ]),
            reasoner_attempt_count=sum(
                completion.attempt_count
                for role, completion in zip(completion_roles, completions)
                if role in {"version_iii:evidence_aware_activation",
                            "version_iii:direct_differential"}
            ),
            verifier_attempt_count=sum(
                completion.attempt_count
                for role, completion in zip(completion_roles, completions)
                if role in {"version_iii:adversarial_verifier",
                            "version_iii:attack_validation"}
            ),
            latency_seconds=round(sum(x.latency_seconds for x in completions), 6),
            cache_hit=all(x.cache_hit for x in completions), error=None,
            argument_method=ARGUMENT_METHOD,
            candidate_inventory=candidates,
            retrieval_bundle_sha256=_method_bundle_sha256(
                bundle, ARGUMENT_METHOD),
            candidate_inventory_sha256=_candidate_inventory_sha256(candidates),
            reasoner_artifact_sha256=_json_sha256(raw_reasoner),
            critic_artifact_sha256=_json_sha256(raw_verifier),
            shared_argument_critic_artifact_sha256=hashlib.sha256(combined).hexdigest(),
            shared_argument_critic_artifact_bytes=len(combined), raw_reasoner_payload=raw_reasoner,
            raw_verifier_payload=raw_verifier, verifier_output_token_cap=self.verifier_output_token_cap,
            knowledge_source_provenance=tuple({"evidence_id": x.evidence_id, "node_id": x.node_id,
                "diagnostic_path": list(x.diagnostic_path), "source_chunk_id": x.source_chunk_id,
                "diagnosis_label": x.diagnosis_label, "knowledge_source_ids": list(x.knowledge_source_ids)}
                for x in (*bundle.facts, *all_child_facts)), citation_allowlist=_bundle_citation_allowlist(bundle),
            citation_allowlist_sha256=provenance_contract_sha256(list(_bundle_citation_allowlist(bundle))),
            entailment_shortlist=tuple({"graph_id": x.graph_id, "evidence_id": x.fact.evidence_id,
                "premise_type": x.fact.premise_type, "source_chunk_id": x.fact.source_chunk_id,
                "premise": x.fact.text, "child_identity_rendered": False} for x in shortlist),
            dialectical_trace={"dialogue_id": ARGUMENT_METHOD.prompt_version,
                "activation_policy_id": VERSION_III_ACTIVATION_ID,
                "candidate_pool_ids": [x.candidate_id for x in candidate_pool],
                "activation": {
                    **asdict(activation),
                },
                "evidence_anchor_report": list(evidence_anchor_report),
                "active_candidate_ids": list(candidate_ids), "direct_differential": asdict(direct),
                "protected_graph_incumbent": (
                    asdict(protected_incumbent) if protected_incumbent else None
                ),
                "attack": asdict(attack),
                "attack_validation": (
                    asdict(attack_validation) if attack_validation else None
                )}, dialectical_resolution=resolution_payload,
            selected_candidate_id=selected_id, selected_argument_ids=tuple(x.argument_id for x in grounded),
            selected_evidence_ids=selected_evidence,
            subtype_filter={"executed": False, "reason": "version_iii_direct_family_resolution"})

    def argument_team_error_result(
        self,
        case: ClinicalCase,
        bundle: RetrievalBundle,
        *,
        run_id: str,
        error: Exception,
    ) -> ArgumentTeamResult:
        """Materialize a case-local abstention after an argument-model failure.

        Retrieval and immutable evidence provenance remain available for audit;
        the failed model call receives no diagnostic authority and later cases
        can continue under the experiment's record-and-continue policy.
        """
        method = self.argument_method
        try:
            candidates = family_candidate_inventory_entries(
                case,
                bundle.facts,
                self.retriever.normalizer,
                bundle.family_routes,
            ) or kg_candidate_inventory_entries(
                case,
                bundle.facts,
                self.retriever.normalizer,
            )
        except Exception:
            candidates = ()
        retrieval_hash = _method_bundle_sha256(bundle, method)
        citation_allowlist = _bundle_citation_allowlist(bundle)
        failure = f"{type(error).__name__}: {error}"
        failure_artifact = {
            "stage": "argument_team",
            "policy_id": CURRENT_GENERATION_FAILURE_POLICY,
            "error": failure,
            "decision": "abstain",
        }
        artifact_bytes = _canonical_json_bytes(failure_artifact)
        resolver_id = method.resolver_id
        return ArgumentTeamResult(
            trace_id=f"{run_id}:{case.case_id}",
            case_id=case.case_id,
            proposal=None,
            verifier=None,
            graph=None,
            reasoner_prompt_hash="",
            verifier_prompt_hash="",
            reasoner_attempt_count=0,
            verifier_attempt_count=0,
            latency_seconds=0.0,
            cache_hit=False,
            error=failure,
            argument_method=method,
            candidate_inventory=tuple(candidates),
            retrieval_bundle_sha256=retrieval_hash,
            candidate_inventory_sha256=_candidate_inventory_sha256(candidates),
            reasoner_artifact_sha256=_json_sha256(failure_artifact),
            critic_artifact_sha256=_json_sha256({}),
            shared_argument_critic_artifact_sha256=hashlib.sha256(
                artifact_bytes
            ).hexdigest(),
            shared_argument_critic_artifact_bytes=len(artifact_bytes),
            raw_reasoner_payload={"generation_failure": failure_artifact},
            verifier_output_token_cap=self.verifier_output_token_cap,
            generation_recovery=failure_artifact,
            knowledge_source_provenance=tuple(
                {
                    "evidence_id": fact.evidence_id,
                    "node_id": fact.node_id,
                    "diagnostic_path": list(fact.diagnostic_path),
                    "source_chunk_id": fact.source_chunk_id,
                    "diagnosis_label": fact.diagnosis_label,
                    "knowledge_source_ids": list(fact.knowledge_source_ids),
                }
                for fact in bundle.facts
            ),
            citation_allowlist=citation_allowlist,
            citation_allowlist_sha256=provenance_contract_sha256(
                list(citation_allowlist)
            ),
            dialectical_trace={
                "dialogue_id": method.prompt_version,
                "active_candidate_ids": [],
                "generation_failure": failure_artifact,
            },
            dialectical_resolution={
                "resolver_id": resolver_id,
                "outcome": "abstain",
                "action": "abstain",
                "selected_candidate_id": "",
                "selected_diagnosis": "",
                "reason": "A case-local argument generation failure forced abstention.",
            },
            subtype_filter={"executed": False, "reason": "case_local_failure"},
        )

    def run_argument_team(
        self,
        case: ClinicalCase,
        bundle: RetrievalBundle,
        *,
        run_id: str,
        graph_incumbent: PredictionRecord | None = None,
        stage_callback: Callable[[str, str], None] | None = None,
    ) -> ArgumentTeamResult:
        """Run the supported evidence-grounded argumentation method."""
        self._assert_data_boundary(case)
        return self._run_version_iii_argument_team(
            case, bundle, run_id=run_id, graph_incumbent=graph_incumbent,
            stage_callback=stage_callback
        )
