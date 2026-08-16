"""Version III direct differential RAG with adversarial verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping


VERSION_III_RESOLVER_ID = "direct-differential-family-first-resolver-v1"
VERSION_III_ACTIVATION_ID = "evidence-aware-eight-to-four-activation-version-iii"

# vLLM compiles a guided-decoding grammar from each schema's maxItems bound.
# These caps must stay fixed regardless of how many candidate pairs/patient
# findings are supplied, or the compiled grammar grows with dossier size and
# can blow the completion-token/latency budget on a paid GPU job.
MAX_DECISIVE_PAIR_ITEMS = 4
MAX_UNEXPLAINED_EVIDENCE_ITEMS = 8
MAX_ATTACK_EVIDENCE_ITEMS = 8


class DirectDecision(StrEnum):
    SUPPORTED = "diagnosis_supported"
    INSUFFICIENT = "insufficient"
    NONE_OF_CANDIDATES = "none_of_supplied_candidates"


class AttackType(StrEnum):
    NONE = "none"
    CITATION_FAILURE = "citation_failure"
    WRONG_SUBJECT_OR_ENCOUNTER = "wrong_subject_or_encounter"
    DIAGNOSTIC_INSUFFICIENCY = "diagnostic_insufficiency"
    EXPLICIT_CONTRADICTION = "explicit_contradiction"
    BETTER_SUPPORTED_ALTERNATIVE = "better_supported_alternative"
    UNSUPPORTED_SPECIFICITY = "unsupported_specificity"
    MATERIAL_UNEXPLAINED_EVIDENCE = "material_unexplained_evidence"


class ResolutionAction(StrEnum):
    MAINTAIN = "maintain"
    SWITCH = "switch"
    FAMILY_FALLBACK = "family_fallback"
    PROTECTED_INCUMBENT = "protected_incumbent"
    ABSTAIN = "abstain"


class AttackScope(StrEnum):
    NONE = "none"
    CHILD_ONLY = "child_only"
    FAMILY = "family"


@dataclass(frozen=True)
class DirectDifferential:
    candidate_id: str
    child_label: str
    decision: DirectDecision
    decisive_pair_ids: tuple[str, ...]
    strongest_alternative_id: str
    alternative_pair_ids: tuple[str, ...]
    unexplained_patient_evidence_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class DifferentialAttack:
    attack: bool
    attack_type: AttackType
    target_candidate_id: str
    alternative_candidate_id: str
    evidence_pair_ids: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class AttackValidation:
    valid: bool
    citations_support_stated_attack: bool
    targets_proposed_diagnosis: bool
    not_merely_coexisting_or_alternative: bool
    scope: AttackScope
    explanation: str
    directly_falsifies_target: bool = False
    falsified_condition_is_necessary: bool = False
    independently_establishes_alternative: bool = False


@dataclass(frozen=True)
class VersionIIIResolution:
    resolver_id: str
    action: ResolutionAction
    selected_candidate_id: str
    selected_diagnosis: str
    reason: str
    selected_pair_ids: tuple[str, ...] = ()
    leading_candidate_ids: tuple[str, ...] = ()
    leading_diagnoses: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationSelection:
    candidate_id: str
    evidence_pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProtectedIncumbent:
    """A cited Graph RAG family answer eligible only as an abstention backstop."""

    candidate_id: str
    diagnosis: str
    evidence_pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceAwareActivation:
    policy_id: str
    selections: tuple[ActivationSelection, ...]
    rationale: str

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.selections)


def evidence_aware_activation_schema(
    candidate_ids: Iterable[str],
    pair_owner: Mapping[str, str],
    *,
    limit: int = 4,
) -> dict[str, object]:
    candidates = tuple(dict.fromkeys(candidate_ids))
    pairs = tuple(dict.fromkeys(pair_owner))
    count = min(limit, len(candidates))
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selections": {
                "type": "array",
                "minItems": 0,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate_id": {"type": "string", "enum": list(candidates)},
                        "evidence_pair_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 2,
                            "items": {"type": "string", "enum": list(pairs)},
                            "uniqueItems": True,
                        },
                    },
                    "required": ["candidate_id", "evidence_pair_ids"],
                },
            },
            "rationale": {"type": "string", "minLength": 0, "maxLength": 400},
        },
        "required": ["selections", "rationale"],
    }


def parse_evidence_aware_activation(
    payload: Mapping[str, Any],
    candidate_ids: Iterable[str],
    pair_owner: Mapping[str, str],
    *,
    limit: int = 4,
    strict: bool = True,
) -> EvidenceAwareActivation:
    candidates = tuple(dict.fromkeys(candidate_ids))
    candidate_set = set(candidates)
    if strict:
        if set(payload) != {"selections", "rationale"}:
            raise ValueError("Activation fields differ from the frozen schema.")
    rows = payload.get("selections", ()) if isinstance(payload, Mapping) else ()
    required_count = min(limit, len(candidates))
    if strict and (not isinstance(rows, list) or len(rows) != required_count):
        raise ValueError("Activation must fill the exact bounded agenda.")
    if required_count < 0:
        required_count = 0
    selections: list[ActivationSelection] = []
    seen: set[str] = set()
    owner_pairs_by_candidate: dict[str, tuple[str, ...]] = {
        candidate: tuple(
            pair_id for pair_id, owner in pair_owner.items()
            if owner == candidate
        )
        for candidate in candidates
    }

    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                if strict:
                    raise ValueError("Activation selection differs from the contract.")
                continue
            candidate_id = row.get("candidate_id")
            pair_ids = row.get("evidence_pair_ids")
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in candidate_set
            ):
                if strict:
                    raise ValueError("Activation requires valid candidate IDs.")
                continue
            if candidate_id in seen:
                if strict:
                    raise ValueError("Activation requires unique candidates.")
                continue
            clean_pairs = ()
            pairs_explicitly_provided = isinstance(pair_ids, list) and len(pair_ids) > 0
            if isinstance(pair_ids, list):
                clean_pairs = tuple(dict.fromkeys(
                    pair_id
                    for pair_id in pair_ids
                    if isinstance(pair_id, str) and pair_owner.get(pair_id) == candidate_id
                ))
            elif strict:
                raise ValueError("Activation requires candidate-owned evidence.")

            if not clean_pairs:
                # Missing evidence_pair_ids gets a recovery fallback to the
                # candidate's own top pairs; explicitly cited evidence that
                # is entirely cross-family/invalid must fail closed instead
                # of being silently replaced with different evidence.
                if strict and pairs_explicitly_provided:
                    raise ValueError("Activation requires candidate-owned evidence.")
                clean_pairs = owner_pairs_by_candidate.get(candidate_id, ())[:2]
                if strict and not clean_pairs:
                    raise ValueError(
                        "Activation requires candidate-owned evidence."
                    )

            seen.add(candidate_id)
            selections.append(ActivationSelection(candidate_id, clean_pairs[:2]))

    for candidate_id in candidates:
        if len(selections) >= required_count:
            break
        if candidate_id in seen:
            continue
        fallback_pairs = owner_pairs_by_candidate.get(candidate_id, ())
        if not fallback_pairs:
            continue
        seen.add(candidate_id)
        selections.append(ActivationSelection(candidate_id, fallback_pairs[:2]))

    if strict and len(selections) < required_count:
        raise ValueError("Activation requires enough candidates with evidence.")
    rationale = " ".join(
        str(payload.get("rationale", "")).split()
    ) if isinstance(payload, Mapping) else ""
    if not rationale:
        rationale = "No rationale was provided."
        if strict:
            raise ValueError("Activation requires a rationale.")
    return EvidenceAwareActivation(
        VERSION_III_ACTIVATION_ID, tuple(selections[:required_count]), rationale
    )


def _validate_payload_shape(
    payload: Mapping[str, Any],
    required: set[str],
    array_fields: Iterable[str],
    *,
    context: str,
) -> None:
    if set(payload) != required:
        raise ValueError(f"{context} fields differ from the frozen schema.")
    for field in array_fields:
        value = payload[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(f"{context} {field} must be an array of strings.")


def direct_differential_schema(
    candidate_ids: Iterable[str], pair_ids: Iterable[str],
    patient_evidence_ids: Iterable[str], child_labels: Iterable[str],
) -> dict[str, object]:
    candidates = tuple(dict.fromkeys(candidate_ids))
    pairs = tuple(dict.fromkeys(pair_ids))
    patients = tuple(dict.fromkeys(patient_evidence_ids))
    children = tuple(dict.fromkeys(("", *child_labels)))
    # Property order is generation order under guided decoding: fields are
    # filled left-to-right per the schema, autoregressively. "decision" used
    # to be the third field decoded, before any citation or rationale text
    # existed for the model to condition on -- it had to guess the verdict,
    # then pattern-complete a plausible-looking justification afterward.
    # Reasoning/evidence fields now precede the verdict fields so each
    # decision is generated after, and conditioned on, its own citations and
    # rationale rather than before them.
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "rationale": {"type": "string", "minLength": 0, "maxLength": 400},
            "candidate_id": {"type": "string", "enum": ["", *candidates]},
            "decisive_pair_ids": {"type": "array", "minItems": 0, "maxItems": min(MAX_DECISIVE_PAIR_ITEMS, len(pairs)), "items": {"type": "string", "enum": list(pairs)}, "uniqueItems": True},
            "decision": {"type": "string", "enum": [x.value for x in DirectDecision]},
            "child_label": {"type": "string", "enum": list(children)},
            "strongest_alternative_id": {"type": "string", "enum": ["", *candidates]},
            "alternative_pair_ids": {"type": "array", "minItems": 0, "maxItems": min(MAX_DECISIVE_PAIR_ITEMS, len(pairs)), "items": {"type": "string", "enum": list(pairs)}, "uniqueItems": True},
            "unexplained_patient_evidence_ids": {"type": "array", "minItems": 0, "maxItems": min(MAX_UNEXPLAINED_EVIDENCE_ITEMS, len(patients)), "items": {"type": "string", "enum": list(patients)}, "uniqueItems": True},
        },
        "required": ["rationale", "candidate_id", "decisive_pair_ids", "decision", "child_label", "strongest_alternative_id", "alternative_pair_ids", "unexplained_patient_evidence_ids"],
    }


def parse_direct_differential(payload: Mapping[str, Any], candidate_ids: Iterable[str],
                              pair_owner: Mapping[str, str], patient_ids: Iterable[str],
                              allowed_children: Mapping[str, Iterable[str]],
                              establishing_pair_ids: Iterable[str] | None = None,
                              strict: bool = True) -> DirectDifferential:
    if strict:
        _validate_payload_shape(
            payload,
            {
                "candidate_id", "child_label", "decision", "decisive_pair_ids",
                "strongest_alternative_id", "alternative_pair_ids",
                "unexplained_patient_evidence_ids", "rationale",
            },
            {
                "decisive_pair_ids", "alternative_pair_ids",
                "unexplained_patient_evidence_ids",
            },
            context="Direct differential",
        )
        if any(
            not isinstance(payload[field], str)
            for field in (
                "candidate_id", "child_label", "decision",
                "strongest_alternative_id", "rationale",
            )
        ):
            raise ValueError("Direct differential scalar fields must be strings.")
    if not isinstance(payload, Mapping):
        raise ValueError("Direct differential payload must be an object.")
    candidates = set(candidate_ids)
    patients = set(patient_ids)
    decision_raw = str(payload.get("decision", "")).strip() if isinstance(payload.get("decision"), str) else ""
    decision = DirectDecision.INSUFFICIENT
    if not strict:
        try:
            decision = DirectDecision(decision_raw)
        except ValueError:
            decision = DirectDecision.INSUFFICIENT
    else:
        decision = DirectDecision(decision_raw)

    candidate = str(payload.get("candidate_id", "")) if isinstance(payload.get("candidate_id"), str) else ""
    if candidate not in candidates:
        candidate = ""
    establishing = (
        frozenset(establishing_pair_ids) if establishing_pair_ids is not None else None
    )
    decisive = tuple(dict.fromkeys(x for x in payload.get("decisive_pair_ids", ()) if isinstance(x, str) and pair_owner.get(x) == candidate))
    # A citation set that contains only non-establishing pairs (e.g. risk
    # factors, or findings that fail the typed-binding critical questions)
    # cannot by itself support a diagnosis -- at least one cited pair must
    # independently establish it.
    if establishing is not None and decisive and establishing.isdisjoint(decisive):
        decisive = ()
    alternative = str(payload.get("strongest_alternative_id", "")) if isinstance(payload.get("strongest_alternative_id"), str) else ""
    alternative_pairs = tuple(dict.fromkeys(x for x in payload.get("alternative_pair_ids", ()) if isinstance(x, str) and pair_owner.get(x) == alternative))
    if establishing is not None and alternative_pairs and establishing.isdisjoint(alternative_pairs):
        alternative_pairs = ()
    unexplained = tuple(dict.fromkeys(x for x in payload.get("unexplained_patient_evidence_ids", ()) if x in patients))
    child = str(payload.get("child_label", "")) if isinstance(payload.get("child_label"), str) else ""
    if strict and (candidate not in candidates or (alternative and alternative not in candidates)):
        raise ValueError("Direct differential cites an unknown candidate.")
    if not strict and alternative and alternative not in candidates:
        alternative = ""
    # A model may declare ``none_of_supplied_candidates`` while naming an
    # allowed child of its evidence-cited strongest supplied alternative.
    # This is a malformed representation of a family-level answer, not
    # evidence for an out-of-set diagnosis. Recover only that contradiction,
    # retaining the parent family rather than promoting the child subtype.
    if (
        decision == DirectDecision.NONE_OF_CANDIDATES
        and alternative in candidates
        and alternative_pairs
        and child in set(allowed_children.get(alternative, ()))
    ):
        candidate = alternative
        decisive = alternative_pairs
        child = ""
        decision = DirectDecision.SUPPORTED
    if child not in set(allowed_children.get(candidate, ())):
        child = ""
    if decision == DirectDecision.SUPPORTED and (not candidate or not decisive):
        decision = DirectDecision.INSUFFICIENT
        candidate = ""
        child = ""
        decisive = ()
    if decision != DirectDecision.SUPPORTED and strict and (not candidate or not decisive):
        child = ""
        candidate, decisive = "", ()
    rationale = " ".join(str(payload.get("rationale", "")).split())
    if not rationale:
        if strict:
            raise ValueError("Direct differential requires a rationale.")
        rationale = "No rationale was provided."
    return DirectDifferential(candidate, child, decision, decisive, alternative,
                              alternative_pairs, unexplained, rationale)


def differential_attack_schema(candidate_ids: Iterable[str], pair_ids: Iterable[str]) -> dict[str, object]:
    candidates = tuple(dict.fromkeys(candidate_ids)); pairs = tuple(dict.fromkeys(pair_ids))
    # Same "reason before verdict" ordering as direct_differential_schema:
    # "attack" used to be decoded first, forcing the model to commit to a
    # true/false verdict before writing the explanation or citing any
    # evidence that would justify it. In production this produced payloads
    # where attack=false coexisted with a fully-formed attack_type/
    # alternative_candidate_id/evidence_pair_ids, as if the model had
    # constructed a real attack and then declined to flag it -- the boolean
    # was decided before the reasoning existed to inform it.
    return {"type": "object", "additionalProperties": False, "properties": {
        "target_candidate_id": {"type": "string", "enum": ["", *candidates]},
        "explanation": {"type": "string", "minLength": 0, "maxLength": 400},
        "attack_type": {"type": "string", "enum": [x.value for x in AttackType]},
        "alternative_candidate_id": {"type": "string", "enum": ["", *candidates]},
        "evidence_pair_ids": {"type": "array", "minItems": 0, "maxItems": min(MAX_ATTACK_EVIDENCE_ITEMS, len(pairs)), "items": {"type": "string", "enum": list(pairs)}, "uniqueItems": True},
        "attack": {"type": "boolean"},
    }, "required": ["target_candidate_id", "explanation", "attack_type", "alternative_candidate_id", "evidence_pair_ids", "attack"]}


def parse_differential_attack(payload: Mapping[str, Any], candidate_ids: Iterable[str],
                              pair_owner: Mapping[str, str], target: str,
                              establishing_pair_ids: Iterable[str] | None = None,
                              strict: bool = True) -> DifferentialAttack:
    if strict:
        _validate_payload_shape(
            payload,
            {
                "attack", "attack_type", "target_candidate_id",
                "alternative_candidate_id", "evidence_pair_ids", "explanation",
            },
            {"evidence_pair_ids"},
            context="Differential attack",
        )
        if not isinstance(payload["attack"], bool) or any(
            not isinstance(payload[field], str)
            for field in (
                "attack_type", "target_candidate_id", "alternative_candidate_id",
                "explanation",
            )
        ):
            raise ValueError("Differential attack scalar fields have invalid types.")
        candidates = set(candidate_ids)
        attack = payload["attack"] is True
        kind = AttackType(payload["attack_type"])
        alternative = payload["alternative_candidate_id"]
        cited = tuple(dict.fromkeys(x for x in payload["evidence_pair_ids"] if x in pair_owner))
    else:
        payload = payload if isinstance(payload, Mapping) else {}
        candidates = set(candidate_ids)
        attack = bool(payload.get("attack", False)) if isinstance(payload.get("attack"), bool) else False
        try:
            kind = AttackType(payload.get("attack_type", AttackType.NONE.value))
        except ValueError:
            kind = AttackType.NONE
            attack = False
        alternative = payload.get("alternative_candidate_id", "")
        if not isinstance(alternative, str):
            alternative = ""
        cited = tuple(dict.fromkeys(
            item for item in payload.get("evidence_pair_ids", ())
            if isinstance(item, str) and item in pair_owner
        ))
    if not attack:
        kind, alternative, cited = AttackType.NONE, "", ()
    elif (
        kind == AttackType.NONE
        or str(payload.get("target_candidate_id", "")) != target
        or not cited
    ):
        attack, kind, alternative, cited = False, AttackType.NONE, "", ()
    if alternative and alternative not in candidates:
        alternative = ""
    if kind == AttackType.BETTER_SUPPORTED_ALTERNATIVE:
        alt_citations = tuple(x for x in cited if pair_owner.get(x) == alternative)
        target_citations = tuple(x for x in cited if pair_owner.get(x) == target)
        # The alternative must be independently established, not merely
        # cited -- a risk-factor-only or non-admissible citation set cannot
        # be the sole grounds for a switch.
        establishing = (
            frozenset(establishing_pair_ids) if establishing_pair_ids is not None else None
        )
        alt_establishes = establishing is None or not establishing.isdisjoint(alt_citations)
        if not alternative or not alt_citations or not target_citations or not alt_establishes:
            attack, kind, alternative, cited = False, AttackType.NONE, "", ()
    elif attack and not any(pair_owner.get(x) == target for x in cited):
        attack, kind, alternative, cited = False, AttackType.NONE, "", ()
    explanation = " ".join(str(payload.get("explanation", "")).split())
    if not explanation:
        if strict:
            raise ValueError("Differential verification requires an explanation.")
        explanation = "No explanation was provided."
    return DifferentialAttack(attack, kind, target if attack else "", alternative, cited, explanation)


def attack_validation_schema() -> dict[str, object]:
    # Same "reason before verdict" ordering as the two schemas above: this
    # stage only starts running once the attack schema above actually
    # produces attack=true, at which point its six critical-question
    # booleans would hit the identical premature-commitment problem if left
    # in their original order (booleans first, explanation last).
    return {"type": "object", "additionalProperties": False, "properties": {
        "explanation": {"type": "string", "minLength": 1, "maxLength": 400},
        "scope": {"type": "string", "enum": [x.value for x in AttackScope]},
        "citations_support_stated_attack": {"type": "boolean"},
        "targets_proposed_diagnosis": {"type": "boolean"},
        "not_merely_coexisting_or_alternative": {"type": "boolean"},
        "directly_falsifies_target": {"type": "boolean"},
        "falsified_condition_is_necessary": {"type": "boolean"},
        "independently_establishes_alternative": {"type": "boolean"},
    }, "required": ["explanation", "scope", "citations_support_stated_attack",
        "targets_proposed_diagnosis", "not_merely_coexisting_or_alternative",
        "directly_falsifies_target", "falsified_condition_is_necessary",
        "independently_establishes_alternative"]}


def parse_attack_validation(payload: Mapping[str, Any]) -> AttackValidation:
    required = {"citations_support_stated_attack", "targets_proposed_diagnosis",
        "not_merely_coexisting_or_alternative", "directly_falsifies_target",
        "falsified_condition_is_necessary", "independently_establishes_alternative",
        "scope", "explanation"}
    if set(payload) != required or any(
        not isinstance(payload[key], bool) for key in required if key not in {"scope", "explanation"}
    ) or not isinstance(payload["scope"], str) or not isinstance(payload["explanation"], str):
        raise ValueError("Attack validation differs from the frozen schema.")
    scope = AttackScope(payload["scope"])
    explanation = " ".join(payload["explanation"].split())
    valid = bool(payload["citations_support_stated_attack"]
        and payload["targets_proposed_diagnosis"]
        and payload["not_merely_coexisting_or_alternative"]
        and payload["directly_falsifies_target"]
        and payload["falsified_condition_is_necessary"]
        and scope != AttackScope.NONE and explanation)
    return AttackValidation(
        valid=valid,
        citations_support_stated_attack=payload["citations_support_stated_attack"],
        targets_proposed_diagnosis=payload["targets_proposed_diagnosis"],
        not_merely_coexisting_or_alternative=payload["not_merely_coexisting_or_alternative"],
        scope=scope,
        explanation=explanation,
        directly_falsifies_target=payload["directly_falsifies_target"],
        falsified_condition_is_necessary=payload["falsified_condition_is_necessary"],
        independently_establishes_alternative=payload["independently_establishes_alternative"],
    )


def resolve_direct_differential(proposal: DirectDifferential, attack: DifferentialAttack,
                                diagnoses: Mapping[str, str],
                                validation: AttackValidation | None = None,
                                protected_incumbent: ProtectedIncumbent | None = None,
                                ) -> VersionIIIResolution:
    leading_ids = tuple(dict.fromkeys(x for x in (
        proposal.candidate_id, proposal.strongest_alternative_id
    ) if x and (x == proposal.candidate_id and proposal.decisive_pair_ids
                or x == proposal.strongest_alternative_id and proposal.alternative_pair_ids)))
    leading = tuple(diagnoses[x] for x in leading_ids)
    if proposal.decision != DirectDecision.SUPPORTED:
        if protected_incumbent is not None and (
            not attack.attack or validation is None or not validation.valid
        ):
            return VersionIIIResolution(
                VERSION_III_RESOLVER_ID,
                ResolutionAction.PROTECTED_INCUMBENT,
                protected_incumbent.candidate_id,
                protected_incumbent.diagnosis,
                "The direct comparison was inconclusive, so the cited Graph RAG "
                "incumbent was retained at family level after surviving verification.",
                protected_incumbent.evidence_pair_ids,
                leading_ids,
                leading,
            )
        if (
            protected_incumbent is not None
            and validation is not None
            and validation.valid
            and attack.attack_type == AttackType.BETTER_SUPPORTED_ALTERNATIVE
            and attack.alternative_candidate_id
            and validation.independently_establishes_alternative
        ):
            alternative = attack.alternative_candidate_id
            alternative_pairs = tuple(
                pair_id for pair_id in attack.evidence_pair_ids
                if pair_id not in protected_incumbent.evidence_pair_ids
            )
            return VersionIIIResolution(
                VERSION_III_RESOLVER_ID, ResolutionAction.SWITCH,
                alternative, diagnoses[alternative],
                "The protected incumbent was directly defeated and the cited "
                "alternative was independently established.",
                alternative_pairs,
                leading_ids, leading,
            )
        return VersionIIIResolution(VERSION_III_RESOLVER_ID, ResolutionAction.ABSTAIN, "", "", "No single supplied family was sufficiently established; leading evidence-cited possibilities are retained without a definitive diagnosis.", (), leading_ids, leading)
    if (attack.attack and validation is not None and not validation.valid
            and attack.attack_type == AttackType.BETTER_SUPPORTED_ALTERNATIVE
            and attack.alternative_candidate_id
            and validation.independently_establishes_alternative):
        competing_ids = tuple(dict.fromkeys(
            (proposal.candidate_id, attack.alternative_candidate_id)
        ))
        return VersionIIIResolution(
            VERSION_III_RESOLVER_ID,
            ResolutionAction.ABSTAIN,
            "",
            "",
            "The alternative is independently supported but does not refute the proposed diagnosis; both are retained without a definitive diagnosis.",
            (),
            competing_ids,
            tuple(diagnoses[x] for x in competing_ids),
        )
    if not attack.attack or validation is None or not validation.valid:
        diagnosis = proposal.child_label or diagnoses[proposal.candidate_id]
        action = ResolutionAction.MAINTAIN if proposal.child_label else ResolutionAction.FAMILY_FALLBACK
        return VersionIIIResolution(VERSION_III_RESOLVER_ID, action, proposal.candidate_id, diagnosis, "The cited direct diagnosis survived independent adversarial verification.", proposal.decisive_pair_ids)
    if attack.attack_type == AttackType.UNSUPPORTED_SPECIFICITY or validation.scope == AttackScope.CHILD_ONLY:
        return VersionIIIResolution(VERSION_III_RESOLVER_ID, ResolutionAction.FAMILY_FALLBACK, proposal.candidate_id, diagnoses[proposal.candidate_id], "The family survived, but the proposed child diagnosis did not.", proposal.decisive_pair_ids)
    if attack.attack_type == AttackType.BETTER_SUPPORTED_ALTERNATIVE and attack.alternative_candidate_id:
        alt = attack.alternative_candidate_id
        return VersionIIIResolution(VERSION_III_RESOLVER_ID, ResolutionAction.SWITCH, alt, diagnoses[alt], "A cited alternative survived a grounded better-explanation attack.", tuple(x for x in attack.evidence_pair_ids if x not in proposal.decisive_pair_ids))
    return VersionIIIResolution(VERSION_III_RESOLVER_ID, ResolutionAction.ABSTAIN, "", "", "A grounded attack defeated the proposed diagnosis without establishing a safe replacement.")
