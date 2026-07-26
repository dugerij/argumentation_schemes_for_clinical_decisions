from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import combinations
from typing import Any, Iterable, Mapping

from clinical_cds.direct import label_key, normalize_label
from clinical_cds.schema import ClinicalCase


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


SUPPORT_SCHEMES = (
    ArgumentScheme.CLINICAL_SIGN,
    ArgumentScheme.DIAGNOSTIC_CRITERION,
    ArgumentScheme.RISK_FACTOR,
    ArgumentScheme.GUIDELINE_AUTHORITY,
)
COUNTER_SCHEMES = (
    ArgumentScheme.NEGATIVE_EVIDENCE,
    ArgumentScheme.ALTERNATIVE_EXPLANATION,
)
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
MAX_CANDIDATES = 3
MAX_ARGUMENTS_PER_CANDIDATE = 2
MAX_REVIEWS = MAX_CANDIDATES * MAX_ARGUMENTS_PER_CANDIDATE
MAX_COUNTERARGUMENTS = 3


@dataclass(frozen=True)
class CriticalQuestion:
    question_id: str
    text: str


COMMON_CRITICAL_QUESTIONS = (
    CriticalQuestion(
        "observation_present",
        "Is the cited patient observation explicitly present in the submitted state?",
    ),
    CriticalQuestion(
        "contradictory_evidence",
        "Is there contradictory or expected-but-absent evidence that weakens the claim?",
    ),
    CriticalQuestion(
        "alternative_explanation",
        "Could a supported alternative diagnosis explain the same observations better?",
    ),
)
SCHEME_CRITICAL_QUESTIONS = {
    ArgumentScheme.CLINICAL_SIGN: COMMON_CRITICAL_QUESTIONS
    + (
        CriticalQuestion(
            "evidence_specificity",
            "Is the finding sufficiently specific to support this diagnosis?",
        ),
    ),
    ArgumentScheme.DIAGNOSTIC_CRITERION: COMMON_CRITICAL_QUESTIONS
    + (
        CriticalQuestion(
            "criterion_satisfied",
            "Does the observed result satisfy the criterion rather than merely mention it?",
        ),
    ),
    ArgumentScheme.RISK_FACTOR: COMMON_CRITICAL_QUESTIONS
    + (
        CriticalQuestion(
            "risk_not_proof",
            "Is the risk factor used only to adjust plausibility rather than as proof?",
        ),
    ),
    ArgumentScheme.GUIDELINE_AUTHORITY: COMMON_CRITICAL_QUESTIONS
    + (
        CriticalQuestion(
            "guideline_applicability",
            "Is the retrieved guideline premise applicable to this patient and conclusion?",
        ),
    ),
}
CRITICAL_QUESTION_IDS = tuple(
    sorted(
        {
            question.question_id
            for questions in SCHEME_CRITICAL_QUESTIONS.values()
            for question in questions
        }
    )
)


@dataclass(frozen=True)
class ProposedArgument:
    argument_id: str
    scheme: ArgumentScheme
    premise: str
    conclusion: str
    evidence_ids: tuple[str, ...]


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArgumentReview:
    argument_id: str
    verdict: ReviewVerdict
    failed_critical_questions: tuple[str, ...]
    explanation: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CounterArgument:
    argument_id: str
    target_argument_id: str
    scheme: ArgumentScheme
    premise: str
    conclusion: str
    evidence_ids: tuple[str, ...]
    relation: RelationType


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


@dataclass(frozen=True)
class PatientArgumentGraph:
    graph_id: str
    nodes: tuple[ArgumentNode, ...]
    relations: tuple[ArgumentRelation, ...]
    quality: ArgumentGraphQuality

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SymbolicResolution:
    resolver_id: str
    selected_diagnosis: str
    abstained: bool
    accepted_argument_ids: tuple[str, ...]
    rejected_argument_ids: tuple[str, ...]
    undecided_argument_ids: tuple[str, ...]
    candidate_scores: dict[str, int]
    trace: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REASONER_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_CANDIDATES,
            "items": {
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string", "maxLength": 200},
                    "arguments": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_ARGUMENTS_PER_CANDIDATE,
                        "items": {
                            "type": "object",
                            "properties": {
                                "scheme": {
                                    "type": "string",
                                    "enum": [scheme.value for scheme in SUPPORT_SCHEMES],
                                },
                                "premise": {"type": "string", "maxLength": 400},
                                "evidence_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 4,
                                    "items": {"type": "string", "maxLength": 50},
                                },
                            },
                            "required": ["scheme", "premise", "evidence_ids"],
                        },
                    },
                },
                "required": ["diagnosis", "arguments"],
            },
        },
        "preferred_diagnosis": {"type": "string", "maxLength": 200},
        "abstain": {"type": "boolean"},
    },
    "required": ["candidates", "preferred_diagnosis", "abstain"],
}


VERIFIER_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "maxItems": MAX_REVIEWS,
            "items": {
                "type": "object",
                "properties": {
                    "argument_id": {"type": "string", "maxLength": 20},
                    "verdict": {
                        "type": "string",
                        "enum": [verdict.value for verdict in ReviewVerdict],
                    },
                    "failed_critical_questions": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {
                            "type": "string",
                            "enum": list(CRITICAL_QUESTION_IDS),
                        },
                    },
                    "explanation": {"type": "string", "maxLength": 160},
                    "evidence_ids": {
                        "type": "array",
                        "maxItems": 4,
                        "items": {"type": "string", "maxLength": 50},
                    },
                },
                "required": [
                    "argument_id",
                    "verdict",
                    "failed_critical_questions",
                    "explanation",
                    "evidence_ids",
                ],
            },
        },
        "counterarguments": {
            "type": "array",
            "maxItems": MAX_COUNTERARGUMENTS,
            "items": {
                "type": "object",
                "properties": {
                    "target_argument_id": {"type": "string", "maxLength": 20},
                    "scheme": {
                        "type": "string",
                        "enum": [scheme.value for scheme in COUNTER_SCHEMES],
                    },
                    "premise": {"type": "string", "maxLength": 400},
                    "conclusion": {"type": "string", "maxLength": 200},
                    "evidence_ids": {
                        "type": "array",
                        "maxItems": 4,
                        "items": {"type": "string", "maxLength": 50},
                    },
                    "relation": {
                        "type": "string",
                        "enum": [
                            RelationType.REBUTS.value,
                            RelationType.UNDERCUTS.value,
                        ],
                    },
                },
                "required": [
                    "target_argument_id",
                    "scheme",
                    "premise",
                    "conclusion",
                    "evidence_ids",
                    "relation",
                ],
            },
        },
        "abstain": {"type": "boolean"},
    },
    "required": ["reviews", "counterarguments", "abstain"],
}


def verifier_output_schema(proposal: ReasonerProposal) -> dict[str, object]:
    schema: dict[str, Any] = deepcopy(VERIFIER_OUTPUT_SCHEMA)
    argument_ids = [
        argument.argument_id
        for candidate in proposal.candidates
        for argument in candidate.arguments
    ]
    candidate_ids = [
        candidate.candidate_id for candidate in proposal.candidates
    ]
    reviews = schema["properties"]["reviews"]
    reviews["minItems"] = len(argument_ids)
    reviews["maxItems"] = len(argument_ids)
    reviews["items"]["properties"]["argument_id"] = {
        "type": "string",
        "enum": argument_ids,
    }
    counterarguments = schema["properties"]["counterarguments"]
    counterarguments["items"]["properties"]["target_argument_id"] = {
        "type": "string",
        "enum": candidate_ids + argument_ids,
    }
    return schema


def _text(value: object) -> str:
    return normalize_label(str(value or ""))


def _string_list(value: object, *, uppercase: bool = False) -> tuple[str, ...]:
    values = value if isinstance(value, list) else [value] if value is not None else []
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item).strip("[](),.;")
        if uppercase:
            text = text.upper()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return tuple(output)


def _candidate_label(value: object, case: ClinicalCase) -> str:
    text = _text(value)
    option_key = text.upper().strip(" .():")
    if case.options and option_key in case.options:
        return case.options[option_key]
    for option in case.options.values():
        if label_key(option) == label_key(text):
            return option
    return text


def parse_reasoner_proposal(
    payload: dict[str, Any],
    case: ClinicalCase,
) -> ReasonerProposal:
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    candidates: list[DiagnosisCandidate] = []
    seen_diagnoses: set[str] = set()
    raw_argument_count = 0
    invalid_argument_count = 0
    next_argument_id = 1
    for raw_candidate in raw_candidates[:MAX_CANDIDATES]:
        if not isinstance(raw_candidate, dict):
            continue
        diagnosis = _candidate_label(raw_candidate.get("diagnosis"), case)
        diagnosis_key = label_key(diagnosis)
        if not diagnosis or diagnosis_key in seen_diagnoses:
            continue
        seen_diagnoses.add(diagnosis_key)

        raw_arguments = raw_candidate.get("arguments")
        if not isinstance(raw_arguments, list):
            raw_arguments = []
        arguments: list[ProposedArgument] = []
        retained_arguments = raw_arguments[:MAX_ARGUMENTS_PER_CANDIDATE]
        raw_argument_count += len(raw_arguments)
        invalid_argument_count += len(raw_arguments) - len(retained_arguments)
        for raw_argument in retained_arguments:
            if not isinstance(raw_argument, dict):
                invalid_argument_count += 1
                continue
            try:
                scheme = ArgumentScheme(_text(raw_argument.get("scheme")))
            except ValueError:
                invalid_argument_count += 1
                continue
            premise = _text(raw_argument.get("premise"))
            evidence_ids = _string_list(
                raw_argument.get("evidence_ids"),
                uppercase=True,
            )
            if scheme not in SUPPORT_SCHEMES or not premise:
                invalid_argument_count += 1
                continue
            arguments.append(
                ProposedArgument(
                    argument_id=f"A{next_argument_id}",
                    scheme=scheme,
                    premise=premise,
                    conclusion=diagnosis,
                    evidence_ids=evidence_ids,
                )
            )
            next_argument_id += 1
        candidates.append(
            DiagnosisCandidate(
                candidate_id=f"D{len(candidates) + 1}",
                diagnosis=diagnosis,
                arguments=tuple(arguments),
            )
        )

    preferred = _candidate_label(payload.get("preferred_diagnosis"), case)
    preferred_match = next(
        (
            candidate.diagnosis
            for candidate in candidates
            if label_key(candidate.diagnosis) == label_key(preferred)
        ),
        "",
    )
    return ReasonerProposal(
        candidates=tuple(candidates),
        preferred_diagnosis=preferred_match,
        abstain=bool(payload.get("abstain")) or not preferred_match,
        raw_argument_count=raw_argument_count,
        invalid_argument_count=invalid_argument_count,
    )


def parse_verifier_report(
    payload: dict[str, Any],
    proposal: ReasonerProposal,
) -> VerifierReport:
    valid_targets = {
        candidate.candidate_id
        for candidate in proposal.candidates
    } | {
        argument.argument_id
        for candidate in proposal.candidates
        for argument in candidate.arguments
    }
    argument_ids = {
        argument.argument_id
        for candidate in proposal.candidates
        for argument in candidate.arguments
    }

    raw_reviews = payload.get("reviews")
    if not isinstance(raw_reviews, list):
        raw_reviews = []
    reviews: list[ArgumentReview] = []
    reviewed: set[str] = set()
    retained_reviews = raw_reviews[:MAX_REVIEWS]
    invalid_review_count = len(raw_reviews) - len(retained_reviews)
    for raw_review in retained_reviews:
        if not isinstance(raw_review, dict):
            invalid_review_count += 1
            continue
        argument_id = _text(raw_review.get("argument_id")).upper()
        if argument_id not in argument_ids or argument_id in reviewed:
            invalid_review_count += 1
            continue
        try:
            verdict = ReviewVerdict(_text(raw_review.get("verdict")))
        except ValueError:
            invalid_review_count += 1
            continue
        failed_questions = tuple(
            question_id
            for question_id in _string_list(
                raw_review.get("failed_critical_questions")
            )
            if question_id in CRITICAL_QUESTION_IDS
        )
        reviews.append(
            ArgumentReview(
                argument_id=argument_id,
                verdict=verdict,
                failed_critical_questions=failed_questions,
                explanation=_text(raw_review.get("explanation")),
                evidence_ids=_string_list(
                    raw_review.get("evidence_ids"),
                    uppercase=True,
                ),
            )
        )
        reviewed.add(argument_id)

    raw_counters = payload.get("counterarguments")
    if not isinstance(raw_counters, list):
        raw_counters = []
    counterarguments: list[CounterArgument] = []
    retained_counters = raw_counters[:MAX_COUNTERARGUMENTS]
    invalid_counterargument_count = len(raw_counters) - len(retained_counters)
    for raw_counter in retained_counters:
        if not isinstance(raw_counter, dict):
            invalid_counterargument_count += 1
            continue
        target_id = _text(raw_counter.get("target_argument_id")).upper()
        try:
            scheme = ArgumentScheme(_text(raw_counter.get("scheme")))
            relation = RelationType(_text(raw_counter.get("relation")))
        except ValueError:
            invalid_counterargument_count += 1
            continue
        premise = _text(raw_counter.get("premise"))
        if (
            target_id not in valid_targets
            or scheme not in COUNTER_SCHEMES
            or relation not in {RelationType.REBUTS, RelationType.UNDERCUTS}
            or not premise
        ):
            invalid_counterargument_count += 1
            continue
        counterarguments.append(
            CounterArgument(
                argument_id=f"C{len(counterarguments) + 1}",
                target_argument_id=target_id,
                scheme=scheme,
                premise=premise,
                conclusion=_text(raw_counter.get("conclusion")),
                evidence_ids=_string_list(
                    raw_counter.get("evidence_ids"),
                    uppercase=True,
                ),
                relation=relation,
            )
        )

    return VerifierReport(
        reviews=tuple(reviews),
        counterarguments=tuple(counterarguments),
        abstain=bool(payload.get("abstain")),
        raw_review_count=len(raw_reviews),
        invalid_review_count=invalid_review_count,
        raw_counterargument_count=len(raw_counters),
        invalid_counterargument_count=invalid_counterargument_count,
    )


def critical_questions_for(scheme: ArgumentScheme) -> tuple[CriticalQuestion, ...]:
    return SCHEME_CRITICAL_QUESTIONS.get(scheme, COMMON_CRITICAL_QUESTIONS)


def render_proposal_for_verification(proposal: ReasonerProposal) -> str:
    lines: list[str] = []
    for candidate in proposal.candidates:
        lines.append(f"{candidate.candidate_id} | diagnosis={candidate.diagnosis}")
        for argument in candidate.arguments:
            lines.append(
                (
                    f"{argument.argument_id} | scheme={argument.scheme.value} | "
                    f"premise={argument.premise} | "
                    f"evidence={','.join(argument.evidence_ids)} | "
                    f"conclusion={argument.conclusion}"
                )
            )
            lines.extend(
                f"{argument.argument_id}.{question.question_id}: {question.text}"
                for question in critical_questions_for(argument.scheme)
            )
    return "\n".join(lines) or "No valid candidate arguments were produced."


def _argument_has_required_evidence(
    argument: ProposedArgument,
    valid_ids: set[str],
    knowledge_support: Mapping[str, set[str]],
) -> bool:
    evidence_ids = set(argument.evidence_ids)
    if not evidence_ids or not evidence_ids.issubset(valid_ids):
        return False
    has_patient_evidence = any(value.startswith("S-") for value in evidence_ids)
    knowledge_evidence = {
        value for value in evidence_ids if value.startswith("K")
    }
    conclusion_key = label_key(argument.conclusion)
    has_aligned_knowledge = bool(knowledge_evidence) and all(
        conclusion_key in knowledge_support.get(evidence_id, set())
        for evidence_id in knowledge_evidence
    )
    if argument.scheme == ArgumentScheme.GUIDELINE_AUTHORITY:
        return has_aligned_knowledge
    return has_patient_evidence and has_aligned_knowledge


def build_patient_argument_graph(
    *,
    case_id: str,
    proposal: ReasonerProposal,
    verifier: VerifierReport,
    valid_evidence_ids: Iterable[str],
    knowledge_support: Mapping[str, Iterable[str]],
) -> PatientArgumentGraph:
    valid_ids = {value.upper() for value in valid_evidence_ids}
    normalized_knowledge_support = {
        evidence_id.upper(): {
            label_key(label)
            for label in labels
            if label_key(label)
        }
        for evidence_id, labels in knowledge_support.items()
    }
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
    reference_count = sum(len(argument.evidence_ids) for argument in proposed_arguments)
    valid_reference_count = sum(
        evidence_id in valid_ids
        for argument in proposed_arguments
        for evidence_id in argument.evidence_ids
    )

    for candidate in proposal.candidates:
        for argument in candidate.arguments:
            evidence_valid = _argument_has_required_evidence(
                argument,
                valid_ids,
                normalized_knowledge_support,
            )
            valid_argument_count += int(evidence_valid)
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
                )
            )

            review = reviews.get(argument.argument_id)
            challenge_reasons: list[str] = []
            challenge_evidence: tuple[str, ...] = ()
            if not evidence_valid:
                challenge_reasons.append(
                    "The argument does not contain valid patient evidence and "
                    "diagnosis-aligned guideline evidence."
                )
            if review is None:
                challenge_reasons.append("The verifier did not review this argument.")
            else:
                challenge_evidence = review.evidence_ids
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
                source="argument_graph_builder",
                priority=SCHEME_PRIORITY[ArgumentScheme.BEST_EXPLANATION],
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

    for counterargument in verifier.counterarguments:
        target = next(
            (node for node in nodes if node.argument_id == counterargument.target_argument_id),
            None,
        )
        candidate_id = target.candidate_id if target is not None else None
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
            )
        )
        relations.append(
            ArgumentRelation(
                source_id=counterargument.argument_id,
                target_id=counterargument.target_argument_id,
                relation=counterargument.relation,
            )
        )
        counter_evidence_valid = (
            bool(counterargument.evidence_ids)
            and set(counterargument.evidence_ids).issubset(valid_ids)
        )
        if not counter_evidence_valid:
            challenge_id = f"Q-{counterargument.argument_id}"
            add_node(
                ArgumentNode(
                    argument_id=challenge_id,
                    node_type="challenge",
                    scheme=ArgumentScheme.CRITICAL_QUESTION,
                    premise="The counterargument cites no valid supplied evidence.",
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
    reviewed_count = len({review.argument_id for review in verifier.reviews})
    quality = ArgumentGraphQuality(
        argument_schema_validity=(
            parsed_argument_count / proposal.raw_argument_count
            if proposal.raw_argument_count
            else 0.0
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
    )
    return PatientArgumentGraph(
        graph_id=f"argument:{case_id}",
        nodes=tuple(nodes),
        relations=tuple(relations),
        quality=quality,
    )


def _grounded_extension(
    argument_ids: set[str],
    attacks: set[tuple[str, str]],
) -> set[str]:
    attackers = {
        argument_id: {
            source_id
            for source_id, target_id in attacks
            if target_id == argument_id
        }
        for argument_id in argument_ids
    }
    current: set[str] = set()
    while True:
        acceptable = {
            argument_id
            for argument_id in argument_ids
            if all(
                any(
                    (defender_id, attacker_id) in attacks
                    for defender_id in current
                )
                for attacker_id in attackers[argument_id]
            )
        }
        if acceptable == current:
            return current
        current = acceptable


def resolve_argument_graph(
    graph: PatientArgumentGraph,
    proposal: ReasonerProposal,
    verifier: VerifierReport,
) -> SymbolicResolution:
    node_index = {node.argument_id: node for node in graph.nodes}
    non_diagnosis_ids = {
        node.argument_id
        for node in graph.nodes
        if node.node_type != "diagnosis"
    }
    non_diagnosis_attacks = {
        (relation.source_id, relation.target_id)
        for relation in graph.relations
        if relation.relation != RelationType.SUPPORTS
        and relation.source_id in non_diagnosis_ids
        and relation.target_id in non_diagnosis_ids
    }
    grounded = _grounded_extension(non_diagnosis_ids, non_diagnosis_attacks)
    rejected_non_diagnosis = {
        target_id
        for source_id, target_id in non_diagnosis_attacks
        if source_id in grounded
    }

    candidate_scores: dict[str, int] = {}
    candidate_supports: dict[str, tuple[str, ...]] = {}
    candidate_has_diagnostic_support: dict[str, bool] = {}
    for candidate in proposal.candidates:
        accepted_supports = tuple(
            argument.argument_id
            for argument in candidate.arguments
            if argument.argument_id in grounded
        )
        candidate_supports[candidate.candidate_id] = accepted_supports
        distinct_supports = {
            (
                node_index[argument_id].scheme,
                tuple(sorted(node_index[argument_id].evidence_ids)),
            ): node_index[argument_id].priority
            for argument_id in accepted_supports
        }
        candidate_scores[candidate.candidate_id] = sum(
            distinct_supports.values()
        )
        candidate_has_diagnostic_support[candidate.candidate_id] = any(
            node_index[argument_id].scheme
            in {
                ArgumentScheme.CLINICAL_SIGN,
                ArgumentScheme.DIAGNOSTIC_CRITERION,
            }
            for argument_id in accepted_supports
        )

    accepted_counterarguments = {
        node.argument_id
        for node in graph.nodes
        if node.node_type == "counterargument" and node.argument_id in grounded
    }
    directly_blocked = {
        relation.target_id
        for relation in graph.relations
        if relation.source_id in accepted_counterarguments
        and relation.target_id.startswith("D")
    }
    if proposal.abstain or verifier.abstain:
        directly_blocked.update(candidate_scores)

    eligible = {
        candidate_id: score
        for candidate_id, score in candidate_scores.items()
        if score > 0
        and candidate_has_diagnostic_support[candidate_id]
        and candidate_id not in directly_blocked
    }
    selected_candidate_id = ""
    if eligible:
        highest_score = max(eligible.values())
        leaders = sorted(
            candidate_id
            for candidate_id, score in eligible.items()
            if score == highest_score
        )
        if len(leaders) == 1:
            selected_candidate_id = leaders[0]

    diagnosis_ids = {
        candidate.candidate_id for candidate in proposal.candidates
    }
    accepted = set(grounded)
    rejected = set(rejected_non_diagnosis)
    undecided = non_diagnosis_ids - accepted - rejected
    if selected_candidate_id:
        accepted.add(selected_candidate_id)
        rejected.update(diagnosis_ids - {selected_candidate_id})
    else:
        rejected.update(
            candidate_id
            for candidate_id in diagnosis_ids
            if candidate_scores.get(candidate_id, 0) == 0
            or not candidate_has_diagnostic_support.get(candidate_id, False)
            or candidate_id in directly_blocked
        )
        undecided.update(diagnosis_ids - rejected)

    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in proposal.candidates
    }
    selected_diagnosis = (
        candidate_by_id[selected_candidate_id].diagnosis
        if selected_candidate_id
        else ""
    )
    trace = [
        (
            f"{candidate.candidate_id} ({candidate.diagnosis}): "
            f"accepted_support={','.join(candidate_supports[candidate.candidate_id]) or 'none'}; "
            f"scheme_score={candidate_scores[candidate.candidate_id]}; "
            f"diagnostic_support={candidate_has_diagnostic_support[candidate.candidate_id]}."
        )
        for candidate in proposal.candidates
    ]
    if selected_diagnosis:
        trace.append(
            f"Accepted {selected_candidate_id}: {selected_diagnosis} is the unique "
            "highest-priority undefeated best explanation."
        )
    else:
        trace.append(
            "No unique undefeated best explanation remained; the resolver abstained."
        )

    return SymbolicResolution(
        resolver_id="preference-grounded-bipolar-v1",
        selected_diagnosis=selected_diagnosis,
        abstained=not bool(selected_diagnosis),
        accepted_argument_ids=tuple(sorted(accepted)),
        rejected_argument_ids=tuple(sorted(rejected)),
        undecided_argument_ids=tuple(sorted(undecided)),
        candidate_scores=dict(sorted(candidate_scores.items())),
        trace=tuple(trace),
    )


def render_resolution_trace(
    graph: PatientArgumentGraph,
    resolution: SymbolicResolution,
) -> str:
    node_index = {node.argument_id: node for node in graph.nodes}
    lines = [
        f"Resolver: {resolution.resolver_id}",
        *resolution.trace,
        "Accepted arguments:",
    ]
    lines.extend(
        (
            f"{argument_id} | {node_index[argument_id].scheme.value} | "
            f"{node_index[argument_id].conclusion} | "
            f"evidence={','.join(node_index[argument_id].evidence_ids) or 'none'}"
        )
        for argument_id in resolution.accepted_argument_ids
        if argument_id in node_index
    )
    lines.append("Rejected arguments:")
    lines.extend(
        (
            f"{argument_id} | {node_index[argument_id].scheme.value} | "
            f"{node_index[argument_id].conclusion}"
        )
        for argument_id in resolution.rejected_argument_ids
        if argument_id in node_index
    )
    lines.append("Undecided arguments:")
    lines.extend(
        (
            f"{argument_id} | {node_index[argument_id].scheme.value} | "
            f"{node_index[argument_id].conclusion}"
        )
        for argument_id in resolution.undecided_argument_ids
        if argument_id in node_index
    )
    return "\n".join(lines)
